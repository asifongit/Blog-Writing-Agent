from __future__ import annotations

import json
import operator
import re
import sqlite3
import uuid
from pathlib import Path
from typing import TypedDict, List, Annotated, Literal

from pydantic import BaseModel, Field, model_validator
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Missing dependency for SQLite checkpointing. "
        "Install with: pip install langgraph-checkpoint-sqlite"
    )

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# ── Database setup ────────────────────────────────────────────────────────────

DB_PATH = "blog.db"

_conn = sqlite3.connect(DB_PATH, check_same_thread=False)

# Metadata table for sidebar listing
_conn.execute("""
    CREATE TABLE IF NOT EXISTS blog_history (
        thread_id  TEXT PRIMARY KEY,
        topic      TEXT NOT NULL,
        blog_title TEXT NOT NULL,
        audience   TEXT,
        tone       TEXT,
        final_md   TEXT,
        tasks_json TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")
_conn.commit()

checkpointer = SqliteSaver(_conn)


# ── Models ────────────────────────────────────────────────────────────────────

class Task(BaseModel):
    id: int
    title: str

    goal: str = Field(
        ...,
        description="One sentence describing what the reader should be able to do/understand after this section.",
    )

    bullets: List[str] = Field(
        ...,
        min_length=3,
        max_length=5,
        description="3-5 concrete, non-overlapping subpoints to cover in this section.",
    )

    target_words: int = Field(
        ..., description="Target word count for this section (120-450)."
    )

    section_type: Literal[
        "intro", "core", "example", "checklist", "common_mistakes", "conclusion"
    ] = Field(
        ...,
        description="Use 'common_mistakes' exactly once in the plan.",
    )


class Plan(BaseModel):
    blog_title: str = Field(
        ...,
        description="The full title of the blog post. This field MUST be named 'blog_title' in the JSON output.",
    )
    audience: str = Field(..., description="Who this blog is for.")
    tone: str = Field(..., description="Writing tone (e.g., practical, crisp.)")
    tasks: List[Task]

    @model_validator(mode="before")
    @classmethod
    def _remap_name_to_blog_title(cls, data: dict) -> dict:
        """Some models return 'name' instead of 'blog_title' — normalise it."""
        if isinstance(data, dict) and "name" in data and "blog_title" not in data:
            data["blog_title"] = data.pop("name")
        return data


class State(TypedDict):
    topic: str
    plan: Plan
    sections: Annotated[List[str], operator.add]
    final: str


# ── LLM ──────────────────────────────────────────────────────────────────────

llm = ChatGroq(model="llama-3.3-70b-versatile")


# ── Nodes ─────────────────────────────────────────────────────────────────────

def orchestrator(state: State) -> dict:
    plan = llm.with_structured_output(Plan).invoke(
        [
            SystemMessage(
                content=(
                    "You are a senior technical writer and developer advocate. Your job is to produce a "
                    "highly actionable outline for a technical blog post.\n\n"
                    "CRITICAL RULE — TOPIC ADHERENCE:\n"
                    "- The blog MUST be written EXACTLY about the topic the user provided.\n"
                    "- Do NOT switch to a different, more specific, or tangentially related topic.\n"
                    "- If the topic is broad (e.g. 'computer'), write a broad overview blog about that exact subject.\n"
                    "- Do NOT invent a narrower subtopic (e.g. do not turn 'computer' into 'Rust LRU Cache').\n\n"
                    "Hard requirements:\n"
                    "- Create 5–7 sections (tasks) that fit a technical blog.\n"
                    "- Each section must include:\n"
                    "  1) goal (1 sentence: what the reader can do/understand after the section)\n"
                    "  2) 3–5 bullets that are concrete, specific, and non-overlapping\n"
                    "  3) target word count (120–450)\n"
                    "- Include EXACTLY ONE section with section_type='common_mistakes'.\n\n"
                    "Make it technical (not generic):\n"
                    "- Assume the reader is a developer; use correct terminology.\n"
                    "- Prefer design/engineering structure: problem → intuition → approach → implementation → "
                    "trade-offs → testing/observability → conclusion.\n"
                    "- Bullets must be actionable and testable (e.g., 'Show a minimal code snippet for X', "
                    "'Explain why Y fails under Z condition', 'Add a checklist for production readiness').\n"
                    "- Explicitly include at least ONE of the following somewhere in the plan (as bullets):\n"
                    "  * a minimal working example (MWE) or code sketch\n"
                    "  * edge cases / failure modes\n"
                    "  * performance/cost considerations\n"
                    "  * security/privacy considerations (if relevant)\n"
                    "  * debugging tips / observability (logs, metrics, traces)\n"
                    "- Avoid vague bullets like 'Explain X' or 'Discuss Y'. Every bullet should state what "
                    "to build/compare/measure/verify.\n\n"
                    "Ordering guidance:\n"
                    "- Start with a crisp intro and problem framing.\n"
                    "- Build core concepts before advanced details.\n"
                    "- Include one section for common mistakes and how to avoid them.\n"
                    "- End with a practical summary/checklist and next steps.\n\n"
                    "Output must strictly match the Plan schema."
                )
            ),
            HumanMessage(content=f"Write a technical blog post strictly and only about this topic: {state['topic']}"),
        ]
    )
    return {"plan": plan}


def fanout(state: State):
    return [
        Send("worker", {"task": task, "topic": state["topic"], "plan": state["plan"]})
        for task in state["plan"].tasks
    ]


def worker(payload: dict) -> dict:
    task = payload["task"]
    topic = payload["topic"]
    plan = payload["plan"]

    bullets_text = "\n- " + "\n- ".join(task.bullets)

    section_md = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a senior technical writer and developer advocate. Write ONE section of a technical blog post in Markdown.\n\n"
                    "Hard constraints:\n"
                    "- Follow the provided Goal and cover ALL Bullets in order (do not skip or merge bullets).\n"
                    "- Stay close to the Target words (±15%).\n"
                    "- Output ONLY the section content in Markdown (no blog title H1, no extra commentary).\n\n"
                    "Technical quality bar:\n"
                    "- Be precise and implementation-oriented (developers should be able to apply it).\n"
                    "- Prefer concrete details over abstractions: APIs, data structures, protocols, and exact terms.\n"
                    "- When relevant, include at least one of:\n"
                    "  * a small code snippet (minimal, correct, and idiomatic)\n"
                    "  * a tiny example input/output\n"
                    "  * a checklist of steps\n"
                    "  * a diagram described in text (e.g., 'Flow: A -> B -> C')\n"
                    "- Explain trade-offs briefly (performance, cost, complexity, reliability).\n"
                    "- Call out edge cases / failure modes and what to do about them.\n"
                    "- If you mention a best practice, add the 'why' in one sentence.\n\n"
                    "Markdown style:\n"
                    "- Start with a '## <Section Title>' heading.\n"
                    "- Use short paragraphs, bullet lists where helpful, and code fences for code.\n"
                    "- Avoid fluff. Avoid marketing language.\n"
                    "- If you include code, keep it focused on the bullet being addressed.\n"
                )
            ),
            HumanMessage(
                content=(
                    f"Blog: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Topic: {topic}\n\n"
                    f"Section: {task.title}\n"
                    f"Section type: {task.section_type}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Bullets:{bullets_text}\n"
                )
            ),
        ]
    ).content.strip()

    return {"sections": [section_md]}


def reducer(state: State) -> dict:
    title = state["plan"].blog_title
    body = "\n\n".join(state["sections"]).strip()
    final_md = f"# {title}\n\n{body}\n"

    # save .md file
    safe_title = re.sub(r"[^\w\s-]", "", title.lower())
    filename = safe_title.replace(" ", "_") + ".md"
    Path(filename).write_text(final_md, encoding="utf-8")

    return {"final": final_md}


# ── Graph ─────────────────────────────────────────────────────────────────────

g = StateGraph(State)
g.add_node("orchestrator", orchestrator)
g.add_node("worker", worker)
g.add_node("reducer", reducer)

g.add_edge(START, "orchestrator")
g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile(checkpointer=checkpointer)


# ── Public API ────────────────────────────────────────────────────────────────

def generate_blog(topic: str) -> dict:
    """
    Run the blog-writing agent and persist the result to blog.db.

    Returns a dict with keys: topic, plan, sections, final, thread_id.
    """
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    result = app.invoke({"topic": topic, "sections": []}, config=config)

    plan: Plan = result["plan"]

    # Serialise tasks for later retrieval (no need to re-run the graph)
    tasks_data = [
        {
            "id": t.id,
            "title": t.title,
            "goal": t.goal,
            "bullets": t.bullets,
            "target_words": t.target_words,
            "section_type": t.section_type,
        }
        for t in plan.tasks
    ]

    _conn.execute(
        """
        INSERT INTO blog_history
            (thread_id, topic, blog_title, audience, tone, final_md, tasks_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            thread_id,
            topic,
            plan.blog_title,
            plan.audience,
            plan.tone,
            result["final"],
            json.dumps(tasks_data),
        ),
    )
    _conn.commit()

    result["thread_id"] = thread_id
    return result


def list_blogs() -> list[dict]:
    """Return all past blogs (newest first) for the sidebar."""
    cursor = _conn.execute(
        "SELECT thread_id, topic, blog_title, created_at "
        "FROM blog_history ORDER BY created_at DESC"
    )
    return [
        {
            "thread_id": row[0],
            "topic": row[1],
            "blog_title": row[2],
            "created_at": row[3],
        }
        for row in cursor.fetchall()
    ]


def get_blog(thread_id: str) -> dict | None:
    """Retrieve a saved blog by thread_id without re-running the graph."""
    cursor = _conn.execute(
        "SELECT topic, blog_title, audience, tone, final_md, tasks_json "
        "FROM blog_history WHERE thread_id = ?",
        (thread_id,),
    )
    row = cursor.fetchone()
    if not row:
        return None

    topic, blog_title, audience, tone, final_md, tasks_json = row
    tasks = [Task(**t) for t in json.loads(tasks_json)]
    plan = Plan(blog_title=blog_title, audience=audience, tone=tone, tasks=tasks)

    return {"topic": topic, "plan": plan, "final": final_md, "thread_id": thread_id}


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "RAG"
    result = generate_blog(topic)
    print(result["final"])
