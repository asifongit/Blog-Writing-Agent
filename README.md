# Blog Writing Agent

An AI-powered blog generator built with LangGraph, Groq, and Streamlit.

![Project Dashboard Screenshot](assets/screenshot.png)

---

## What it does

- Takes a topic → generates a full technical blog post
- Saves every blog to a local SQLite database (`blog.db`)
- Sidebar lets you re-read any past blog with one click
- Download the blog as a `.md` file

---

## Project structure

```
BlogWritingAgent/
├── main.py        # Backend — LangGraph agent + database
├── frontend.py    # Frontend — Streamlit UI
├── blog.db        # Auto-created on first run
└── .env           # Your API keys (see below)
```

---

## Setup

**1. Install dependencies**
```bash
pip install streamlit langgraph langchain-groq langchain-core \
            langgraph-checkpoint-sqlite pydantic python-dotenv
```

**2. Create a `.env` file**
```
GROQ_API_KEY=your_groq_api_key_here
```

**3. Run the app**
```bash
streamlit run frontend.py
```

---

## Usage

1. Type a topic in the input box
2. Click **Generate Blog**
3. Wait ~30 seconds while sections are written in parallel
4. Read, expand the plan, or download the `.md` file
5. Click **✏️ New Blog** in the sidebar to start a new one
6. Click any past title in the sidebar to re-read it

---

## How it works

```
Topic → Orchestrator (plan) → Workers (sections, parallel) → Reducer (assemble) → Output
```

- **Orchestrator** — creates a structured blog plan (5–7 sections)  
- **Workers** — each section is written in parallel by the LLM  
- **Reducer** — assembles sections into the final markdown  
- **SqliteSaver** — checkpoints every run to `blog.db`
