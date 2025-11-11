# app.py
"""
Multi-Agent Travel Planner

Highlights:
- Clear separation of concerns (tools, agents, orchestration, UI)
- Simple global logger to display tool calls live in the sidebar
- Planner → Reviewer pipeline enforced before rendering any answer
- Minimal dependencies and straightforward control flow
"""

from __future__ import annotations

import os
import asyncio
import time
from typing import Callable, Dict, List, Optional, Any

import streamlit as st
from dotenv import load_dotenv
from tavily import TavilyClient

# ──────────────────────────────────────────────────────────────────────────────
# Environment & Globals
# ──────────────────────────────────────────────────────────────────────────────

load_dotenv()  # Loads variables from a local .env if present
os.environ.setdefault("OPENAI_LOG", "error")
os.environ.setdefault("OPENAI_TRACING", "false")

# Tool call logger: the UI sets this per request. The tool checks it and logs.
# Using a simple global makes this easy to teach and reason about.
TOOL_LOGGER: Optional[Callable[[Dict[str, Any]], None]] = None


def set_tool_logger(logger: Optional[Callable[[Dict[str, Any]], None]]) -> None:
    """Install or remove the UI logger used by tools to report activity."""
    global TOOL_LOGGER
    TOOL_LOGGER = logger


def log_tool_event(event: Dict[str, Any]) -> None:
    """If a logger is installed, send the event to the UI."""
    if TOOL_LOGGER is not None:
        try:
            TOOL_LOGGER(event)
        except Exception:
            # Logging should never break the app or the tool itself
            pass


def redact_for_logs(value: Any) -> Any:
    """
    Make sure we don't leak secrets and keep logs small.
    This is deliberately simple for teaching.
    """
    if isinstance(value, str):
        low = value.lower()
        if any(k in low for k in ("api_key", "token", "secret", "password")):
            return "[redacted]"
        return value if len(value) <= 300 else value[:120] + "… [truncated]"
    if isinstance(value, dict):
        return {k: ("[redacted]" if any(s in k.lower() for s in ("key", "token", "secret", "password"))
                    else redact_for_logs(v))
                for k, v in value.items()}
    if isinstance(value, list):
        return [redact_for_logs(v) for v in value]
    return value


# ──────────────────────────────────────────────────────────────────────────────
# Agent Framework Imports (provided by you)
# ──────────────────────────────────────────────────────────────────────────────
# These come from your own framework. We assume:
# - Agent: defines a model + instructions + optional tools
# - Runner.run(agent, input): executes an agent and returns an object with text
from agents import Agent, Runner, function_tool  # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# Tools
# ──────────────────────────────────────────────────────────────────────────────

@function_tool
def internet_search(query: str) -> str:
    """
    Internet search backed by Tavily.
    - Reads TAVILY_API_KEY from environment.
    - Sends simple log events before/after the call so the UI can show activity.
    """
    log_tool_event({"type": "call", "tool": "internet_search", "args": {"query": redact_for_logs(query)}})

    try:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            msg = "missing TAVILY_API_KEY in environment."
            log_tool_event({"type": "error", "tool": "internet_search", "error": msg})
            return f"Search error: {msg}"

        client = TavilyClient(api_key=api_key)
        response = client.search(query, max_results=3)

        items = response.get("results", [])
        lines = [f"- {it.get('title', 'N/A')}: {it.get('content', 'N/A')}" for it in items]
        output = "\n".join(lines) if lines else "No results found."

        log_tool_event({
            "type": "result",
            "tool": "internet_search",
            "preview": redact_for_logs(output[:400] + ("…" if len(output) > 400 else "")),
        })
        return output

    except Exception as e:
        log_tool_event({"type": "error", "tool": "internet_search", "error": str(e)})
        return f"Search error: {e}"

    finally:
        log_tool_event({"type": "end", "tool": "internet_search"})


# ──────────────────────────────────────────────────────────────────────────────
# Agents
# ──────────────────────────────────────────────────────────────────────────────

# BEGIN SOLUTION
REVIEWER_INSTRUCTIONS = """
You are the REVIEWER AGENT in a two-agent travel-planning system.

There are two agents:
1. Planner Agent — takes the user's vague travel prompt and creates a detailed, day-by-day itinerary.
2. Reviewer Agent (YOU) — validates and improves the Planner’s itinerary using live internet search.

You ONLY ever see the Planner’s itinerary as your input. You do NOT talk directly to the user.
Your job is to:
- Fact-check the itinerary with live internet search (via the `internet_search` tool).
- Identify feasibility issues.
- Suggest clearly structured fixes.
- Produce a revised, improved itinerary for the user.

CRITICAL CONSTRAINTS
- You MUST use the `internet_search` tool to validate key facts whenever possible:
  - Opening hours and days (e.g., is the museum open on that day and at that time?).
  - Approximate ticket prices and whether they are in line with the plan.
  - Travel time between cities / neighborhoods (e.g., train vs. car vs. walking).
  - Any major closures, “must-reserve” tickets, or obvious conflicts.
- You MUST NOT assume internet knowledge without checking when it matters.
- The Planner has NO TOOL ACCESS and may hallucinate details; your job is to correct that.

INPUT YOU RECEIVE
You receive a single string which is the Planner’s full itinerary, typically in Markdown, e.g.:

- Overview of destination, dates, budget, and interests.
- Day-by-day breakdown with times, activities, locations, and cost estimates.

OUTPUT FORMAT (VERY IMPORTANT)
Always respond in this exact, structured Markdown format:

1. "## Feasibility Check"
   - Brief paragraph summarizing how realistic the plan is overall.
   - Then bullet points like:
     - ✅ Items that are feasible as written.
     - ⚠️ Items that are risky / ambiguous.
     - ❌ Items that are clearly infeasible or wrong.

2. "## Delta List (Concrete Changes)"
   - A bullet list of specific edits you recommend, each tied to a day:
     - Format: `- [Day X] <original or short reference> → <proposed change>. Reason: <short reason>.`
   - Be concrete, NOT vague.
   - Example:
     - `[Day 2] Louvre 5–8 pm → 10 am–2 pm. Reason: museum closes earlier than 8 pm on that day.`
   - Include cost/budget adjustments if your fact-checking shows large differences.

3. "## Revised Itinerary (Post-Review)"
   - Rewrite the full itinerary that the user will actually see.
   - Keep the user’s constraints (dates, budget, pacing, interests).
   - Adjust times/locations to be feasible based on your tool calls.
   - Still present it as a readable day-by-day plan:
     - `### Day 1 – City Name`
       - Morning: ...
       - Afternoon: ...
       - Evening: ...
       - Estimated daily cost: ...
       - Logistics: ...
   - Incorporate the Deltas you described above.
   - Maintain clarity and friendliness, but stay concise.

HOW TO USE THE TOOL
- Call `internet_search(query: str)` whenever you need up-to-date information.
- Good query examples:
  - "opening hours Sagrada Familia Sunday 2025"
  - "train travel time Paris to Amsterdam"
  - "Alhambra ticket price 2025"
- Do not spam the tool; 3–10 targeted calls per itinerary are usually enough.
- If search results are inconclusive, say so explicitly and then make a reasonable, clearly-labeled assumption.

STYLE & TONE
- You are professional and concise.
- You never apologize for the Planner’s mistakes; you simply fix them.
- You DO NOT add disclaimers about being an AI system.
- You DO NOT ask questions back to the user (the UI doesn’t support that right now).

Your final message is what the user sees (plus, optionally, the app may show the raw planner output in an expandable panel),
so make sure your "Revised Itinerary" is self-contained and clear.
"""

PLANNER_INSTRUCTIONS = """
You are the PLANNER AGENT in a two-agent travel-planning system.

Your job:
- Take a vague travel prompt from a user (destination, duration, budget, interests, timing).
- Produce a clear, realistic and enjoyable day-by-day itinerary.
- The itinerary will then be passed to a Reviewer Agent who can access live internet search and fix issues.

CRITICAL CONSTRAINTS
- You DO NOT have internet or tool access. You must rely on your general world knowledge.
- You MUST NOT mention tools, APIs, or the Reviewer in your output.
- You MUST assume the Reviewer will later fact-check and adjust details such as exact opening hours and prices.

WHAT TO CONSIDER
- User constraints:
  - Dates or duration (e.g., "7 days in mid-June").
  - Budget (per trip or per day).
  - Interests (e.g., history, food, nature, nightlife, museums, hiking).
  - Pacing: 
    - If budget is low or trip is short, avoid excessive inter-city travel.
    - Include some rest / flexible time, not just wall-to-wall activities.
- General feasibility:
  - Cluster activities by neighborhood/city to minimize backtracking.
  - Avoid unrealistic city hops (e.g., three far-apart cities in one day).
  - Keep daily schedules within reasonable waking hours.

OUTPUT FORMAT (VERY IMPORTANT)
Always produce your plan in this structured Markdown format (NO extra sections):

1. "## Trip Overview"
   - Destination(s)
   - Total duration (number of days, dates if given)
   - High-level theme: what kind of trip this is
   - Very rough budget breakdown (e.g., lodging vs. food vs. activities vs. transport)

2. "## City / Region Clusters"
   - Bullet list showing how days are grouped by location.
   - Example:
     - "Days 1–3: Paris (historic core, food)"
     - "Days 4–5: Lyon (food and old town)"
     - "Days 6–7: Nice (coast and relaxation)"

3. "## Day-by-Day Itinerary"
   For EACH day, use the following structure:

   ### Day X – City or Region Name
   - **Morning (~09:00–12:00)**: short description of 1–2 activities, with approximate neighborhoods.
   - **Lunch (~12:00–13:30)**: style of food / area (e.g., "casual local bistro in the Latin Quarter").
   - **Afternoon (~14:00–18:00)**: 1–2 main activities, specifying major sights or experiences.
   - **Evening (~19:00–22:00)**: dinner + light activity (walk, viewpoint, bar, etc.).
   - **Logistics**: brief notes on transportation and tickets for that day (e.g., "Use metro line X", "Book museum tickets in advance").
   - **Estimated Daily Cost**:
     - Lodging (rough)
     - Food (rough)
     - Activities / tickets (rough)
     - Local transport (rough)
     - Total daily estimate

4. "## Budget & Pacing Summary"
   - Summarize estimated total trip cost vs the user’s stated budget.
   - Comment on pacing: which days are "light", "medium", or "heavy".
   - Mention any days that could optionally be swapped or simplified.

STYLE & TONE
- Be concrete but not overly verbose.
- Choose well-known sights and neighborhoods that a first-time traveler might enjoy.
- Assume the user is not an expert; explain things in a friendly, clear way.
- Do NOT hedge with long disclaimers; the Reviewer will handle fine-grained validation later.

REMINDER
- You produce ONLY the itinerary as described above.
- You do NOT call tools or browse the internet.
- You do NOT discuss the internal agent pipeline.
"""

reviewer_agent = Agent(
    name="Reviewer Agent",
    model="openai.gpt-4o",
    instructions=REVIEWER_INSTRUCTIONS.strip(),
    tools=[internet_search]
)

planner_agent = Agent(
    name="Planner Agent",
    model="openai.gpt-4o",
    instructions=PLANNER_INSTRUCTIONS.strip(),
)

# END SOLUTION


# ──────────────────────────────────────────────────────────────────────────────
# Orchestration Helpers
# ──────────────────────────────────────────────────────────────────────────────

def extract_text(result_obj: Any) -> str:
    """
    Pull a usable string from the Runner result in a tolerant way.
    Your Runner may expose final_output, text, or __str__.
    """
    return (
        getattr(result_obj, "final_output", None)
        or getattr(result_obj, "text", None)
        or str(result_obj)
    )


def run_planner(user_text: str) -> str:
    """Run the Planner and return its itinerary text."""
    result = asyncio.run(Runner.run(planner_agent, user_text))
    return extract_text(result)


def run_reviewer(plan_text: str) -> str:
    """Run the Reviewer on the planner’s output and return validated text."""
    result = asyncio.run(Runner.run(reviewer_agent, plan_text))
    return extract_text(result)


# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Travel Planner", page_icon="✈️")

st.title("✈️ Multi-Agent Travel Planner")
st.caption("Planner → Reviewer (with live tool calls in the sidebar)")

# Sidebar: session controls + examples + dev panel
with st.sidebar:
    st.header("Session")
    if st.button("🔄 Reset conversation"):
        st.session_state.clear()
        st.rerun()

    st.subheader("Try these prompts")
    st.code("Plan a week-long Europe trip for a student on a $1,500 budget who loves history and food")
    st.code("3-day Paris trip for art lovers with $800 budget")

    st.subheader("Developer view")
    show_tools = st.toggle("Show tool activity (live)", value=True)
    if show_tools:
        tool_expander = st.expander("🔧 Tool activity", expanded=True)
        tool_panel = tool_expander.container()
    else:
        tool_panel = st.container()  # inert sink

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []  # list[dict(role, content)]
if "meta" not in st.session_state:
    st.session_state.meta = []      # list[dict(trace)]

# Render history
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and i < len(st.session_state.meta):
            meta = st.session_state.meta[i]
            if meta:
                st.caption(meta.get("trace", ""))

# Chat input
user_input = st.chat_input("Describe your travel (destination, duration, budget, interests)…")

if user_input:
    # Add user message to history and render it
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.meta.append(None)
    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant output block
    with st.chat_message("assistant"):
        # Live “working…” text and progress bar
        live_msg = st.empty()
        progress = st.progress(0)

        # Per-request tool log (shown in the sidebar)
        tool_events: List[Dict[str, Any]] = []

        def ui_tool_logger(event: Dict[str, Any]) -> None:
            """Append an event and re-render the sidebar log."""
            tool_events.append(event)
            with tool_panel:
                st.markdown("**Recent tool calls**")
                for ev in tool_events[-60:]:  # last N entries
                    t = ev.get("tool", "unknown")
                    et = ev.get("type", "event")
                    if et == "call":
                        st.write(f"• **{t}** called with `{ev.get('args')}`")
                    elif et == "result":
                        st.write(f"• **{t}** result preview:\n\n> {ev.get('preview')}")
                    elif et == "error":
                        st.error(f"• **{t}** error: {ev.get('error')}")
                    elif et == "end":
                        st.write(f"• **{t}** finished")

        # Install the logger so tools can report to the sidebar
        set_tool_logger(ui_tool_logger)

        try:
            # Optional: clear sidebar panel on each run
            with tool_panel:
                st.empty()

            # Step 1: Planner
            with st.status("🧭 Planner Agent: generating itinerary…", expanded=True) as status:
                live_msg.markdown("🧭 Planner Agent is creating your itinerary…")
                plan_text = run_planner(user_input)
                progress.progress(40)
                status.update(label="🔎 Reviewer Agent: validating with live searches…", state="running")

            # Step 2: Reviewer (tool calls will appear live in sidebar)
            live_msg.markdown("🔎 Reviewer Agent is validating the plan with live searches…")
            review_text = run_reviewer(plan_text)
            progress.progress(90)

            # Completed
            live_msg.markdown("✅ Validation complete. Rendering results…")
            time.sleep(0.2)
            progress.progress(100)

            # Final render: show only the validated result, with the raw plan expandable
            st.info("🤖 **Reviewer Agent** (validated)")
            st.markdown(review_text)
            with st.expander("See raw plan from Planner Agent"):
                st.markdown(plan_text)

            # Save only the validated result to history
            st.session_state.messages.append({"role": "assistant", "content": review_text})
            st.session_state.meta.append({"trace": "Planner Agent → Reviewer Agent"})
            st.caption("Planner Agent → Reviewer Agent")

        except Exception as e:
            # Friendly error box
            live_msg.markdown("❌ Something went wrong.")
            err = f"⚠️ Error while processing your request:\n\n```\n{e}\n```"
            st.markdown(err)
            st.session_state.messages.append({"role": "assistant", "content": err})
            st.session_state.meta.append({"trace": "Runtime error."})

        finally:
            # Always remove the logger so it doesn't leak into the next request
            set_tool_logger(None)
