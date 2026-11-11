"""Lab: Multi-Agent Systems, the "What to Wear" crew (reference solution).

One manager agent coordinates three specialist agents (weather, clothing,
activities) through the OpenAI Agents SDK. A single-agent version of the same
bot runs alongside it so the two can be compared, and a triage agent shows the
other coordination pattern: handoffs.
"""
import asyncio
import os
import time
from urllib.parse import quote

import requests
import streamlit as st
from agents import Agent, ModelSettings, Runner, function_tool
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions

# The Agents SDK reads the key from the environment, like LangChain in Lab 6.
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

MODEL = "gpt-5-mini"
DEFAULT_LOCATION = "Syracuse, NY"


# ---------------------------------------------------------------------------
# Part B: the weather function from Lab 5. Students paste their own here.
# ---------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def get_current_weather(location):
    """Today's weather from wttr.in, trimmed to what the agents need."""
    url = f"https://wttr.in/{quote(location, safe='+,')}?format=j1"
    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        return {"error": f"Could not find weather for {location}"}
    data = response.json()
    current = data["current_condition"][0]
    today = data["weather"][0]
    return {
        "location": location,
        "now": {
            "description": current["weatherDesc"][0]["value"],
            "temperature_F": float(current["temp_F"]),
            "feels_like_F": float(current["FeelsLikeF"]),
            "wind_mph": int(current["windspeedMiles"]),
        },
        "high_F": float(today["maxtempF"]),
        "low_F": float(today["mintempF"]),
        "hourly": [
            {
                "hour": int(slot["time"]) // 100,
                "description": slot["weatherDesc"][0]["value"],
                "feels_like_F": float(slot["FeelsLikeF"]),
                "chance_of_rain_pct": int(slot["chanceofrain"]),
            }
            for slot in today["hourly"]
            if 6 <= int(slot["time"]) // 100 <= 21
        ],
    }


@function_tool
def get_weather(location: str = DEFAULT_LOCATION) -> dict:
    """Get today's weather for a location: conditions now, the high and low,
    and the forecast every three hours from 6 AM to 9 PM.

    Args:
        location: A city such as 'Syracuse, NY' or 'Lima, Peru'.
    """
    return get_current_weather(location)


# ---------------------------------------------------------------------------
# Part C: three specialists and a manager that calls them as tools.
# ---------------------------------------------------------------------------
weather_agent = Agent(
    name="Weather agent",
    instructions=(
        "Call get_weather for the location you are given. Summarize today's "
        "weather in four short lines: temperature and feels-like now, the high "
        "and low, rain chances through the day, and wind. Facts only, no advice."
    ),
    tools=[get_weather],
    model=MODEL,
)

stylist_agent = Agent(
    name="Stylist agent",
    instructions=(
        "You receive a weather summary. Recommend what to wear today as a "
        "bulleted list of 4 to 6 specific items. Go by the feels-like "
        "temperature, suggest layers if the day swings by 15 degrees or more, "
        "and add rain gear only if rain is likely. Never invent weather."
    ),
    model=MODEL,
)

activity_agent = Agent(
    name="Activity agent",
    instructions=(
        "You receive a weather summary. Suggest three outdoor activities that "
        "suit it, each with the best time of day. If it is a poor day to be "
        "outside, say so and suggest indoor options. Never invent weather."
    ),
    model=MODEL,
)

manager_agent = Agent(
    name="Manager",
    instructions=(
        "You coordinate a team that helps people decide what to wear and do "
        "today. First call weather_report with the user's location. Then give "
        "that weather summary to clothing_advice and activity_ideas. Combine "
        "their answers into three sections: **Weather today**, **What to "
        "wear**, **Outdoor activities**. Stay under 250 words. If the input is "
        "clearly not a place, ask for a city in one sentence instead."
    ),
    tools=[
        weather_agent.as_tool(
            tool_name="weather_report",
            tool_description="Look up and summarize today's weather for a location.",
        ),
        stylist_agent.as_tool(
            tool_name="clothing_advice",
            tool_description="Recommend clothing for a given weather summary.",
        ),
        activity_agent.as_tool(
            tool_name="activity_ideas",
            tool_description="Suggest outdoor activities for a given weather summary.",
        ),
    ],
    model=MODEL,
)

# ---------------------------------------------------------------------------
# Part E: the same bot as ONE agent, for comparison.
# ---------------------------------------------------------------------------
single_agent = Agent(
    name="Single agent",
    instructions=(
        "You help people decide what to wear and do today. Call get_weather "
        "for the user's location, then answer in three sections: **Weather "
        "today**, **What to wear** (4 to 6 items, layers if the day swings by "
        "15 degrees, rain gear only if rain is likely), and **Outdoor "
        "activities** (three ideas with the best time of day). Stay under 250 "
        "words. If the input is clearly not a place, ask for a city."
    ),
    tools=[get_weather],
    model=MODEL,
)

# ---------------------------------------------------------------------------
# Bonus: handoffs. Triage passes the conversation to a specialist, who answers.
# ---------------------------------------------------------------------------
packing_agent = Agent(
    name="Packing agent",
    handoff_description="Builds a packing list for a trip to another city.",
    instructions=prompt_with_handoff_instructions(
        "You have just taken over from the triage agent, and you answer the "
        "user yourself. Always call get_weather for the trip destination "
        "first. Then write a packing list grouped into clothing, rain/sun "
        "gear, and extras, based on that forecast. Under 150 words."
    ),
    tools=[get_weather],
    # Force a tool call on the packing agent's first turn. Without this it
    # sometimes replies "Transferred to packing agent" and stops. The SDK
    # resets tool_choice after one tool call, so it then writes the list.
    model_settings=ModelSettings(tool_choice="required"),
    model=MODEL,
)

outfit_agent = manager_agent.clone(
    name="Outfit team",
    handoff_description="Advice on what to wear and do today in one place.",
)

triage_agent = Agent(
    name="Triage agent",
    instructions=(
        "Decide what the user needs. Hand off to the Packing agent for trips "
        "and packing questions, and to the Outfit team for what to wear or do "
        "today. Do not answer yourself."
    ),
    handoffs=[packing_agent, outfit_agent],
    model=MODEL,
)


@st.cache_resource
def get_event_loop():
    """One event loop for the whole app. Runner.run_sync starts a new loop on
    every Streamlit rerun, but the SDK's OpenAI client stays tied to the first
    one, so the second click would fail with "Event loop is closed"."""
    return asyncio.new_event_loop()


def run(agent, prompt):
    """Run an agent on the app's event loop and time it."""
    start = time.perf_counter()
    result = get_event_loop().run_until_complete(Runner.run(agent, prompt))
    return result, time.perf_counter() - start


def show_trace(result, seconds):
    """Part D: what happened inside the run."""
    usage = result.context_wrapper.usage
    with st.expander("Behind the scenes"):
        st.write(
            f"**Model requests:** {usage.requests} · **Tokens:** "
            f"{usage.total_tokens:,} · **Time:** {seconds:.1f}s · "
            f"**Answered by:** {result.last_agent.name}"
        )
        for item in result.new_items:
            if item.type == "tool_call_item":
                st.write(f"🔧 {item.agent.name} called **{item.raw_item.name}** "
                         f"with `{item.raw_item.arguments}`")
            elif item.type == "tool_call_output_item":
                st.write(f"↩️ result: {str(item.output)[:300]}")
            elif item.type == "handoff_output_item":
                st.write(f"➡️ handoff: {item.source_agent.name} → {item.target_agent.name}")
            elif item.type == "message_output_item":
                st.write(f"💬 {item.agent.name} wrote the answer")


# ---------------------------------------------------------------------------
# Part D answers
# a. The manager calls weather_report first, then clothing_advice and
#    activity_ideas (often both of those in the same turn).
# b. A summary, not the raw forecast. The weather agent wrote that text, and
#    the manager passes it on as input. Specialists only see what the manager
#    chooses to send them.
# c. Each as_tool call is a full agent run with its own model calls (the
#    weather agent makes two: one to call get_weather, one to summarize).
#    The manager also makes its own calls: deciding which tools to call,
#    then writing the final combined answer.
# ---------------------------------------------------------------------------


st.title('Lab: The "What to Wear" Crew')
st.write(
    "A manager agent coordinates three specialist agents. Compare it with a "
    "single agent, or try the triage agent, which hands off to a specialist."
)

mode = st.radio(
    "Which system?",
    ["Team (manager + 3 specialists)", "Single agent", "Bonus: triage with handoffs"],
)
location = st.text_input("City, or a trip question for the triage bonus", DEFAULT_LOCATION)

if st.button("Get advice"):
    agent = {
        "Team (manager + 3 specialists)": manager_agent,
        "Single agent": single_agent,
        "Bonus: triage with handoffs": triage_agent,
    }[mode]
    prompt = location if mode.startswith("Bonus") else f"What should I wear and do today in {location or DEFAULT_LOCATION}?"
    with st.spinner(f"{agent.name} is working..."):
        result, seconds = run(agent, prompt)
    with st.chat_message("assistant"):
        st.write(result.final_output)
    show_trace(result, seconds)
