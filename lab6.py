from urllib.parse import quote
import os
import requests
import streamlit as st
from langchain.agents import create_agent

SYSTEM_PROMPT = (
    "You are a helpful 'What to Wear' assistant. When the user asks what to "
    "wear or what to do outside, call the get_weather_data tool to get the "
    "weather before answering. If the user does not mention a location, call the "
    "tool anyway without a location. A default location will be used. "
    "Never ask the user for their location. "
    "If the tool returns an error, tell the user the location could not be found."
)

DEFAULT_LOCATION = "Syracuse, NY"


# Weather data (reused from Lab 5)
# note: hard codes units to degrees Fahrenheit

def format_hour(raw_time):
    """wttr.in hourly times look like '0', '300', '1500' -> '12 AM', '3 AM', '3 PM'."""
    hour = int(raw_time) // 100
    suffix = "AM" if hour < 12 else "PM"
    hour_12 = hour % 12 or 12
    return f"{hour_12} {suffix}"


@st.cache_data(ttl=600)  # avoid hitting wttr.in on every Streamlit rerun
def fetch_weather(location):
    url = f"https://wttr.in/{quote(location)}?format=j1"
    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        raise Exception(f"wttr.in error: status {response.status_code}")
    try:
        data = response.json()
    except ValueError:
        # unknown locations come back as plain text, not JSON
        raise Exception(f"Could not find a location named {location}")

    # j1 has three top-level sections:
    #   current_condition -- one entry, conditions right now
    #   weather           -- three entries, one per day (min/max, astronomy, hourly)
    #   nearest_area      -- the location wttr.in actually matched
    current = data["current_condition"][0]
    today = data["weather"][0]
    area = data["nearest_area"][0]
    astronomy = today["astronomy"][0]

    matched_location = (
        f"{area['areaName'][0]['value']}, "
        f"{area['region'][0]['value']}, "
        f"{area['country'][0]['value']}"
    )

    # How the day changes: wttr.in gives 8 hourly entries (every 3 hours)
    hourly = []
    for h in today["hourly"]:
        hourly.append({
            "time": format_hour(h["time"]),
            "temp_F": float(h["tempF"]),
            "feels_like_F": float(h["FeelsLikeF"]),
            "description": h["weatherDesc"][0]["value"],
            "chance_of_rain_pct": int(h["chanceofrain"]),
            "chance_of_snow_pct": int(h["chanceofsnow"]),
            "wind_mph": float(h["windspeedMiles"]),
        })

    return {
        "requested_location": location,
        "matched_location": matched_location,
        "current": {
            "temperature_F": float(current["temp_F"]),
            "feels_like_F": float(current["FeelsLikeF"]),
            "description": current["weatherDesc"][0]["value"],
            "humidity_pct": int(current["humidity"]),
            "wind_mph": float(current["windspeedMiles"]),
            "wind_direction": current["winddir16Point"],
            "precip_inches": float(current["precipInches"]),
            "uv_index": int(current["uvIndex"]),
            "visibility_miles": float(current["visibilityMiles"]),
        },
        "today": {
            "high_F": float(today["maxtempF"]),
            "low_F": float(today["mintempF"]),
            "total_snow_cm": float(today["totalSnow_cm"]),
            "sunrise": astronomy["sunrise"],
            "sunset": astronomy["sunset"],
            "max_chance_of_rain_pct": max(h["chance_of_rain_pct"] for h in hourly),
            "max_chance_of_snow_pct": max(h["chance_of_snow_pct"] for h in hourly),
        },
        "hourly_forecast": hourly,
    }


# Part B: the tool. LangChain sends this docstring to the model as the
# tool's description, so it is part of the prompt.
def get_weather_data(location: str = "Syracuse, NY") -> dict:
    """Return current conditions and today's forecast for one location, in
    Fahrenheit: temperature, feels-like, description, humidity, wind, UV,
    today's high/low, sunrise/sunset, chances of rain and snow, and a
    3-hourly forecast for the rest of the day.

    Accepts a city ("Syracuse, NY", "Lima, Peru"), a landmark ("Eiffel Tower"),
    or a US ZIP code. Looks up one location per call; for several places,
    call it once per place. If no location is given, uses Syracuse, NY.
    Returns {"error": ...} if the location cannot be found.
    """
    try:
        return fetch_weather(location)
    except Exception as e:
        return {"error": str(e)}


# Part C: build the agent
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

agent = create_agent(
    model="openai:gpt-4.1-mini",
    tools=[get_weather_data],
    system_prompt=SYSTEM_PROMPT,
)


# UI
st.title("What to Wear Bot – LangChain")
st.write(
    "Enter a city and I'll check the weather and suggest what to wear "
    "and which outdoor activities fit the day."
)
city = st.text_input("City", placeholder="e.g. Syracuse, NY or Lima, Peru",
                     help="Leave blank to use Syracuse, NY")

if st.button("Get advice"):
    location = city.strip() or DEFAULT_LOCATION
    result = agent.invoke(
        {"messages": [{"role": "user", "content": f"What should I wear in {location}?"}]}
    )

    with st.chat_message("assistant"):
        st.write(result["messages"][-1].content)

    # Part D: look inside the agent
    with st.expander("Behind the scenes"):
        for message in result["messages"]:
            st.write(f"**{message.type}**")
            st.write(message.content)
            if message.type == "ai" and message.tool_calls:
                st.write(message.tool_calls)


# Part D answers (confirm against "Behind the scenes" after running)
# a. How many messages are there, and what is each one?
#    Expected for one city: 4 messages
#      1. human -- "What should I wear in <location>?"
#      2. ai    -- the model's request to call get_weather_data (empty content)
#      3. tool  -- the weather data returned by get_weather_data
#      4. ai    -- the final what-to-wear answer
#
# b. Which message is the model asking for the weather tool?
#    Message 2 (ai). Its content is empty; the request is in
#    message.tool_calls (tool name, arguments like {"location": ...}, and an id).
#
# c. In Lab 5, which lines of your code created each of those messages?
#    TODO: fill in with line numbers from your Lab5.py
#      1. human: line(s) ___
#      2. ai tool request: line(s) ___
#      3. tool result: line(s) ___
#      4. final ai: line(s) ___