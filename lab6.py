from urllib.parse import quote
import requests
import os
import streamlit as st
from langchain.agents import create_agent


SYSTEM_PROMPT = (
    "You are a helpful 'What to Wear' assistant. When the user asks what to "
    "wear or what to do outside, call the get_current_weather tool to get the "
    "weather before answering. If the user does not mention a location, call the "
    "tool anyway without a location. A default location will be used. "
    "Never ask the user for their location"
)
 
# Part A: weather data function 
# note: hard codes units to degrees Fahrenheit

DEFAULT_LOCATION = "Syracuse, NY"

def format_hour(raw_time):
    """wttr.in hourly times look like '0', '300', '1500' -> '12 AM', '3 AM', '3 PM'."""
    hour = int(raw_time) // 100
    suffix = "AM" if hour < 12 else "PM"
    hour_12 = hour % 12 or 12
    return f"{hour_12} {suffix}"
 
 
@st.cache_data(ttl=600)  # avoid hitting wttr.in on every Streamlit rerun
def get_current_weather(location:str = DEFAULT_LOCATION):
    """Get current weather and forecast for a location from wttr.in."""
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

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

agent = create_agent(
    model = "openai:gpt-4.1-mini",
    tools = [get_current_weather],
    system_prompt = SYSTEM_PROMPT,
)


#UI
city = st.text_input("City")
if st.button("Get advice"):
    location = city.strip() or DEFAULT_LOCATION
    result = agent.invoke({
        "messages": [{"role": "user", "content": f"What should I wear in {location}?"}]
    })
    st.write(result["messages"][-1].content)