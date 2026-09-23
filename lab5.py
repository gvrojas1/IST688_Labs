import json
from urllib.parse import quote
 
import requests
import streamlit as st
from openai import OpenAI
 
# Page setup

st.title(" Lab 5 – What to Wear Bot")
st.write(
    "Enter a city and I'll check the weather and suggest what to wear "
    "and which outdoor activities fit the day."
)
 
openai_api_key = st.secrets.get("OPENAI_API_KEY", "")
if not openai_api_key:
    st.error("Missing OPENAI_API_KEY in secrets.")
    st.stop()
 
client = OpenAI(api_key=openai_api_key)
MODEL = "gpt-4.1-mini"
DEFAULT_LOCATION = "Syracuse, NY"
 
 
# Part A: weather data function 
# note: hard codes units to degrees Fahrenheit

def format_hour(raw_time):
    """wttr.in hourly times look like '0', '300', '1500' -> '12 AM', '3 AM', '3 PM'."""
    hour = int(raw_time) // 100
    suffix = "AM" if hour < 12 else "PM"
    hour_12 = hour % 12 or 12
    return f"{hour_12} {suffix}"
 
 
@st.cache_data(ttl=600)  # avoid hitting wttr.in on every Streamlit rerun
def get_current_weather(location):
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
 
# Part B: tool definition for OpenAI
# location is NOT required, so the model can call the tool without one
# and it falls back to Syracuse, NY

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": (
                "Get current weather and today's forecast for a location. "
                "Use this whenever the user asks about weather, clothing, or "
                "outdoor activities."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": (
                            "City and state/country, e.g. 'Syracuse, NY' or 'Lima, Peru'. "
                            "Omit if the user did not specify a location."
                        ),
                    }
                },
                "required": [],
            },
        },
    }
]
 
SYSTEM_PROMPT = (
    "You are a helpful 'What to Wear' assistant. When the user asks what to "
    "wear or what to do outside, call the get_current_weather tool to get the "
    "weather before answering. If the user does not mention a location, call the "
    "tool anyway without a location. A default location will be used. "
    "Never ask the user for their location"
)
 
ADVICE_PROMPT = """Here is today's weather data (JSON):
 
{weather}
 
Using this data, give practical advice for today:
1. **What to wear** – specific clothing and layers. Consider how the temperature \
changes through the day (morning vs. afternoon vs. evening), the feels-like \
temperature, wind, and whether to bring an umbrella, sunglasses, or sunscreen (UV).
2. **Outdoor activities** – 3 to 4 activities that suit the conditions, including \
the best time of day to do them.
3. **Heads up** – any warnings (rain later, big temperature swing, high UV, strong wind).
 
Start by stating the matched location and a one-line weather summary. Keep it \
friendly and concise."""
 
 
# UI – not a chatbot: one input, one output

city = st.text_input("City", placeholder="e.g. Syracuse, NY or Lima, Peru", 
                     help = "Leave blank to use Syracuse, NY")
 
if st.button("Get my outfit advice"):
    if city.strip():
        user_request = f"What should I wear today in {city.strip()}, and what outdoor activities would be good?"
    else:
        # no location -> the tool call will fall back to Syracuse, NY
        user_request = "What should I wear today, and what outdoor activities would be good?"
 
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_request},
    ]
 
    # First call: let the model decide whether it needs the weather tool
    with st.spinner("Thinking..."):
        first_response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
 
    response_message = first_response.choices[0].message
 
    if response_message.tool_calls:
        tool_call = response_message.tool_calls[0]
        args = json.loads(tool_call.function.arguments or "{}")
        location = args.get("location") or DEFAULT_LOCATION
        if not city.strip(): 
            st.info(f"No city entered, so I'm using {DEFAULT_LOCATION} as the default location.")
 
        try:
            with st.spinner(f"Checking the weather in {location}..."):
                weather = get_current_weather(location)
        except Exception as e:
            st.error(str(e))
            st.stop()
 
        # Second call: give the weather to the LLM and ask for advice
        with st.spinner("Picking your outfit..."):
            second_response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You give clothing and outdoor activity advice based on weather data."},
                    {"role": "user", "content": ADVICE_PROMPT.format(weather=json.dumps(weather, indent=2))},
                ],
            )
 
        st.markdown(second_response.choices[0].message.content)
 
        with st.expander("See the weather data the bot used"):
            st.json(weather)
    else:
        # Model answered without needing the tool
        st.markdown(response_message.content)