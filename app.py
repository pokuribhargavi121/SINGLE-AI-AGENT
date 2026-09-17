import os
import requests
import streamlit as st

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain.tools import tool
from langchain.agents import create_agent


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")


# ============================================================
# 2. STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Single AI Agent",
    page_icon="🤖",
    layout="centered"
)


# ============================================================
# 3. PAGE TITLE
# ============================================================

st.title("🤖 Single AI Agent")

st.write(
    "Ask questions, search the web, or check the current weather."
)

st.divider()


# ============================================================
# 4. CHECK API KEYS
# ============================================================

missing_keys = []

if not GROQ_API_KEY:
    missing_keys.append("GROQ_API_KEY")

if not TAVILY_API_KEY:
    missing_keys.append("TAVILY_API_KEY")

if not WEATHERSTACK_API_KEY:
    missing_keys.append("WEATHERSTACK_API_KEY")


if missing_keys:

    st.warning(
        "The following API keys are missing: "
        + ", ".join(missing_keys)
    )

    st.info(
        "Create a .env file in the same folder as app.py "
        "and add your API keys."
    )


# ============================================================
# 5. TAVILY SEARCH TOOL
# ============================================================

if TAVILY_API_KEY:

    Search_tool = TavilySearch(
        max_results=3,
        tavily_api_key=TAVILY_API_KEY
    )

else:

    Search_tool = TavilySearch(
        max_results=3
    )


# ============================================================
# 6. WEATHER TOOL
# ============================================================

@tool
def get_weather(city: str) -> str:
    """
    Fetch current weather information for a city.
    """

    api_key = os.getenv("WEATHERSTACK_API_KEY")

    # Check API key
    if not api_key:

        return "WEATHERSTACK_API_KEY is not set."


    # WeatherStack API URL
    url = (
        "http://api.weatherstack.com/current"
        f"?access_key={api_key}"
        f"&query={city}"
    )


    try:

        response = requests.get(
            url,
            timeout=15
        )

        response.raise_for_status()

        data = response.json()


    except requests.RequestException as e:

        return (
            "Could not connect to the weather service.\n"
            f"Error: {e}"
        )


    # Check weather response
    if "current" not in data:

        return (
            f"Could not fetch weather data for {city}.\n"
            f"Response: {data}"
        )


    # Get weather description
    descriptions = data["current"].get(
        "weather_descriptions",
        []
    )


    if descriptions:

        condition = descriptions[0]

    else:

        condition = "Unknown"


    # Get temperature
    temperature = data["current"].get(
        "temperature",
        "N/A"
    )


    # Get humidity
    humidity = data["current"].get(
        "humidity",
        "N/A"
    )


    # Return formatted weather
    return (
        f"City: {city}\n"
        f"Temperature: {temperature}°C\n"
        f"Weather: {condition}\n"
        f"Humidity: {humidity}%"
    )


# ============================================================
# 7. CREATE GROQ LLM
# ============================================================

@st.cache_resource
def create_llm():

    if not GROQ_API_KEY:

        return None


    llm = ChatGroq(

        model="openai/gpt-oss-20b",

        api_key=GROQ_API_KEY

    )


    return llm


llm = create_llm()


# ============================================================
# 8. CREATE AI AGENT
# ============================================================

@st.cache_resource
def create_ai_agent():

    if not GROQ_API_KEY:

        return None


    # Create Groq model
    llm = ChatGroq(

        model="openai/gpt-oss-20b",

        api_key=GROQ_API_KEY

    )


    # Available tools
    tools = [

        Search_tool,

        get_weather

    ]


    # Create agent
    agent = create_agent(

        model=llm,

        tools=tools

    )


    return agent


agent = create_ai_agent()


# ============================================================
# 9. SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# 10. SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🤖 Single AI Agent")


    st.write(
        "This AI agent can use multiple tools "
        "to answer your questions."
    )


    st.divider()


    st.subheader("Available Tools")


    st.write("🔎 **Tavily Search**")

    st.write(
        "Searches the web for current information."
    )


    st.write("🌤️ **WeatherStack**")

    st.write(
        "Gets current weather information."
    )


    st.divider()


    st.subheader("Example Questions")


    st.write(
        "• What is the latest news on AI?"
    )

    st.write(
        "• Find the current weather in Ongole"
    )

    st.write(
        "• What happened in technology today?"
    )


    st.divider()


    # Clear chat button
    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# 11. DISPLAY PREVIOUS MESSAGES
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# 12. CHAT INPUT
# ============================================================

user_input = st.chat_input(
    "Ask me anything..."
)


# ============================================================
# 13. PROCESS USER INPUT
# ============================================================

if user_input:

    # --------------------------------------------------------
    # Check whether agent exists
    # --------------------------------------------------------

    if agent is None:

        st.error(
            "GROQ_API_KEY is missing. "
            "Please add it to your .env file."
        )

        st.stop()


    # --------------------------------------------------------
    # Add user message to session state
    # --------------------------------------------------------

    st.session_state.messages.append({

        "role": "user",

        "content": user_input

    })


    # --------------------------------------------------------
    # Display user message
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.markdown(user_input)


    # --------------------------------------------------------
    # Generate AI response
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "🤔 AI Agent is thinking..."
        ):

            try:

                # Send request to agent
                response = agent.invoke({

                    "messages": [

                        {
                            "role": "user",

                            "content": user_input
                        }

                    ]

                })


                # Get final AI response
                final_answer = response[
                    "messages"
                ][-1].content


                # Display response
                st.markdown(
                    final_answer
                )


                # Save response
                st.session_state.messages.append({

                    "role": "assistant",

                    "content": final_answer

                })


            except Exception as e:

                error_message = (
                    "❌ An error occurred:\n\n"
                    f"{str(e)}"
                )


                st.error(
                    error_message
                )


                st.session_state.messages.append({

                    "role": "assistant",

                    "content": error_message

                })


# ============================================================
# 14. FOOTER
# ============================================================

st.divider()

st.caption(
    "Built with Streamlit • LangChain • Groq • Tavily • WeatherStack"
)