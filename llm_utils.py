# utils/llm_utils.py

import json
from typing import List

from langchain import PromptTemplate, LLMChain
from langchain_groq import ChatGroq   # ← Use Groq’s ChatGroq class instead of OpenAI

# Must match exactly the names in COLOR_NAMES from color_utils.py
ALLOWED_COLORS = ["red", "blue", "black", "white", "green", "yellow", "silver"]

# 1) Prompt template (unchanged). Make sure there are NO curly quotes or stray backticks.
PROMPT_TEMPLATE = """\
You are given a user’s natural-language request. Identify which colors the user wants to track
(from this exact list of options): ["red","blue","black","white","green","yellow","silver"].

Return a JSON array of lowercase color names exactly as they appear in the list.
If none of those colors are mentioned, return an empty JSON array [].

Examples:
- "Show me red and blue cars"        → ["red","blue"]
- "I only care about silver vehicles" → ["silver"]
- "Detect dark cars"                  → []
- "Track white, black, and green."    → ["white","black","green"]

Now parse this user request:
{user_input}
"""

# 2) Build a deterministic (temperature=0.0) LangChain/ChatGroq chain
prompt = PromptTemplate(
    input_variables=["user_input"],
    template=PROMPT_TEMPLATE
)

# Instantiate Groq’s LLM. You can pick any Groq‐hosted model, e.g., "llama-3.1-8b-instant".
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,       # deterministic output
    max_tokens=None,
    timeout=None,
    max_retries=2
)
color_chain = LLMChain(llm=llm, prompt=prompt)

def extract_colors_from_text(user_input: str) -> List[str]:
    """
    Calls the Groq chain to parse which colors (from ALLOWED_COLORS) are mentioned in user_input.
    Returns a Python list of valid color names, e.g. ["red","silver"], or [] if none found.
    """
    response = color_chain.run(user_input=user_input).strip()
    try:
        colors = json.loads(response)
        # Filter out anything not in ALLOWED_COLORS
        valid = [c for c in colors if c in ALLOWED_COLORS]
        return valid
    except Exception:
        # If parsing fails or Groq didn’t return a valid JSON array, return empty list
        return []
