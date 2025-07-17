# utils/llm_utils.py

import json
from typing import List

from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq

# Must match exactly the names in COLOR_NAMES from color_utils.py
ALLOWED_COLORS = ["red", "blue", "black", "white", "green", "yellow", "silver"]

# 1) A strict prompt: RETURN ONLY the JSON array, nothing else.
PROMPT_TEMPLATE = """\
You are given a user’s natural‐language request. Identify which colors the user wants to track
(from this exact list of options): ["red","blue","black","white","green","yellow","silver"].

**IMPORTANT**: Return *only* a valid JSON array (for example: ["black","white"]), with no extra text,
no code blocks, and no explanation. If none of those colors are mentioned, return an empty array [].

### Examples (exact, with no surrounding text or explanation):
"Show me red and blue cars"        → ["red","blue"]
"I only care about silver vehicles" → ["silver"]
"Detect dark cars"                  → []
"Track white, black, and green."    → ["white","black","green"]

Now, for the user request below, produce exactly the JSON array (no commentary, no extra formatting):

{user_input}
"""

# 2) Build a deterministic (temperature=0.0) LangChain/ChatGroq chain
prompt = PromptTemplate(
    input_variables=["user_input"],
    template=PROMPT_TEMPLATE
)

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
    # 1) Run the LLM. invoke() now returns a dict like {"text": "..."}.
    result = color_chain.invoke({"user_input": user_input})
    # 2) Extract the actual LLM output string and strip whitespace
    response = result.get("text", "").strip()

    # 3) Debug print: show exactly what the model returned
    print(f"\n[DEBUG] Raw LLM response: >>>{response}<<<\n")

    # 4) Attempt to parse as JSON
    try:
        colors = json.loads(response)
        # 5) Filter to only our allowed list
        valid = [c for c in colors if c in ALLOWED_COLORS]
        return valid
    except Exception as e:
        print(f"[DEBUG] JSON parse error: {e}")
        return []
