import json
import os
from pathlib import Path
from typing import Generator, List, Dict, Any

from llm.provider import LangChainProvider
from llm.controller import ExecutionController
from llm.tools import registry

# Constant System Prompt
SYSTEM_PROMPT = """
You are Krish, a real-time AI voice assistant for the KrishCommerce application.

Core behavior:
- Speak naturally and concisely, as if talking to a human on a phone.
- Keep responses short, clear, and action-oriented.
- Do NOT use markdown, bullet points, emojis, or long explanations.
- Do NOT reveal internal reasoning to the user.

Commerce behavior:
- Help users browse products, track orders, manage carts, payments, and support issues.
- Ask one clarifying question only if absolutely required.
- Prefer confirming actions before executing purchases or cancellations.
- If you don’t know something, say so briefly and offer the next best step.

Voice constraints:
- Assume responses are read aloud via TTS.
- Avoid lists longer than three items.
- Avoid technical jargon unless the user is clearly technical.
- Be interruptible: never ramble.
"""

HISTORY_FILE = Path(__file__).parent / "history.json"

def load_history() -> List[Dict[str, Any]]:
    """Loads conversation history from JSON file."""
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load history: {e}")
            return []
    return []

def save_history(history: List[Dict[str, Any]]):
    """Saves conversation history to JSON file."""
    try:
        # We only save the last 20 turns to keep context manageable
        trimmed_history = history[-20:]
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(trimmed_history, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save history: {e}")

def clear_history():
    """Clears the history file."""
    if HISTORY_FILE.exists():
        try:
            os.remove(HISTORY_FILE)
            print("Conversation history cleared.")
        except Exception as e:
            print(f"Warning: Failed to clear history: {e}")

def run_agent(user_input: str) -> Generator[str, None, None]:
    """
    Main generator for the voice agent using the new Execution Controller.
    """
    # 1. Initialize Components
    # In a real app, you might want to instantiate these once globally, 
    # but for this script structure, we do it here.
    provider = LangChainProvider() # Uses default meta-llama/llama-3.1-8b-instruct
    controller = ExecutionController(provider, registry)
    
    # 2. Handle admin commands
    if user_input.lower().strip() == "clear":
        clear_history()
        yield "History cleared."
        return

    # 3. Load History
    history = load_history()
    
    # 4. Run Controller
    # The controller yields strings for TTS
    # It also manages the 'Turn' logic internally for tool calls
    
    # We need to capture the updated history from the controller's run
    # However, since controller yields strings, we need a way to get the state back.
    # The provided controller design in the previous step didn't explicitly return state.
    # We will modify the interaction slightly here or assume the controller modifies a passed list?
    # Actually, let's keep it simple: We will reconstruct the history based on what happened.
    # BUT, strict adherence to the new architecture is required.
    
    # Let's adjust usage:
    # The controller yields chunks. We wrap it to catch the chunks.
    # State management is a bit tricky with generators. 
    # Let's trust the controller to do its job for this turn.
    # BUT we need to persist the conversation.
    
    # Refined approach: 
    # We pass the history list to the controller. The controller appends to it in-place (because lists are mutable).
    # Then we save it.
    
    # Refined approach: 
    # We pass the history list to the controller.
    # We execute it once, yield the results, and capture them for history.
    
    # Update history with user input
    history.append({"role": "user", "content": user_input})
    
    # Run Controller
    # It yields content strings.
    runner = controller.run(user_input, history[:-1], SYSTEM_PROMPT) 
    
    full_response = ""
    for chunk in runner:
        full_response += chunk
        yield chunk
        
    history.append({"role": "assistant", "content": full_response})
    save_history(history)
