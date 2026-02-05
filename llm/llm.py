import os
import requests
import json
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
# Load environment variables
load_dotenv()

import atexit

def clear_history():
    """
    Clears the conversation history file.
    """
    history_file = Path(__file__).parent / "history.json"
    if history_file.exists():
        try:
            os.remove(history_file)
            print("Conversation history cleared.")
        except Exception as e:
            print(f"Warning: Failed to clear history: {e}")

# Register cleanup on exit
atexit.register(clear_history)

SYSTEM_PROMPT = """
You are Krish, a real-time AI voice assistant for the KrishCommerce application.

Core behavior:
- Speak naturally and concisely, as if talking to a human on a phone.
- Keep responses short, clear, and action-oriented.
- Do NOT use markdown, bullet points, emojis, or long explanations.
- Do NOT reveal internal reasoning to the user.
- If you need to think/reason, you MUST put it in <think> tags at the VERY BEGINNING of the response.
- The user will ONLY hear text outside of <think> tags.

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

Tone:
- Calm, confident, friendly, professional.
- Indian English neutral accent.
"""

def get_llm_response(prompt: str):
    """
    Send prompt to OpenRouter LLM and yield the response chunks incrementally.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        yield "Error: OPENROUTER_API_KEY not found."
        return

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Initialize messages with System Prompt
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Load existing history
    history_file = Path(__file__).parent / "history.json"
    if history_file.exists():
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                history = json.load(f)
                if isinstance(history, list):
                    messages.extend(history)
        except Exception as e:
            print(f"Warning: Failed to load history: {e}")

    # Append current user prompt
    messages.append({"role": "user", "content": prompt})
    
    payload = {
        "model": "allenai/olmo-3.1-32b-instruct",
        "max_tokens": 1000,
        "stream": True,
        "messages": messages
    }

    full_response = ""

    try:
        print(f"Sending streaming request to LLM with prompt length: {len(prompt)}")
        response = requests.post(url, headers=headers, data=json.dumps(payload), stream=True)
        
        if response.status_code != 200:
            yield f"Error: {response.status_code} - {response.text}"
            return
            
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data_str = line[6:]
                    if data_str.strip() == '[DONE]':
                        break
                    try:
                        data = json.loads(data_str)
                        if "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0]["delta"]
                            content = delta.get("content", "")
                            if content:
                                full_response += content 
                                yield content
                    except json.JSONDecodeError:
                        continue
        
        # Save updated history
        new_history = messages[1:] # Get everything after System
        cleaned_response = remove_think_tags(full_response)
        if cleaned_response: # only append if there is content left
            new_history.append({"role": "assistant", "content": cleaned_response})
        
        try:
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(new_history, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save history: {e}")
            
    except Exception as e:
        yield f"Error calling LLM: {str(e)}"

def remove_think_tags(text: str) -> str:
    """
    Removes <think>...</think> blocks from text, handling potential single/multi-line.
    Also handles the case where the closing tag is missing (though less common in final output).
    For streaming, we might need a more robust approach, but for now we apply this to the full text
    or try to apply it to chunks if they are large enough.
    
    Actually, for this specific LLM (Nemotron), better to just suppress it via system prompt modification
    if possible, but regex is safer.
    """
    import re
    # Remove complete think blocks
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Remove potential starting think tag if unclosed (risky if it's valid text, but unlikely)
    cleaned = re.sub(r'<think>.*', '', cleaned, flags=re.DOTALL)
    return cleaned.strip()


if __name__ == "__main__":
    # Test
    print("Testing Streaming LLM connection...")
    for chunk in get_llm_response("I want to go to Bangalore"):
        print(chunk, end="", flush=True)
    print()
