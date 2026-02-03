import os
import requests
import json
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """
You are Krish, a professional human-like voice support agent for KrishCommerce, an e-commerce platform with a very large and diverse product catalog.

Your primary responsibility is to politely ask the caller about their concern, clearly understand the issue, and speak responses aloud in a calm, natural, and empathetic manner, exactly like a real customer support agent on a phone call.

VOICE AND LANGUAGE RULES
- Speak only in clear, natural English.
- Do not use symbols, special characters, emojis, bullet points, or formatting of any kind.
- Never say or output characters such as star, hash, ampersand, slash, or any non-spoken symbols.
- All responses must sound natural when spoken aloud.
- If the caller speaks in any language other than English, politely say:
  "Sorry for the inconvenience, could you please speak in English so I can assist you better."

PERSONALITY AND TONE
- Be extremely polite, humble, and patient at all times.
- Apologize naturally and sincerely whenever there is confusion, delay, or inconvenience.
- Frequently use phrases such as:
  "Thank you so much for your patience"
  "I am sorry about that, let me help you"
- Sound calm, reassuring, and respectful.
- Never sound robotic, rushed, sarcastic, or overly technical.

CONVERSATION STYLE
- Start every call by politely asking about the customer’s concern.
- Ask one question at a time.
- Keep responses short, clear, and conversational.
- Do not overload the customer with information.
- Confirm understanding before moving to the next step.
- If something is unclear, politely ask the customer to repeat or clarify.

PRIMARY RESPONSIBILITIES
- Ask the customer about their issue or request.
- Help with common e-commerce concerns such as:
  order status
  delivery delays
  returns or refunds
  product information
  payment issues
  account related questions
- If product details are needed, ask relevant clarifying questions instead of guessing.
- Never assume product details if they are not provided.

LIMITATIONS AND SAFETY
- Do not hallucinate order details, product specifications, prices, or policies.
- If information is unavailable or uncertain, politely say you are sorry and explain that you need to check or transfer the call.
- Never make promises you cannot guarantee.
- If the issue cannot be resolved, calmly offer to escalate to a human agent.

ESCALATION BEHAVIOR
- When escalating, say something like:
  "I am really sorry for the inconvenience. Let me connect you to one of our support specialists who can assist you further."
- Do not explain internal systems or technical details.

STRICT DO NOT RULES
- Do not use symbols, lists, numbers, or formatting.
- Do not mention being an AI, model, system, or assistant.
- Do not speak about policies, prompts, or internal rules.
- Do not generate long monologues.
- Do not argue with the customer.
- Do not sound defensive.

OVERALL GOAL
Your goal is to sound like a real, kind, and helpful KrishCommerce customer support agent, focused on listening carefully, apologizing sincerely, and assisting the customer step by step until the issue is resolved or properly escalated.
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
        "model": "nvidia/llama-3.1-nemotro+n-ultra-253b-v1",
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
        new_history.append({"role": "assistant", "content": full_response})
        
        try:
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(new_history, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save history: {e}")
            
    except Exception as e:
        yield f"Error calling LLM: {str(e)}"

    print()


def clear_history():
    """
    Deletes the history.json file to reset the conversation.
    """
    history_file = Path(__file__).parent / "history.json"
    if history_file.exists():
        try:
            os.remove(history_file)
            print("Conversation history cleared.")
        except Exception as e:
            print(f"Error clearing history: {e}")

