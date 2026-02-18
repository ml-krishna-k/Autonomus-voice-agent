# Migration Guide: OpenRouter & LangChain Refactor

This guide documents the refactoring of the LLM backend from raw `requests` to a modular `LangChain` architecture, and the upgrade to NVIDIA Riva TTS.

## Overview
- **LLM**: Now uses `ChatOpenAI` via `langchain-openai`, pointing to OpenRouter.
- **Tools**: Implemented using `StructuredTool` and Pydantic schemas.
- **Agent**: A custom, deterministic loop in `llm/agent.py` handles tool execution and streaming.
- **TTS**: Upgraded to **NVIDIA Riva TTS** for low-latency, high-quality speech synthesis.
- **No Dependencies**: Removed `easyspeech`, `nemotron`, etc.

## File Structure

```
Backend/
├── llm/
│   ├── llm.py       # Configures ChatOpenAI and binds tools
│   ├── tools.py     # Defines ecommerce tools (add_to_cart, track_order, place_order)
│   └── agent.py     # Main execution loop (history -> LLM -> tools -> LLM)
├── tts/
│   ├── riva_tts.py  # NVIDIA Riva TTS client impl (Streaming & Non-blocking)
│   └── tts.py       # Legacy SherpaTTS (Unused)
├── main.py          # Updated to use llm.agent.run_agent and RivaTTS
└── requirements.txt # Updated with langchain-core, nvidia-riva-client
```

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **NVIDIA Riva Server**:
   You must have a running NVIDIA Riva server (usually via Docker).
   By default, the agent connects to `localhost:50051`.
   
   To change the URI, edit `main.py` or `tts/riva_tts.py`.

3. **Environment Variables**:
   Ensure `.env` contains:
   ```
   OPENROUTER_API_KEY=your_key_here
   ```

## Usage

Run the main voice agent:
```bash
python main.py
```

The agent now supports:
- Natural conversation with context.
- Tool calling.
- **Interruption during synthesis**: You can interrupt the agent even before it starts speaking.
- **Low Latency**: Uses threaded synthesis to fetch audio while the LLM is still generating text.
