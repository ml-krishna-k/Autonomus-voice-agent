# Quick Start Guide - NVIDIA Nemotron + EasyTurn

Get up and running in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- Windows OS
- Microphone connected
- ~5GB disk space for dependencies

## Installation (One-Time Setup)

### Option 1: Automated Setup (Recommended)

```powershell
cd "c:\Autonomus voice agent\Backend"
.\setup_nemotron.ps1
```

This will:
- Install all dependencies (~5-10 minutes)
- Verify installation
- Create template .env file

### Option 2: Manual Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create .env file
echo "OPENROUTER_API_KEY=your_key_here" > .env
```

## Configuration

Edit `.env` file and add your OpenRouter API key:

```bash
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxx
```

Get your API key from: https://openrouter.ai/keys

## Running the Agent

### Basic Usage (Recommended Settings)

```bash
python main_easyturn.py
```

This uses:
- **Balanced latency mode** (160ms) - good balance of speed and accuracy
- **CPU mode** - works on all systems
- **Full EasyTurn architecture** - intelligent turn-taking

### Say Hello!

When you see:
```
--- EasyTurn Agent Ready ---
Listening... (full-duplex mode)
```

Just start speaking! Say:
- "Hello, how are you?"
- "Tell me a joke"
- "What's the weather like?"

### Interrupting

The agent supports natural interruptions:

**Backchannels** (agent continues):
- "okay"
- "great"
- "uh-huh"

**Stop commands** (agent stops immediately):
- "stop"
- "wait"
- "enough"

**Barge-ins** (agent stops and listens):
- "What about the price?"
- "Tell me more about that"

### Quitting

Press `q` to quit the agent.

## Advanced Usage

### Ultra-Low Latency (Power Users)

```bash
python main_easyturn.py --latency ultra_low
```

- 80ms latency
- Extremely responsive
- Slightly lower accuracy

### High Accuracy Mode

```bash
python main_easyturn.py --latency accurate
```

- 560ms latency
- Higher accuracy
- Better for noisy environments

### GPU Acceleration

```bash
python main_easyturn.py --device cuda
```

Requires:
- NVIDIA GPU
- CUDA toolkit installed
- Will significantly speed up ASR processing

### Simple Mode (Fallback)

If EasyTurn causes issues, use simple turn-taking:

```bash
python main_easyturn.py --simple
```

This disables:
- Full-duplex operation
- Interruption handling
- Backchannel detection

## System Latency Breakdown

Understanding the ~300ms total latency:

```
User stops speaking
    ↓
[400ms] Silence detection (min_silence_to_speak_ms)
    ↓
[160ms] ASR processing (balanced mode)
    ↓
[100ms] LLM first token
    ↓
[50ms]  TTS startup
    ↓
System starts speaking
───────────────────
~710ms total
```

With optimizations (ultra_low mode, GPU, etc.), can achieve <300ms.

## Troubleshooting

### "NeMo not installed" error

```bash
pip install Cython
pip install nemo_toolkit[asr]
```

### "OPENROUTER_API_KEY not found"

Edit `.env` file and add your API key:
```bash
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxx
```

### Microphone not detected

```bash
# List available devices
python -m sounddevice
```

Then edit `audio/recorder.py` to select the correct device.

### Agent responds too quickly

Edit `main_easyturn.py`:
```python
min_silence_to_speak_ms=500  # Increased from 400
```

### Agent doesn't respond

Check logs for turn-taking decisions. May need to:
- Speak louder
- Reduce background noise
- Increase microphone sensitivity

### High CPU usage

Normal! ASR is computationally intensive. Options:
1. Use GPU: `--device cuda`
2. Use lower latency mode: `--latency ultra_low` (smaller chunks)
3. Close other applications

## Performance Tips

### For Best Responsiveness
```bash
python main_easyturn.py --latency ultra_low --device cuda
```

### For Best Accuracy
```bash
python main_easyturn.py --latency accurate
```

### For Low-Resource Systems
```bash
python main_easyturn.py --simple --latency balanced
```

## Next Steps

1. **Read the full documentation**: `README_NEMOTRON_EASYTURN.md`
2. **Explore EasyTurn**: `easyturn/README.md`
3. **Customize configuration**: Edit `main_easyturn.py`
4. **Tune thresholds**: See configuration section in main README

## Common Use Cases

### Customer Service Bot
```bash
# High accuracy, patient responses
python main_easyturn.py --latency accurate
```

Edit config for longer silence:
```python
min_silence_to_speak_ms=600
```

### Gaming Assistant
```bash
# Ultra-responsive
python main_easyturn.py --latency ultra_low --device cuda
```

### Accessibility Tool
```bash
# Balanced, reliable
python main_easyturn.py --latency balanced
```

## Getting Help

1. Check `README_NEMOTRON_EASYTURN.md` for detailed documentation
2. Review `easyturn/README.md` for turn-taking specifics
3. Enable debug logging in `main_easyturn.py`:
   ```python
   logging.basicConfig(level=logging.DEBUG)
   ```

## Key Files Reference

| File | Purpose |
|------|---------|
| `main_easyturn.py` | Entry point for EasyTurn agent |
| `asr/nemotron_asr.py` | NVIDIA Nemotron ASR wrapper |
| `easyturn/backend_adapters.py` | Service adapters |
| `easyturn/dialogue_orchestrator.py` | Main orchestration |
| `requirements.txt` | Dependencies |
| `.env` | API keys and secrets |

---

**You're all set! Start talking to your agent! 🎙️**
