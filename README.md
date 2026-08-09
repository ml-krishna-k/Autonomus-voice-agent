# Autonomous Voice Agent

A real-time voice assistant you can actually hold a conversation with. You talk, it listens, thinks, and talks back. It starts speaking before it has finished deciding what to say, which is what makes it feel like a conversation instead of a form submission.

Speech recognition and speech synthesis both run **locally on CPU**. Only the language model is a remote call. That means no per-minute speech API bill, no audio leaving your infrastructure, and the whole thing fits in a single container of about 1.4 GB.

```
    ┌────────┐   PCM 16 kHz    ┌──────────────────────────────────────┐
    │ Client │ ──────────────► │  VAD  →  Whisper  →  LLM  →  Piper   │
    │ (web,  │ ◄────────────── │  ▲                            │      │
    │  phone)│   PCM + events  │  └──── barge-in cancels ───────┘      │
    └────────┘                 └──────────────────────────────────────┘
```

## Table of contents

- [How it works](#how-it-works)
- [Quick start](#quick-start)
- [The WebSocket API](#the-websocket-api)
- [Configuration](#configuration)
- [Project layout](#project-layout)
- [Deployment](#deployment)
- [Performance and cost](#performance-and-cost)
- [Security](#security)
- [Troubleshooting](#troubleshooting)
- [Phase 2](#phase-2)

## How it works

A conversation turn passes through four stages. The thing to understand about the design is that every stage starts before the one before it has finished. A voice agent that waits politely for each step to complete feels broken even when all the individual steps are fast.

| Stage | Component | What it does |
|---|---|---|
| 1. Endpointing | [`app/segmenter.py`](app/segmenter.py) | Watches the incoming audio and works out when you have stopped talking. Uses a volume gate with a 300 ms pre-roll so the first sound of your first word doesn't get chopped off. |
| 2. Recognition | [`app/asr.py`](app/asr.py) | `faster-whisper` (`distil-small.en`, int8) turns the audio into text. The model loads once and is shared by everyone; transcriptions run on a thread pool behind a semaphore. |
| 3. Generation | [`app/llm.py`](app/llm.py) | Streams tokens back from any OpenAI-compatible endpoint. |
| 4. Synthesis | [`app/tts.py`](app/tts.py) | Piper turns text into speech. The voice stays loaded in memory rather than being reloaded for every sentence. |

Sitting between stages 3 and 4 is [`app/textproc.py`](app/textproc.py), which does two jobs on the token stream as it goes past.

**It strips out the reasoning.** The model is told to put its thinking inside `<think>` tags, and none of that should ever reach the speaker. The catch is that tokens arrive in arbitrary pieces, so a tag can show up split down the middle as `<thi` then `nk>`. The filter is a small state machine that holds back anything which might still turn out to be the start of a tag.

**It cuts the text into sentences.** The moment a complete sentence exists, it gets sent off to be spoken while the model carries on writing the next one. Sentence detection needs punctuation followed by a space, so prices like `3.50` and `Rs.499` stay in one piece instead of being read as two sentences.

Put together, this pipelines nicely: sentence 1 is playing out loud while sentence 2 is being synthesised and sentence 3 is still being written.

### Barge-in

Talk while the agent is talking and it shuts up straight away.

A turn is a single `asyncio` task. When the segmenter notices you have started speaking mid-response, it cancels that task, which unwinds generation, synthesis and playback all at once. The client gets told to throw away any audio it has buffered. Whatever the agent did manage to say out loud is saved to history, so on the next turn it knows how far it got before you cut in.

Set `ALLOW_BARGE_IN=false` if you would rather it finished its sentence.

## Quick start

### Docker, which is the easy path

```bash
cp .env.example .env          # then put your OPENROUTER_API_KEY in it
docker compose up --build
```

The first build takes a few minutes because it downloads the speech models and bakes them into the image. That is deliberate: containers then start without any download at all.

Open <http://localhost:8000> and you get a built-in test client. Click **Connect & talk**, allow microphone access, and start talking.

### Running it directly

You need Python 3.12.

```bash
uv venv --python 3.12 && .venv\Scripts\activate     # Windows
# python -m venv .venv && source .venv/bin/activate # macOS / Linux

uv pip install -r requirements.txt
python scripts/fetch_models.py
cp .env.example .env          # add your key

uvicorn app.server:app --reload
```

### Local microphone mode

This skips the network completely and drives the same engines straight from your sound card. It is the fastest way to tune VAD thresholds or play with the system prompt.

```bash
uv pip install -r requirements-dev.txt
python main.py
```

### Checking everything works

```bash
python scripts/smoke_test.py
```

It exercises the text handling, synthesises a sentence to `smoke_test.wav`, then feeds that audio back through recognition and tells you how many words survived the round trip. A low score nearly always means the voice sample rate is wrong.

## The WebSocket API

**This is what Phase 2 will plug into.** One socket is one conversation, and the session dies when the connection does.

```
ws://<host>:8000/ws
```

### Client to server

**Binary frames** are raw audio, streamed continuously:

| Property | Value |
|---|---|
| Encoding | PCM, signed 16-bit, little-endian |
| Channels | 1 (mono) |
| Sample rate | 16000 Hz (also confirmed in the `ready` message) |
| Frame size | Anything. 20 to 100 ms works well. |

**Text frames** are JSON control messages:

| Message | What it does |
|---|---|
| `{"type":"text","text":"..."}` | Send a user turn as text and skip recognition entirely. Handy for testing, and for wiring the agent into a chat interface. |
| `{"type":"end_audio"}` | Endpoint right now instead of waiting for the silence timer. This is your push-to-talk release. |
| `{"type":"reset"}` | Wipe the conversation history. |
| `{"type":"ping"}` | Keepalive. You get a `pong` back. |

### Server to client

**Binary frames** carry the synthesised speech at `output_sample_rate`, same encoding as above. They always arrive between `tts_start` and `tts_end`.

**Text frames** are JSON events:

| Event | Payload | What it means |
|---|---|---|
| `ready` | `session_id`, `input_sample_rate`, `output_sample_rate`, `barge_in` | Handshake. Always arrives first. |
| `speech_start` | | Someone started talking. Stop playing audio if you are playing any. |
| `speech_end` | `reason`: `silence` \| `max_duration` \| `client` | Utterance captured, recognition starting. |
| `transcript` | `text` | What the user said. |
| `no_speech` | | There was nothing transcribable in that audio. |
| `response_start` | | The model has started writing. |
| `token` | `text` | A piece of the reply, with reasoning already removed. |
| `tts_start` | `sample_rate` | Audio frames are about to follow. |
| `tts_end` | | That's all the audio for this turn. |
| `tts_cancel` | | **Throw away buffered audio now.** You get this on barge-in. |
| `response_end` | | Turn finished. |
| `error` | `message` | Something went wrong. The socket stays open. |

### The smallest possible client

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
ws.binaryType = 'arraybuffer';

ws.onmessage = (event) => {
  if (event.data instanceof ArrayBuffer) return playPCM(event.data);

  const msg = JSON.parse(event.data);
  if (msg.type === 'transcript')  console.log('user:', msg.text);
  if (msg.type === 'token')       process.stdout.write(msg.text);
  if (msg.type === 'tts_cancel')  stopPlayback();
};

// then just stream Int16Array PCM chunks:  ws.send(pcmChunk.buffer)
```

If you want a real one, [`clients/web/index.html`](clients/web/index.html) is about 200 lines of plain JavaScript covering microphone capture, resampling, scheduled playback and barge-in handling. No build step, no dependencies.

### Ops endpoints

| Route | Purpose |
|---|---|
| `GET /healthz` | Liveness. Returns 200 the moment the process is up. |
| `GET /readyz` | Readiness. Only returns 200 once the models are loaded and the API key is set. **This is the one your load balancer should watch.** |
| `GET /info` | Which models are active, sample rates, how many sessions are live. |
| `GET /docs` | OpenAPI browser. |

## Configuration

Everything comes from environment variables. [`.env.example`](.env.example) has the full list, but these are the ones you will actually touch:

| Variable | Default | Notes |
|---|---|---|
| `OPENROUTER_API_KEY` | | **Required.** |
| `LLM_BASE_URL` | `https://openrouter.ai/api/v1` | Any OpenAI-compatible endpoint works: vLLM, Ollama, Together. |
| `LLM_MODEL` | `allenai/olmo-3.1-32b-instruct` | |
| `HISTORY_MAX_TURNS` | `8` | **Your main cost lever.** More on this below. |
| `ASR_MODEL` | `distil-small.en` | `tiny.en` is roughly 3x faster, `small.en` is more accurate. |
| `ASR_MAX_CONCURRENCY` | `2` | How many transcriptions run at once. Raise it as you add CPU. |
| `TTS_VOICE` | `en_US-amy-low` | Has to exist in `models/`. See `scripts/fetch_models.py` for what's available. |
| `VAD_THRESHOLD` | `0.01` | Turn it up in a noisy room, down for softly spoken users. |
| `VAD_SILENCE_MS` | `700` | How long a pause has to be before the agent decides you're done. This is the biggest single knob on how responsive it *feels*. |
| `ALLOW_BARGE_IN` | `true` | |
| `SYSTEM_PROMPT` | commerce assistant | Change this to repurpose the agent entirely. |

### Why HISTORY_MAX_TURNS matters more than it looks

Chat APIs have no memory. Every turn re-sends the whole conversation from the start. So without a limit, turn 20 costs twenty times what turn 1 did, and your spend grows with the *square* of how long people talk to it.

`HISTORY_MAX_TURNS` puts a flat ceiling on that. Eight turns is a sensible default for task-focused voice work. Raise it if your users tend to refer back to things they said much earlier in the call.

## Project layout

```
app/
  server.py      FastAPI app, model loading at startup, health endpoints
  ws.py          WebSocket transport and per-connection state machine
  pipeline.py    Turn orchestration: recognise, generate, chunk, speak
  segmenter.py   Streaming VAD and utterance endpointing
  textproc.py    Think-tag stripping and sentence chunking
  asr.py         faster-whisper wrapper
  llm.py         Async streaming client for OpenAI-compatible APIs
  tts.py         Piper wrapper, voice kept in memory
  session.py     Per-session history with a bounded turn window
  config.py      Environment-driven settings

clients/web/     Reference browser client (no build step, no dependencies)
scripts/
  fetch_models.py  Downloads the speech models
  smoke_test.py    Offline end-to-end check
main.py          Local microphone mode
models/          Voice files live here (gitignored)
```

## Deployment

### What it needs

A **long-lived container or VM**. Not serverless.

Conversations hold a socket open for minutes at a time, the models sit at about 1 GB in memory, and loading them cold takes 30 to 60 seconds. None of that fits Lambda, Cloud Run in request mode, or Vercel Functions.

Fly.io, Railway, Render, a plain EC2 or Hetzner box, ECS/Fargate, or any Kubernetes cluster will all work fine.

### Sizing

| Users | vCPU | RAM | Notes |
|---|---|---|---|
| 1 to 2 | 2 | 2 GB | Development. |
| 3 to 5 | 4 | 4 GB | `ASR_MAX_CONCURRENCY=2` |
| 8 to 12 | 8 | 8 GB | `ASR_CPU_THREADS=6`, `ASR_MAX_CONCURRENCY=4` |

Scale by adding **replicas, not workers**. Every worker process loads its own copy of the models, so `--workers 4` costs you 4 GB before it has served a single request. Run one worker per container and add containers.

### Load balancing

Sessions live in memory and belong to a socket, so you don't need sticky sessions. A connection stays on one instance for its whole life by definition. Just make sure:

- WebSocket upgrades are allowed through
- The idle timeout is longer than your longest expected pause, so 60 seconds or more
- Health checks point at `/readyz` with a start period of at least 90 seconds

## Performance and cost

Measured on 4 vCPU with `distil-small.en` int8 and the `en_US-amy-low` voice:

| Stage | Typical |
|---|---|
| Deciding you've stopped talking | `VAD_SILENCE_MS`, 700 ms by default |
| Transcribing a 5 second utterance | 0.4 to 0.9 s |
| First token from the LLM | 0.3 to 1.2 s, mostly network |
| First audio out | another 0.2 to 0.4 s once the first sentence is complete |
| **What it feels like** | **roughly 1.5 to 2.5 s** |

Synthesis runs at about 15x real time, so once it starts it comfortably stays ahead of playback.

### Footprint

| Component | Size |
|---|---|
| Python base image | ~150 MB |
| Dependencies (onnxruntime, ctranslate2, numpy) | ~600 MB |
| Whisper `distil-small.en` int8 | ~200 MB |
| Piper voice | ~60 MB |
| **Total image** | **~1.4 GB** |
| **RAM sitting idle** | **~800 MB** |
| **Extra per concurrent turn** | **+150 to 250 MB** |

There is no PyTorch anywhere in here. `faster-whisper` runs on CTranslate2 and Piper on onnxruntime, which saves roughly 2 GB against a naive install.

### Cost

Infrastructure is a flat $30 to $70 a month for a 4 vCPU box, and that number doesn't move whether you serve one conversation or a thousand. Recognition and synthesis are free once the box is paid for.

The LLM is your only variable cost. Each turn sends the system prompt plus up to `HISTORY_MAX_TURNS` of context, and gets back up to `LLM_MAX_TOKENS`. Check your provider's current per-token pricing and multiply by how many turns you expect. The history window is what keeps that arithmetic linear instead of quadratic.

## Security

> [!WARNING]
> **A live API key was previously committed to this repository and pushed to GitHub.** It sits in the history of commits `74da881` and `d3dd789`. Deleting the file does not get rid of it, because published history is permanent.
>
> **Rotate the OpenRouter key now.** Then clean the history with [`git filter-repo`](https://github.com/newren/git-filter-repo) or [BFG](https://rtyley.github.io/bfg-repo-cleaner/) and force-push.

`.env` is untracked and gitignored as of this version. Use `.env.example` as your template and inject real secrets through your platform's secret manager rather than a file on disk.

Before you put this anywhere public, also think about:

- **Setting `CORS_ORIGINS`** to your actual domains. The `*` default is for development only.
- **Adding authentication.** `/ws` is wide open right now, so anyone who can reach it can spend your LLM budget. A token check in `ConnectionHandler.run()` before `accept()` is the obvious place for it.
- **Rate limiting** connections per IP. `MAX_SESSIONS` caps total concurrency but it is not an abuse control.
- **Serving over TLS.** Browsers refuse microphone access on anything other than HTTPS or localhost, so you need this anyway.

## Troubleshooting

**The voice sounds too fast, or like a chipmunk.**
Sample rate mismatch. The engine reads the correct rate out of `models/<voice>.onnx.json` instead of guessing, so this usually means that file is missing or doesn't match the `.onnx` next to it. Re-run `python scripts/fetch_models.py`.

**`/readyz` keeps returning 503.**
Look at which entry in `checks` is false. If `asr` or `tts` is false the models are either still loading or not there, so run the fetch script. If `llm_key` is false you haven't set `OPENROUTER_API_KEY`.

**It replies to silence, or interrupts itself.**
`VAD_THRESHOLD` is too low and it is probably hearing its own voice. Push it up to 0.02 or 0.03, and make sure your client has echo cancellation switched on. The reference client does.

**It takes too long to answer.**
Drop `VAD_SILENCE_MS` to 400 or 500. Go below about 300 and it will start cutting people off mid-sentence.

**Nothing gets transcribed at all.**
Almost always the audio format. The server wants 16 kHz mono **signed 16-bit little-endian** PCM. Send it float32 or 44.1 kHz and you get silence or nonsense. The `ready` message tells you the rate it expects.

**`piper` fails to install.**
Piper doesn't publish wheels for every Python version. Use Python 3.12, which is what the Dockerfile pins and what `.python-version` asks for.

## Phase 2

The service layer deliberately knows nothing about transport. `ConversationPipeline` only sees two callbacks, `emit` for events and `emit_audio` for audio. Anything that can produce 16 kHz PCM and consume events can drive this agent.

Sensible next moves, roughly easiest first:

- **Telephony.** Bridge Twilio Media Streams or LiveKit into `/ws`. Both already speak PCM over WebSocket, so the real work is resampling 8 kHz mu-law up to 16 kHz.
- **Tool calling.** Order lookups, cart changes and the rest belong inside `pipeline.respond()`, between generation and synthesis.
- **Persistent memory.** Swap `SessionStore` for Redis so conversations survive restarts and span multiple connections. It is a four-method interface.
- **Live captions.** Partial transcripts while someone is still speaking.
- **More voices and languages.** `TTSEngine` currently loads one voice. A dictionary of voices keyed by session would let users pick.
- **Metrics.** The pipeline already logs per-stage latency at `DEBUG`. Exporting those to Prometheus is a small job.

## License

Pick one before you publish. Note that the models carry their own terms: Whisper is MIT, and Piper voices are usually CC-BY-SA, so check the specific voice you ship.
