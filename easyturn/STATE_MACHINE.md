# EasyTurn Controller: State Machine & Decision Logic

## State Machine Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SYSTEM STATE MACHINE                         │
└─────────────────────────────────────────────────────────────────────┘

States:
  [IDLE]      - No active conversation
  [LISTENING] - Actively listening to user
  [PROCESSING]- LLM generating response
  [SPEAKING]  - TTS playing system response
  [INTERRUPTED] - Brief transition state after interruption


                            ┌──────────┐
                            │   IDLE   │
                            └────┬─────┘
                                 │ start()
                                 ▼
                         ┌───────────────┐
                    ┌────┤   LISTENING   │◄────┐
                    │    └───────┬───────┘     │
                    │            │             │
                    │            │ EasyTurn    │
                    │            │ → SPEAK     │
                    │            │             │
                    │            ▼             │
                    │    ┌───────────────┐    │
                    │    │  PROCESSING   │    │ Response
                    │    └───────┬───────┘    │ complete
                    │            │ LLM        │
                    │            │ generates  │
                    │            │ first token│
                    │            ▼            │
                    │    ┌───────────────┐   │
     User           │    │   SPEAKING    │───┘
     interrupts ────┼───►└───────┬───────┘
     (energy >      │            │
     threshold)     │            │ TTS completes
                    │            │ naturally
                    │            ▼
                    │    ┌───────────────┐
                    └───►│  INTERRUPTED  │
                         └───────┬───────┘
                                 │ (50ms pause)
                                 ▼
                         ┌───────────────┐
                         │   LISTENING   │
                         └───────────────┘
```

## Turn State Machine (EasyTurn Controller)

```
┌─────────────────────────────────────────────────────────────────────┐
│                      TURN STATE MACHINE                             │
│                   (Core Decision Logic)                             │
└─────────────────────────────────────────────────────────────────────┘

States:
  [HOLD]  - System is holding, waiting for user to finish
  [SPEAK] - System should respond now


Initial State: HOLD

                        ┌──────────┐
                   ┌───►│   HOLD   │◄───┐
                   │    └────┬─────┘    │
                   │         │          │
                   │         │          │
        Condition: │         │          │ Condition:
        User       │         │          │ System
        resumes    │         │          │ speaking
        speaking   │         │          │ AND user
        (interrupt)│         │          │ starts
                   │         │          │
                   │         │          │
                   │         ▼          │
                   │    ┌──────────┐    │
                   └────┤  SPEAK   ├────┘
                        └──────────┘
                             ▲
                             │
                        Condition:
                        All of:
                        • User stopped speaking
                        • Silence ≥ 400ms
                        • ASR confidence ≥ 0.6
                        • Token stability ≥ 0.8
                        • Text length ≥ 5 chars
                        • Stable for 200ms


Transition Conditions Detail:

HOLD → SPEAK:
  1. is_speaking = FALSE (no voice activity)
  2. silence_duration_ms ≥ MIN_SILENCE_TO_SPEAK_MS (400ms default)
  3. text_length ≥ MIN_TEXT_LENGTH (5 chars)
  4. asr_confidence ≥ MIN_CONFIDENCE_TO_SPEAK (0.6)
  5. token_stability ≥ STABILITY_THRESHOLD (0.8)
  6. time_since_text_change ≥ 150ms
  7. Condition stable for HYSTERESIS_WINDOW_MS (200ms)

SPEAK → HOLD:
  1. frame_energy > INTERRUPT_ENERGY_THRESHOLD (0.015)
     OR vad_prob > INTERRUPT_VAD_THRESHOLD (0.6)
  2. Condition stable for HYSTERESIS_WINDOW_MS (200ms)
```

## EasyTurn Controller Pseudocode

```
ALGORITHM: EasyTurn Controller Decision Loop
INPUT: acoustic_features, linguistic_features
OUTPUT: TurnDecision (HOLD or SPEAK)

// Called every 20-40ms

FUNCTION update(acoustic, linguistic):
    current_time = acoustic.timestamp_ms
    
    // 1. Update history buffers
    energy_history.append(acoustic.frame_energy)
    vad_history.append(acoustic.vad_prob)
    stability_history.append(linguistic.token_stability)
    
    // 2. Track text changes
    IF linguistic.partial_text != previous_text:
        last_text_change_time = current_time
        previous_text = linguistic.partial_text
    
    // 3. Determine if user is currently speaking
    is_speaking = is_user_speaking(acoustic)
    
    // 4. Evaluate desired state based on current state
    IF current_state == HOLD:
        desired_state = evaluate_hold_state(
            acoustic, linguistic, is_speaking, current_time
        )
    ELSE:  // current_state == SPEAK
        desired_state = evaluate_speak_state(
            acoustic, linguistic, is_speaking, current_time
        )
    
    // 5. Apply hysteresis to prevent flapping
    final_state, reason = apply_hysteresis(desired_state, current_time)
    
    // 6. Update state
    current_state = final_state
    
    // 7. Track speech end time
    IF NOT is_speaking AND last_speech_end == NULL:
        last_speech_end = current_time
    ELSE IF is_speaking:
        last_speech_end = NULL
    
    RETURN TurnDecision(
        state=final_state,
        confidence=calculate_confidence(acoustic, linguistic),
        timestamp=current_time,
        reason=reason
    )

FUNCTION is_user_speaking(acoustic):
    energy_speaking = acoustic.frame_energy > ENERGY_THRESHOLD_HIGH
    vad_speaking = acoustic.vad_prob > VAD_THRESHOLD_SPEAKING
    RETURN energy_speaking OR vad_speaking

FUNCTION evaluate_hold_state(acoustic, linguistic, is_speaking, time):
    // Still in HOLD (listening) state
    // Check if we should transition to SPEAK
    
    // User still speaking? Stay in HOLD
    IF is_speaking:
        RETURN HOLD
    
    // Not enough silence? Stay in HOLD
    silence_ms = acoustic.silence_duration_ms
    min_silence = MIN_SILENCE_TO_SPEAK_MS
    IF linguistic.text_length < 20:
        min_silence = MIN_SILENCE_SHORT_UTTERANCE_MS
    
    IF silence_ms < min_silence:
        RETURN HOLD
    
    // Text too short? Stay in HOLD
    IF linguistic.text_length < MIN_TEXT_LENGTH:
        RETURN HOLD
    
    // ASR not confident? Stay in HOLD
    IF linguistic.asr_confidence < MIN_CONFIDENCE_TO_SPEAK:
        RETURN HOLD
    
    // Tokens not stable? Stay in HOLD
    IF linguistic.token_stability < STABILITY_THRESHOLD:
        RETURN HOLD
    
    // Text still changing? Stay in HOLD
    IF last_text_change_time != NULL:
        time_since_change = time - last_text_change_time
        IF time_since_change < 150:
            RETURN HOLD
    
    // All conditions met → suggest SPEAK
    RETURN SPEAK

FUNCTION evaluate_speak_state(acoustic, linguistic, is_speaking, time):
    // In SPEAK state (system should be responding)
    // Check if user is interrupting
    
    // Use more sensitive thresholds for interruption
    interrupt_energy = acoustic.frame_energy > INTERRUPT_ENERGY_THRESHOLD
    interrupt_vad = acoustic.vad_prob > INTERRUPT_VAD_THRESHOLD
    
    IF interrupt_energy OR interrupt_vad:
        // User is interrupting → return to HOLD
        RETURN HOLD
    
    // No interruption → continue SPEAK
    RETURN SPEAK

FUNCTION apply_hysteresis(desired_state, time):
    // Prevent rapid state flapping
    // State must be stable for HYSTERESIS_WINDOW_MS before transitioning
    
    IF desired_state == current_state:
        // No change desired
        pending_state = NULL
        pending_state_start = NULL
        RETURN (current_state, "stable_" + current_state)
    
    // Desired state differs from current
    IF pending_state != desired_state:
        // New pending state - start tracking
        pending_state = desired_state
        pending_state_start = time
        RETURN (current_state, "pending_" + desired_state)
    
    // Pending state has been consistent
    elapsed_ms = time - pending_state_start
    
    IF elapsed_ms >= HYSTERESIS_WINDOW_MS:
        // Transition confirmed
        pending_state = NULL
        pending_state_start = NULL
        
        IF desired_state == HOLD:
            reason = "transition_to_HOLD_interrupt"
        ELSE:
            reason = "transition_to_SPEAK_eot"
        
        RETURN (desired_state, reason)
    
    // Still waiting for stability
    RETURN (current_state, "pending_" + desired_state + "_" + elapsed_ms + "ms")
```

## Orchestrator Integration Pseudocode

```
ALGORITHM: Full-Duplex Dialogue Orchestrator
INPUT: Continuous audio stream
OUTPUT: System speech responses

INITIALIZE:
    orchestrator = DialogueOrchestrator()
    easyturn = EasyTurnController()
    acoustic_extractor = AcousticExtractor()
    stability_tracker = TokenStabilityTracker()
    
    system_state = LISTENING
    turn_state = HOLD
    asr_buffer = []

// Main audio processing loop (called every 20-40ms)
FUNCTION process_audio_frame(audio_frame, timestamp):
    // 1. Extract acoustic features
    acoustic = acoustic_extractor.extract(audio_frame, timestamp)
    
    // 2. Get linguistic features from ASR
    linguistic = get_linguistic_features(timestamp)
    
    // 3. Run EasyTurn controller
    decision = easyturn.update(acoustic, linguistic)
    
    // 4. Handle state transitions
    IF decision.state != turn_state:
        handle_turn_transition(decision.state, turn_state)
    
    turn_state = decision.state

// ASR polling loop (runs independently at ~100ms)
ASYNC FUNCTION asr_polling_loop():
    WHILE system_running:
        partial_text, confidence = ASR.get_partial_result()
        
        IF partial_text:
            stability = stability_tracker.update(
                partial_text, confidence, current_time
            )
            
            // Buffer ASR output
            asr_buffer.append({
                text: partial_text,
                confidence: confidence,
                stability: stability,
                timestamp: current_time
            })
        
        SLEEP(100ms)

// Handle turn transitions
FUNCTION handle_turn_transition(new_state, old_state):
    IF new_state == SPEAK AND old_state == HOLD:
        // User finished speaking → System should respond
        initiate_system_response()
    
    ELSE IF new_state == HOLD AND old_state == SPEAK:
        // User interrupted → Cancel system response
        handle_interruption()

// Initiate system response
ASYNC FUNCTION initiate_system_response():
    // Get final text from buffer
    final_text = asr_buffer.last().text
    
    // Clear buffer
    asr_buffer.clear()
    
    // Change system state
    system_state = PROCESSING
    
    // Generate and speak response
    ASYNC stream_llm_to_tts(final_text)

// Stream LLM to TTS
ASYNC FUNCTION stream_llm_to_tts(user_text):
    system_state = PROCESSING
    
    ASYNC FOR token IN LLM.generate_streaming(user_text):
        // Check for interruption
        IF turn_state == HOLD:
            BREAK  // User interrupted
        
        // Start TTS on first token
        IF system_state == PROCESSING:
            system_state = SPEAKING
        
        // Stream to TTS
        TTS.add_token(token)
    
    // Finish TTS
    TTS.finish()
    AWAIT TTS.completion()
    
    // Return to listening
    system_state = LISTENING

// Handle interruption (<50ms requirement)
ASYNC FUNCTION handle_interruption():
    start_time = NOW()
    
    // 1. Stop TTS immediately
    TTS.stop()  // Must complete in <50ms
    
    // 2. Cancel LLM generation
    LLM.cancel_generation()
    
    // 3. Update state
    system_state = INTERRUPTED
    SLEEP(50ms)
    system_state = LISTENING
    
    // 4. Clear buffers
    asr_buffer.clear()
    
    elapsed = NOW() - start_time
    LOG("Interruption handled in " + elapsed + "ms")
    
    IF elapsed > 50ms:
        LOG_WARNING("Interruption latency exceeded 50ms!")
```

## Critical Timing Requirements

```
┌─────────────────────────────────────────────────────────────────┐
│                    TIMING BUDGET BREAKDOWN                      │
└─────────────────────────────────────────────────────────────────┘

End-to-end latency target: <300ms
────────────────────────────────────────────────────────────────

From: User stops speaking
To: System starts speaking (first TTS audio)

Breakdown:
  • User silence detection     : 400ms  (MIN_SILENCE_TO_SPEAK_MS)
  • Hysteresis stabilization   : 200ms  (HYSTERESIS_WINDOW_MS)
  • ASR token stabilization    : ~100ms (overlaps with silence)
  • EasyTurn decision          : <5ms   (runs every 30ms)
  • LLM first token            : <100ms (streaming LLM critical!)
  • TTS synthesis start        : <50ms  (must be streaming)
  ─────────────────────────────────────
  TOTAL (perceived):           ~600-700ms

To achieve <300ms perceived latency:
  1. Reduce MIN_SILENCE_TO_SPEAK_MS to 300ms (aggressive config)
  2. Reduce HYSTERESIS_WINDOW_MS to 150ms
  3. Optimize LLM first-token latency (<50ms with caching)
  4. Use streaming TTS with <20ms startup

Interruption latency target: <50ms
────────────────────────────────────

From: User starts speaking (while system speaks)
To: TTS audio stops

Breakdown:
  • Audio frame arrival        : 30ms   (one frame period)
  • EasyTurn detection         : <5ms   (single cycle)
  • TTS stop command           : <1ms   (async call)
  • TTS audio buffer flush     : <15ms  (hardware dependent)
  ─────────────────────────────────────
  TOTAL:                       <50ms

Critical optimizations:
  1. Use INTERRUPT thresholds (lower than normal)
  2. TTS must support immediate stop (no buffer draining)
  3. Audio output with minimal buffering (<20ms)
```

## Decision Tree Visualization

```
Every 30ms cycle:
├─ Extract acoustic features (5-10ms)
│  ├─ Frame energy: RMS of audio samples
│  ├─ VAD probability: Sigmoid(energy - threshold)
│  └─ Silence duration: Time since last speech
│
├─ Get linguistic features (cached, <1ms)
│  ├─ Partial ASR text
│  ├─ ASR confidence
│  └─ Token stability score
│
├─ Is user speaking? (1ms)
│  ├─ YES: energy > 0.02 OR vad_prob > 0.7
│  └─ NO: energy < 0.005 AND vad_prob < 0.3
│
├─ Current state?
│  │
│  ├─ HOLD state:
│  │  ├─ User speaking? → Stay HOLD
│  │  ├─ Silence < 400ms? → Stay HOLD
│  │  ├─ Text < 5 chars? → Stay HOLD
│  │  ├─ Confidence < 0.6? → Stay HOLD
│  │  ├─ Stability < 0.8? → Stay HOLD
│  │  ├─ Text still changing? → Stay HOLD
│  │  └─ All checks pass? → Suggest SPEAK
│  │
│  └─ SPEAK state:
│     ├─ Energy > 0.015? → Suggest HOLD (interrupt)
│     ├─ VAD > 0.6? → Suggest HOLD (interrupt)
│     └─ No interruption? → Stay SPEAK
│
└─ Apply hysteresis (2ms)
   ├─ Same as current? → Confirm current state
   ├─ Different from current?
   │  ├─ Stable for 200ms? → Transition
   │  └─ Not stable yet? → Stay in current state
   └─ Return decision
```

This completes the state machine and decision logic documentation.
