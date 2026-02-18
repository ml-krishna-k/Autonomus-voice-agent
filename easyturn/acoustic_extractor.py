"""
Acoustic Feature Extractor

Extracts acoustic features from audio frames for turn-taking decisions.
Runs in real-time with minimal latency.
"""

import numpy as np
from typing import Optional
from collections import deque



class AcousticExtractor:
    """
    Real-time acoustic feature extraction.
    
    Features extracted:
    - Frame energy (RMS)
    - Soft VAD probability
    - Silence duration tracking
    - Pitch (optional, adds latency)
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        energy_noise_floor: float = 0.001,
        extract_pitch: bool = False
    ):
        """
        Initialize acoustic extractor.
        
        Args:
            sample_rate: Audio sample rate (Hz)
            frame_duration_ms: Frame duration in milliseconds
            energy_noise_floor: Noise floor for energy calculation
            extract_pitch: Whether to extract pitch (adds ~5-10ms latency)
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.energy_noise_floor = energy_noise_floor
        self.extract_pitch = extract_pitch
        
        # Silence tracking
        self.silence_start_ms: Optional[float] = None
        self.current_silence_duration_ms = 0
        
        # Energy smoothing
        self.energy_history = deque(maxlen=5)
        
        # VAD state
        self.vad_threshold = 0.02  # Energy-based VAD threshold
        
    def extract(self, audio_frame: np.ndarray, timestamp_ms: float) -> dict:
        """
        Extract acoustic features from a single audio frame.
        
        Args:
            audio_frame: Audio samples (numpy array, float32, [-1, 1])
            timestamp_ms: Timestamp of this frame
            
        Returns:
            Dictionary with acoustic features
        """
        # Ensure correct shape
        if len(audio_frame.shape) > 1:
            audio_frame = audio_frame.flatten()
        
        # Extract features
        energy = self._calculate_energy(audio_frame)
        vad_prob = self._calculate_vad_probability(energy)
        silence_duration_ms = self._update_silence_duration(vad_prob, timestamp_ms)
        
        features = {
            'timestamp_ms': timestamp_ms,
            'frame_energy': float(energy),
            'vad_prob': float(vad_prob),
            'silence_duration_ms': int(silence_duration_ms)
        }
        
        # Optional pitch extraction
        if self.extract_pitch:
            pitch = self._extract_pitch(audio_frame)
            features['pitch'] = pitch
        
        return features
    
    def _calculate_energy(self, audio_frame: np.ndarray) -> float:
        """
        Calculate RMS energy of audio frame.
        
        Uses smoothing to reduce noise.
        """
        # RMS energy
        rms = np.sqrt(np.mean(audio_frame ** 2))
        
        # Add to history for smoothing
        self.energy_history.append(rms)
        
        # Smooth over recent history
        smoothed_energy = np.mean(self.energy_history)
        
        # Subtract noise floor
        energy = max(0.0, smoothed_energy - self.energy_noise_floor)
        
        return energy
    
    def _calculate_vad_probability(self, energy: float) -> float:
        """
        Calculate soft VAD probability from energy.
        
        Uses sigmoid function for smooth probability.
        """
        # Sigmoid centered at VAD threshold
        # Output range: [0, 1]
        # Steepness controls sensitivity
        steepness = 100.0
        vad_prob = 1.0 / (1.0 + np.exp(-steepness * (energy - self.vad_threshold)))
        
        return vad_prob
    
    def _update_silence_duration(
        self,
        vad_prob: float,
        timestamp_ms: float
    ) -> float:
        """
        Track consecutive silence duration.
        
        Args:
            vad_prob: Current VAD probability
            timestamp_ms: Current timestamp
            
        Returns:
            Silence duration in milliseconds
        """
        # Threshold for considering speech vs silence
        SPEECH_THRESHOLD = 0.3
        
        if vad_prob > SPEECH_THRESHOLD:
            # Speech detected - reset silence
            self.silence_start_ms = None
            self.current_silence_duration_ms = 0
        else:
            # Silence
            if self.silence_start_ms is None:
                # Start of silence
                self.silence_start_ms = timestamp_ms
                self.current_silence_duration_ms = 0
            else:
                # Continue silence
                self.current_silence_duration_ms = timestamp_ms - self.silence_start_ms
        
        return self.current_silence_duration_ms
    
    def _extract_pitch(self, audio_frame: np.ndarray) -> Optional[float]:
        """
        Extract pitch using autocorrelation.
        
        WARNING: This adds latency (~5-10ms). Only use if needed.
        """
        if not self.extract_pitch:
            return None
            
        try:
            import librosa
            # Use librosa for pitch detection
            pitches, magnitudes = librosa.piptrack(
                y=audio_frame,
                sr=self.sample_rate,
                hop_length=self.frame_size
            )
            
            # Get the pitch with highest magnitude
            if pitches.size > 0:
                index = magnitudes.argmax()
                pitch = pitches[index]
                if pitch > 0:
                    return float(pitch)
        except ImportError:
            # Librosa not installed - silently fail or log warning
            pass
        except Exception:
            pass
        
        return None
    
    def reset(self):
        """Reset extractor state"""
        self.silence_start_ms = None
        self.current_silence_duration_ms = 0
        self.energy_history.clear()


class BatchAcousticExtractor(AcousticExtractor):
    """
    Optimized batch extractor for processing multiple frames at once.
    Useful for offline testing or when audio arrives in batches.
    """
    
    def extract_batch(
        self,
        audio_frames: np.ndarray,
        start_timestamp_ms: float
    ) -> list[dict]:
        """
        Extract features from multiple frames.
        
        Args:
            audio_frames: Audio array (samples, channels) or (samples,)
            start_timestamp_ms: Timestamp of first frame
            
        Returns:
            List of feature dictionaries
        """
        # Ensure 1D
        if len(audio_frames.shape) > 1:
            audio_frames = audio_frames.flatten()
        
        # Split into frames
        num_frames = len(audio_frames) // self.frame_size
        features_list = []
        
        for i in range(num_frames):
            start_idx = i * self.frame_size
            end_idx = start_idx + self.frame_size
            frame = audio_frames[start_idx:end_idx]
            
            timestamp_ms = start_timestamp_ms + (i * self.frame_duration_ms)
            
            features = self.extract(frame, timestamp_ms)
            features_list.append(features)
        
        return features_list
