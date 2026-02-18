"""
Token Stability Tracker for ASR output

Tracks how stable the ASR output is over time.
Useful for detecting semantic completion.
"""

from collections import deque
from typing import Optional
import difflib


class TokenStabilityTracker:
    """
    Tracks stability of ASR tokens over time.
    
    Stability indicates that the ASR has "settled" and is unlikely to change,
    suggesting semantic completion.
    """
    
    def __init__(self, window_size_ms: int = 500, min_samples: int = 5):
        """
        Initialize stability tracker.
        
        Args:
            window_size_ms: Time window for stability calculation
            min_samples: Minimum samples needed for stability score
        """
        self.window_size_ms = window_size_ms
        self.min_samples = min_samples
        
        # History of (timestamp_ms, text, confidence) tuples
        self.history = deque(maxlen=50)
        
        self.last_text = ""
        self.last_timestamp_ms = 0
        
    def update(
        self,
        partial_text: str,
        confidence: float,
        timestamp_ms: float
    ) -> float:
        """
        Update with new ASR output and calculate stability.
        
        Args:
            partial_text: Current partial ASR text
            confidence: ASR confidence score
            timestamp_ms: Current timestamp
            
        Returns:
            Stability score [0, 1], where 1 = very stable
        """
        # Add to history
        self.history.append((timestamp_ms, partial_text, confidence))
        
        # Calculate stability score
        stability = self._calculate_stability(timestamp_ms)
        
        self.last_text = partial_text
        self.last_timestamp_ms = timestamp_ms
        
        return stability
    
    def _calculate_stability(self, current_time_ms: float) -> float:
        """
        Calculate stability score based on recent history.
        
        Stability is high when:
        1. Text hasn't changed recently
        2. Confidence is consistently high
        3. Text length is stable
        """
        if len(self.history) < self.min_samples:
            return 0.0
        
        # Filter to recent window
        recent = [
            (ts, text, conf)
            for ts, text, conf in self.history
            if current_time_ms - ts <= self.window_size_ms
        ]
        
        if len(recent) < 2:
            return 0.0
        
        # Calculate text stability (similarity between consecutive texts)
        text_stability = self._calculate_text_stability(recent)
        
        # Calculate confidence stability (variance in confidence)
        confidence_stability = self._calculate_confidence_stability(recent)
        
        # Calculate length stability
        length_stability = self._calculate_length_stability(recent)
        
        # Combine scores
        # Weight text stability most heavily
        stability = (
            0.6 * text_stability +
            0.25 * confidence_stability +
            0.15 * length_stability
        )
        
        return stability
    
    def _calculate_text_stability(self, recent: list) -> float:
        """
        Calculate how stable the text is (minimal changes).
        
        Uses edit distance between consecutive texts.
        """
        if len(recent) < 2:
            return 0.0
        
        # Compare recent texts
        similarities = []
        for i in range(len(recent) - 1):
            text1 = recent[i][1]
            text2 = recent[i + 1][1]
            
            # Calculate similarity ratio
            similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
            similarities.append(similarity)
        
        # Average similarity
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        
        return avg_similarity
    
    def _calculate_confidence_stability(self, recent: list) -> float:
        """
        Calculate stability of confidence scores.
        
        Low variance = stable, high variance = unstable.
        """
        confidences = [conf for _, _, conf in recent]
        
        if len(confidences) < 2:
            return 0.0
        
        # Calculate variance
        mean_conf = sum(confidences) / len(confidences)
        variance = sum((c - mean_conf) ** 2 for c in confidences) / len(confidences)
        
        # Convert to stability (low variance = high stability)
        # Normalize: variance of 0.01 → stability ~0.9, variance of 0.25 → stability ~0.0
        stability = max(0.0, 1.0 - (variance * 4.0))
        
        return stability
    
    def _calculate_length_stability(self, recent: list) -> float:
        """
        Calculate stability of text length.
        
        Stable length suggests no major rewrites by ASR.
        """
        lengths = [len(text) for _, text, _ in recent]
        
        if len(lengths) < 2:
            return 0.0
        
        # Calculate coefficient of variation
        mean_length = sum(lengths) / len(lengths)
        if mean_length == 0:
            return 0.0
        
        variance = sum((l - mean_length) ** 2 for l in lengths) / len(lengths)
        std_dev = variance ** 0.5
        cv = std_dev / mean_length  # Coefficient of variation
        
        # Convert to stability (low CV = high stability)
        # CV of 0.1 → stability 0.9, CV of 0.5 → stability 0.5
        stability = max(0.0, 1.0 - cv)
        
        return stability
    
    def has_text_changed_recently(self, threshold_ms: int = 200) -> bool:
        """
        Check if text has changed in the last threshold_ms.
        
        Args:
            threshold_ms: Time threshold in milliseconds
            
        Returns:
            True if text changed recently
        """
        if len(self.history) < 2:
            return False
        
        current_time = self.last_timestamp_ms
        recent_texts = [
            text for ts, text, _ in self.history
            if current_time - ts <= threshold_ms
        ]
        
        if len(recent_texts) < 2:
            return False
        
        # Check if any texts differ
        first_text = recent_texts[0]
        return any(text != first_text for text in recent_texts[1:])
    
    def reset(self):
        """Reset tracker state"""
        self.history.clear()
        self.last_text = ""
        self.last_timestamp_ms = 0
