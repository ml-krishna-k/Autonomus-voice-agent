def segments_to_text(segments) -> str:
    return " ".join(s.text.strip() for s in segments if s.text)