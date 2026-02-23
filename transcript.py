import re


def _ts_to_seconds(ts: str) -> float:
    """Convert SRT timestamp (HH:MM:SS,mmm) to float seconds."""
    ts = ts.strip().replace(",", ".")
    parts = ts.split(":")
    h, m, s = float(parts[0]), float(parts[1]), float(parts[2])
    return h * 3600 + m * 60 + s


def parse_srt(srt_path: str) -> list[dict]:
    """
    Parse an SRT file into a list of segments.

    Returns:
        list of {start: float, end: float, text: str}
    """
    with open(srt_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Split into blocks by double newline
    blocks = re.split(r"\n\s*\n", content.strip())
    segments = []

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue

        # Line 1: index (skip)
        # Line 2: timestamps
        ts_match = re.match(
            r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})",
            lines[1],
        )
        if not ts_match:
            continue

        start = _ts_to_seconds(ts_match.group(1))
        end = _ts_to_seconds(ts_match.group(2))
        text = " ".join(lines[2:]).strip()

        if text:
            segments.append({"start": start, "end": end, "text": text})

    return segments


def clean_transcript(segments: list[dict]) -> list[dict]:
    """
    Remove noise from transcript segments.

    - Strips [Music], [Applause], etc.
    - Removes duplicates
    - Normalizes whitespace
    """
    noise_patterns = re.compile(
        r"\[.*?\]", re.IGNORECASE
    )

    cleaned = []
    seen_texts = set()

    for seg in segments:
        text = noise_patterns.sub("", seg["text"]).strip()
        text = re.sub(r"\s+", " ", text)  # normalize whitespace

        if not text or len(text) < 3:
            continue

        # Skip exact duplicates
        if text.lower() in seen_texts:
            continue
        seen_texts.add(text.lower())

        cleaned.append({"start": seg["start"], "end": seg["end"], "text": text})

    return cleaned


def merge_segments(segments: list[dict], min_length: int = 40) -> list[dict]:
    """
    Merge short adjacent segments into longer, more complete sentences.

    Args:
        segments: cleaned transcript segments
        min_length: minimum character length before a segment stands alone
    """
    if not segments:
        return []

    merged = [segments[0].copy()]

    for seg in segments[1:]:
        prev = merged[-1]

        # Merge if previous segment is short
        if len(prev["text"]) < min_length:
            prev["text"] = prev["text"] + " " + seg["text"]
            prev["end"] = seg["end"]
        else:
            merged.append(seg.copy())

    return merged


def format_for_llm(segments: list[dict]) -> str:
    """
    Convert segments into the LLM-ready timeline format.

    Output:
        [start - end] sentence text
    """
    lines = []
    for seg in segments:
        start = round(seg["start"], 1)
        end = round(seg["end"], 1)
        lines.append(f"[{start} - {end}] {seg['text']}")
    return "\n".join(lines)
