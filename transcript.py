import re


def _ts_to_seconds(ts: str) -> float:
    """Convert SRT timestamp (HH:MM:SS,mmm) to float seconds."""
    ts = ts.strip().replace(",", ".")
    parts = ts.split(":")
    h, m, s = float(parts[0]), float(parts[1]), float(parts[2])
    return h * 3600 + m * 60 + s


def _seconds_to_mmss(seconds: float) -> str:
    """Convert seconds to compact MM:SS format."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


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
        if len(lines) < 2:
            continue

        # Find the timestamp line (could be line 1 or line 0)
        ts_match = None
        text_start_idx = 0
        for i, line in enumerate(lines):
            ts_match = re.match(
                r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})",
                line.strip(),
            )
            if ts_match:
                text_start_idx = i + 1
                break

        if not ts_match or text_start_idx >= len(lines):
            continue

        start = _ts_to_seconds(ts_match.group(1))
        end = _ts_to_seconds(ts_match.group(2))
        text = " ".join(lines[text_start_idx:]).strip()

        if text:
            segments.append({"start": start, "end": end, "text": text})

    return segments


def clean_transcript(segments: list[dict]) -> list[dict]:
    """
    Aggressively clean transcript segments for minimal token usage.

    - Strips [Music], [Applause], etc.
    - Removes exact and near-duplicate text (auto-subs repeat heavily)
    - Normalizes whitespace
    - Skips very short or empty segments
    """
    noise_patterns = re.compile(r"\[.*?\]", re.IGNORECASE)

    cleaned = []
    seen_texts = set()

    for seg in segments:
        text = noise_patterns.sub("", seg["text"]).strip()
        text = re.sub(r"\s+", " ", text)  # normalize whitespace
        text = text.strip()

        if not text or len(text) < 3:
            continue

        # Auto-generated subs heavily duplicate: skip exact duplicates
        text_key = text.lower().strip()
        if text_key in seen_texts:
            continue
        seen_texts.add(text_key)

        cleaned.append({"start": seg["start"], "end": seg["end"], "text": text})

    return cleaned


def merge_segments(segments: list[dict], min_length: int = 50) -> list[dict]:
    """
    Merge short adjacent segments into longer, more complete sentences.
    Also deduplicates overlapping text from auto-generated subtitles where
    each new segment includes the previous segment's text plus a few new words.

    Args:
        segments: cleaned transcript segments
        min_length: minimum character length before a segment stands alone
    """
    if not segments:
        return []

    # Step 1: Deduplicate overlapping text from auto-subs
    # Auto-subs often produce: seg1="A B C", seg2="A B C D E", seg3="D E F G"
    # We want to keep only the new text from each segment
    deduped = [segments[0].copy()]

    for seg in segments[1:]:
        prev_text = deduped[-1]["text"]
        curr_text = seg["text"]

        # Check if current text starts with most of the previous text
        # (auto-sub overlap pattern)
        if len(prev_text) > 10 and curr_text.startswith(prev_text[:len(prev_text)//2]):
            # This is an extension of the previous segment — update end time and text
            deduped[-1]["end"] = seg["end"]
            deduped[-1]["text"] = curr_text
        else:
            deduped.append(seg.copy())

    # Step 2: Merge short segments for complete thoughts
    merged = [deduped[0].copy()]

    for seg in deduped[1:]:
        prev = merged[-1]

        # Merge if previous segment is short
        if len(prev["text"]) < min_length:
            prev["text"] = prev["text"] + " " + seg["text"]
            prev["end"] = seg["end"]
        else:
            merged.append(seg.copy())

    # Handle last segment if it's too short
    if len(merged) > 1 and len(merged[-1]["text"]) < 20:
        merged[-2]["text"] += " " + merged[-1]["text"]
        merged[-2]["end"] = merged[-1]["end"]
        merged.pop()

    return merged


def format_for_llm(segments: list[dict]) -> str:
    """
    Convert segments into a compact dialogue format for the LLM.
    Uses MM:SS timestamps instead of floats to save tokens.

    Output format:
        [0:00–0:35] Dialogue text here
    """
    lines = []
    for seg in segments:
        start = _seconds_to_mmss(seg["start"])
        end = _seconds_to_mmss(seg["end"])
        lines.append(f"[{start}–{end}] {seg['text']}")
    return "\n".join(lines)
