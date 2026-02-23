from config import MIN_CLIP_DURATION, MAX_CLIP_DURATION


def validate_clips(
    clips: list[dict],
    video_duration: float,
    transcript_start: float,
    transcript_end: float,
) -> list[dict]:
    """
    Validate and filter clips.

    Checks:
    - Duration between MIN_CLIP_DURATION and MAX_CLIP_DURATION
    - No overlapping clips
    - Timestamps within video duration
    - Timestamps within transcript range
    - Required fields present

    Returns:
        list of valid clips, sorted by start time
    """
    valid = []

    for clip in clips:
        # Check required fields
        if not all(k in clip for k in ("start", "end", "title")):
            continue

        try:
            start = float(clip["start"])
            end = float(clip["end"])
        except (ValueError, TypeError):
            continue

        # Duration check
        duration = end - start
        if duration < MIN_CLIP_DURATION or duration > MAX_CLIP_DURATION:
            continue

        # Timestamp sanity
        if start < 0 or end < 0:
            continue
        if start >= end:
            continue

        # Within video duration
        if end > video_duration + 1:  # 1s tolerance
            continue

        # Within transcript range (with tolerance)
        if start < transcript_start - 2 or end > transcript_end + 2:
            continue

        valid.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "title": str(clip.get("title", "Untitled")),
            "hook": str(clip.get("hook", "")),
            "duration": round(duration, 2),
        })

    # Sort by start time
    valid.sort(key=lambda c: c["start"])

    # Remove overlapping clips (keep earlier one)
    non_overlapping = []
    for clip in valid:
        if non_overlapping:
            prev = non_overlapping[-1]
            if clip["start"] < prev["end"]:
                continue  # skip overlapping
        non_overlapping.append(clip)

    return non_overlapping
