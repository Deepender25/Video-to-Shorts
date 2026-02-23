from config import MIN_CLIP_DURATION, MAX_CLIP_DURATION


def validate_clips(
    clips: list[dict],
    video_duration: float,
    transcript_start: float,
    transcript_end: float,
) -> list[dict]:
    """
    Validate and filter clips (supports multi-segment compiled shorts).

    Each clip has:
        - title: str
        - hook: str
        - segments: list of {start: float, end: float}

    Checks per segment:
        - Timestamps within video and transcript bounds
        - No negative or reversed timestamps
        - Each segment at least 5 seconds

    Checks per clip:
        - Total duration between MIN_CLIP_DURATION and MAX_CLIP_DURATION
        - Segments sorted by start time
        - Required fields present

    Returns:
        list of valid clips, sorted by first segment's start time
    """
    valid = []

    for clip in clips:
        # Check required fields
        if "segments" not in clip or not isinstance(clip["segments"], list):
            continue
        if not clip["segments"]:
            continue

        # Validate each segment
        valid_segments = []
        skip_clip = False

        for seg in clip["segments"]:
            try:
                start = float(seg["start"])
                end = float(seg["end"])
            except (ValueError, TypeError, KeyError):
                skip_clip = True
                break

            # Basic sanity
            if start < 0 or end < 0 or start >= end:
                skip_clip = True
                break

            # Each segment should be at least 5 seconds
            if end - start < 5:
                skip_clip = True
                break

            # Within video duration (1s tolerance)
            if end > video_duration + 1:
                skip_clip = True
                break

            # Within transcript range (2s tolerance)
            if start < transcript_start - 2 or end > transcript_end + 2:
                skip_clip = True
                break

            valid_segments.append({
                "start": round(start, 2),
                "end": round(end, 2),
            })

        if skip_clip or not valid_segments:
            continue

        # Sort segments by start time within the clip
        valid_segments.sort(key=lambda s: s["start"])

        # Total duration check
        total_duration = sum(s["end"] - s["start"] for s in valid_segments)
        if total_duration < MIN_CLIP_DURATION or total_duration > MAX_CLIP_DURATION:
            continue

        valid.append({
            "title": str(clip.get("title", "Untitled")),
            "hook": str(clip.get("hook", "")),
            "segments": valid_segments,
            "duration": round(total_duration, 2),
            # Keep start/end of the first/last segment for backward compat
            "start": valid_segments[0]["start"],
            "end": valid_segments[-1]["end"],
        })

    # Sort clips by first segment's start time
    valid.sort(key=lambda c: c["start"])

    # Remove clips that have overlapping segments with previous clips
    non_overlapping = []
    used_ranges = []  # list of (start, end) tuples from all accepted clips

    for clip in valid:
        overlaps = False
        for seg in clip["segments"]:
            for used_start, used_end in used_ranges:
                if seg["start"] < used_end and seg["end"] > used_start:
                    overlaps = True
                    break
            if overlaps:
                break

        if not overlaps:
            non_overlapping.append(clip)
            for seg in clip["segments"]:
                used_ranges.append((seg["start"], seg["end"]))

    return non_overlapping
