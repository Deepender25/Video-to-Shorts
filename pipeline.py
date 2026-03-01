import os
import uuid
import traceback
from config import DOWNLOADS_DIR, OUTPUTS_DIR
from downloader import download_video
from transcript import parse_srt, clean_transcript, merge_segments, format_for_llm
from segmenter import segment_transcript
from validator import validate_clips
from cutter import cut_clips

import re as _re


# In-memory job store
_jobs: dict[str, dict] = {}


def _dedup_srt_for_subtitles(segments: list[dict]) -> list[dict]:
    """
    Remove YouTube rolling-window duplicate SRT entries using TIME, not text.

    YouTube auto-captions produce overlapping entries like:
      [0:01-0:03] "aapko aligarh ke bare"
      [0:02-0:04] "aapko aligarh ke bare"   ← same text, overlapping time → DROP
      [0:05-0:07] "theek hai"
      ...
      [1:15-1:17] "theek hai"               ← same text, DIFFERENT time → KEEP

    Rule: drop an entry only if it overlaps in time with the previous KEPT entry
    AND their texts are identical (case-insensitive). Same text at non-overlapping
    times is always preserved — this is crucial for sync correctness.
    """
    if not segments:
        return []

    noise = _re.compile(r"\[.*?\]", _re.IGNORECASE)
    result = []

    for seg in segments:
        text = noise.sub("", seg["text"]).strip()
        text = _re.sub(r"\s+", " ", text).strip()
        if not text or len(text) < 2:
            continue

        if result:
            prev = result[-1]
            # Only drop if this entry overlaps the previous one in time
            overlaps_prev = seg["start"] < prev["end"]
            same_text     = text.lower() == prev["text"].lower()
            if overlaps_prev and same_text:
                # Extend previous entry's end time to cover both
                result[-1] = dict(prev)
                result[-1]["end"] = max(prev["end"], seg["end"])
                continue

        result.append({"start": seg["start"], "end": seg["end"], "text": text})

    return result


def get_job(job_id: str) -> dict | None:
    return _jobs.get(job_id)


def create_job(youtube_url: str) -> str:
    """Create a job entry and return its ID (does not start processing)."""
    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {
        "id": job_id,
        "url": youtube_url,
        "status": "starting",
        "progress": 0,
        "message": "Initializing...",
        "clips": [],
        "error": None,
        "video_title": "",
        # Phase 1 results (populated after download+transcript)
        "video_path": None,
        "video_filename": None,
        "srt_path": None,
        "duration": 0,
        "transcript_segments": [],
        "transcript_formatted": "",
    }
    return job_id


def run_download_phase(job_id: str) -> None:
    """
    Phase 1: Download video + captions, parse transcript.
    On success, status becomes 'review' — the UI shows a preview and waits
    for the user to click 'Continue'.
    """
    job = _jobs[job_id]
    youtube_url = job["url"]

    try:
        # ── Step 1: Download ─────────────────────────────
        job["status"] = "downloading"
        job["progress"] = 10
        job["message"] = "Downloading video and captions..."

        download_dir = os.path.join(DOWNLOADS_DIR, job_id)
        result = download_video(youtube_url, download_dir)
        job["video_title"] = result["title"]
        job["video_path"] = result["video_path"]
        job["video_filename"] = os.path.basename(result["video_path"])
        job["srt_path"] = result["srt_path"]
        job["duration"] = result["duration"]
        job["subtitle_lang"] = result.get("subtitle_lang", "en")

        # Store subtitle-safe SRT segments for subtitle burning.
        # Important: we do NOT use clean_transcript here because it removes ALL
        # entries with the same text globally (across the whole video). If the
        # speaker says the same phrase at t=10s and again at t=60s, the t=60s
        # entry would be silently dropped, causing subtitle→audio sync failures
        # for clips from later in the video.
        # Instead we use _dedup_srt_for_subtitles which only de-duplicates
        # entries that OVERLAP IN TIME (the YouTube rolling-window auto-caption
        # pattern) without ever dropping same-text entries at different times.
        raw_srt = parse_srt(result["srt_path"]) if result.get("srt_path") else []
        job["raw_srt_segments"] = _dedup_srt_for_subtitles(raw_srt)

        # ── Step 2: Parse transcript ─────────────────────
        job["status"] = "parsing"
        job["progress"] = 30
        job["message"] = "Parsing captions..."

        segments = parse_srt(result["srt_path"])
        if not segments:
            raise ValueError(
                "No valid caption segments found in the subtitle file. "
                "The captions may be empty or corrupted."
            )

        cleaned = clean_transcript(segments)
        if not cleaned:
            raise ValueError(
                "All caption segments were noise (e.g. [Music]). "
                "No usable text found."
            )

        merged = merge_segments(cleaned)
        formatted = format_for_llm(merged)

        # Store for later use in phase 2 and for the UI
        job["transcript_segments"] = merged
        job["transcript_formatted"] = formatted

        # ── Ready for review ─────────────────────────────
        job["status"] = "review"
        job["progress"] = 40
        job["message"] = "Download complete! Review the video and transcript."

    except Exception as e:
        job["status"] = "error"
        job["progress"] = 0
        job["message"] = str(e)
        job["error"] = traceback.format_exc()


def run_analysis_phase(job_id: str) -> None:
    """
    Phase 2: LLM segmentation, validation, and cutting.
    Called only after user approves via the 'Continue' button.
    """
    job = _jobs[job_id]

    try:
        formatted = job.get("transcript_formatted", "")
        if not formatted:
            raise ValueError("No transcript data available. Please restart.")

        merged = job.get("transcript_segments", [])
        transcript_start = merged[0]["start"] if merged else 0
        transcript_end = merged[-1]["end"] if merged else 0

        # ── Step 3: LLM segmentation ────────────────────
        job["status"] = "analyzing"
        job["progress"] = 55
        job["message"] = "AI is analyzing the content..."

        subtitle_lang = job.get("subtitle_lang", "en")
        raw_clips = segment_transcript(formatted, subtitle_lang=subtitle_lang)
        if not raw_clips:
            raise ValueError("AI could not identify any suitable clips.")

        # ── Step 4: Validate ─────────────────────────────
        job["status"] = "validating"
        job["progress"] = 75
        job["message"] = "Validating timestamps..."

        video_duration = job.get("duration", 0)
        valid_clips = validate_clips(
            raw_clips, video_duration, transcript_start, transcript_end
        )
        if not valid_clips:
            raise ValueError("No clips passed validation. Try a different video.")

        # ── Step 5: Cut video ────────────────────────────
        job["status"] = "cutting"
        job["progress"] = 88
        job["message"] = f"Cutting {len(valid_clips)} clips..."

        video_path = job.get("video_path", "")
        output_dir = os.path.join(OUTPUTS_DIR, job_id)
        srt_segments = job.get("raw_srt_segments") or None
        final_clips = cut_clips(video_path, valid_clips, output_dir, srt_segments=srt_segments)

        if not final_clips:
            raise ValueError("FFmpeg could not produce any clips.")

        # ── Done ─────────────────────────────────────────
        job["status"] = "done"
        job["progress"] = 100
        job["message"] = f"Successfully created {len(final_clips)} shorts!"
        job["clips"] = final_clips

    except Exception as e:
        job["status"] = "error"
        job["progress"] = 0
        job["message"] = str(e)
        job["error"] = traceback.format_exc()
