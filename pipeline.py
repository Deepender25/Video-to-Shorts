import os
import uuid
import traceback
from config import DOWNLOADS_DIR, OUTPUTS_DIR
from downloader import download_video
from transcript import parse_srt, clean_transcript, merge_segments, format_for_llm
from segmenter import segment_transcript
from validator import validate_clips
from cutter import cut_clips

# In-memory job store
_jobs: dict[str, dict] = {}


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
    Phase 2: LLM segmentation, validation, and cutting concurrently.
    Called only after user approves via the 'Continue' button.
    """
    import concurrent.futures
    import threading

    job = _jobs[job_id]

    try:
        formatted = job.get("transcript_formatted", "")
        if not formatted:
            raise ValueError("No transcript data available. Please restart.")

        merged = job.get("transcript_segments", [])
        transcript_start = merged[0]["start"] if merged else 0
        transcript_end = merged[-1]["end"] if merged else 0

        # ── Step 3: LLM segmentation (Streaming) ────────
        job["status"] = "analyzing"
        job["progress"] = 55
        job["message"] = "AI is streaming analysis and processing clips in parallel..."
        
        subtitle_lang = job.get("subtitle_lang", "en")
        video_duration = job.get("duration", 0)
        video_path = job.get("video_path", "")
        output_dir = os.path.join(OUTPUTS_DIR, job_id)
        os.makedirs(output_dir, exist_ok=True)
        
        used_ranges = []
        state_lock = threading.Lock()
        
        def process_single_clip(valid_clip):
            try:
                # ── Step 5: Cut video ────────────────────────────
                # cut_clips takes a list, so we pass one
                final_clips = cut_clips(video_path, [valid_clip], output_dir)
                if not final_clips:
                    return
                
                final_clip = final_clips[0]
                clip_video_path = os.path.join(output_dir, final_clip["filename"])
                
                # We skip background word-level transcription and burning to keep the pipeline fast.
                # Word-level generation happens ON DEMAND when user opens the React Editor.
                final_clip["word_segments"] = []
                final_clip["burned_filename"] = "" 
                
                with state_lock:
                    job["clips"].append(final_clip)
                    job["message"] = f"Processed {len(job['clips'])} clips so far..."
            except Exception as e:
                print(f"Error processing clip '{valid_clip.get('title')}': {e}")
                traceback.print_exc()

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            
            # segment_transcript yields single clips
            clip_generator = segment_transcript(formatted, subtitle_lang=subtitle_lang)
            
            for raw_clip in clip_generator:
                if job.get("status") == "error":
                    break
                    
                # Evaluate against used ranges to prevent duplicate coverage
                with state_lock:
                    valid = validate_clips([raw_clip], video_duration, transcript_start, transcript_end)
                    if not valid:
                        continue
                    
                    valid_clip = valid[0]
                    
                    # Check overlap manually
                    overlaps = False
                    for seg in valid_clip["segments"]:
                        for used_start, used_end in used_ranges:
                            if seg["start"] < used_end and seg["end"] > used_start:
                                overlaps = True
                                break
                        if overlaps:
                            break
                    
                    if overlaps:
                        continue
                        
                    # Commit ranges
                    for seg in valid_clip["segments"]:
                        used_ranges.append((seg["start"], seg["end"]))
                        
                # Submit to worker thread
                futures.append(executor.submit(process_single_clip, valid_clip))
            
            # Wait for all submitted clips to finish processing
            concurrent.futures.wait(futures)

        if not job["clips"]:
            if job["status"] != "error":
                raise ValueError("No clips passed validation or processing failed.")

        # ── Done ─────────────────────────────────────────
        if job["status"] != "error":
            job["status"] = "done"
            job["progress"] = 100
            job["message"] = f"Successfully created {len(job['clips'])} shorts!"

    except Exception as e:
        job["status"] = "error"
        job["progress"] = 0
        job["message"] = str(e)
        job["error"] = traceback.format_exc()
