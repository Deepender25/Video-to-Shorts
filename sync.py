import os
import re
import difflib
import subprocess
from faster_whisper import WhisperModel

# Initialize Whisper model lazily
_model = None

def get_whisper_model():
    global _model
    if _model is None:
        # We use 'tiny' for blazing fast audio-to-text with word-timestamps
        # compute_type="int8" runs optimally on most CPU/GPU combinations
        print("Loading faster-whisper (tiny model) for precise sync...")
        _model = WhisperModel("tiny", device="cpu", compute_type="int8")
    return _model

def extract_audio_for_sync(video_path: str, start: float, duration: float, out_wav: str) -> bool:
    """Extract a small snippet of audio as WAV for whisper synchronization."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", video_path,
        "-t", str(duration),
        "-ac", "1",          # mono
        "-ar", "16000",      # 16kHz for whisper
        "-vn",               # no video
        out_wav
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def sync_subtitles(
    video_path: str,
    clip_start: float,
    clip_end: float,
    original_youtube_text: str,
    temp_dir: str
) -> list[dict]:
    """
    Generate precisely timed karaoke subtitle words by combining Whisper timings
    with the original YouTube transcribed text.

    Args:
        video_path: Path to the full source video.
        clip_start: Start time of the clip within the video (seconds).
        clip_end: End time of the clip within the video (seconds).
        original_youtube_text: The complete text script for this clip (from original SRT).
        temp_dir: Directory to save temporary extraction files.

    Returns:
        List of dicts: {"word": str, "start": float, "end": float} (all times relative to clip_start=0)
    """
    duration = clip_end - clip_start
    audio_wav = os.path.join(temp_dir, "_sync_audio.wav")
    
    # Extract just the audio for this specific clip
    if not extract_audio_for_sync(video_path, clip_start, duration, audio_wav):
        print("    ⚠ Audio extraction failed. Falling back to default spacing.")
        return _fallback_spacing(original_youtube_text, duration)
    
    model = get_whisper_model()
    # Perform transcription with word-level timestamps enabled
    try:
        segments, _ = model.transcribe(audio_wav, word_timestamps=True, vad_filter=True)
        whisper_words = []
        for segment in segments:
            for word in segment.words:
                cleaned = re.sub(r'[^\w\s]', '', word.word).strip().lower()
                if cleaned:
                    whisper_words.append({
                        "text": word.word.strip(),
                        "clean": cleaned,
                        "start": word.start,
                        "end": word.end
                    })
    except Exception as e:
        print(f"    ⚠ Whisper sync failed ({e}). Falling back to default spacing.")
        if os.path.exists(audio_wav):
            os.remove(audio_wav)
        return _fallback_spacing(original_youtube_text, duration)
    
    # Cleanup audio
    if os.path.exists(audio_wav):
        try:
            os.remove(audio_wav)
        except OSError:
            pass
            
    if not whisper_words:
        return _fallback_spacing(original_youtube_text, duration)

    # ── Sequence Matcher ─────────────────────────────────────────────────────────
    # We want to keep the exact punctuation, capitalization, and formatting
    # of the `original_youtube_text`, but snap it to the `whisper_words` timings.
    
    original_words = []
    for w in original_youtube_text.split():
        original_words.append({
            "text": w,
            "clean": re.sub(r'[^\w\s]', '', w).lower()
        })
        
    seq_whisper = [w["clean"] for w in whisper_words]
    seq_original = [w["clean"] for w in original_words]
    
    matcher = difflib.SequenceMatcher(None, seq_whisper, seq_original)
    
    synced_output = []
    
    # Map each original word to the time bounds of the matching whisper word block
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ('equal', 'replace'):
            # Calculate proportion to map N whisper words onto M original words
            w_block = whisper_words[i1:i2]
            o_block = original_words[j1:j2]
            
            if not w_block or not o_block:
                continue
                
            block_start = w_block[0]["start"]
            block_end = w_block[-1]["end"]
            block_dur = block_end - block_start
            time_per_word = block_dur / len(o_block)
            
            t = block_start
            for o_word in o_block:
                synced_output.append({
                    "word": o_word["text"],
                    "start": t,
                    "end": t + time_per_word
                })
                t += time_per_word
        elif tag == 'insert':
            # Word is in YouTube, but Whisper totally missed it.
            # We must assign it a tiny fallback time.
            for o_word in original_words[j1:j2]:
                synced_output.append({
                    "word": o_word["text"],
                    "start": -1, # will fix in pass 2
                    "end": -1
                })
                
    # Pass 2: interpolate missing times (-1)
    for i, item in enumerate(synced_output):
        if item["start"] == -1:
            prev_end = 0.0 if i == 0 else synced_output[i-1]["end"]
            next_start = duration if i == len(synced_output) - 1 else duration
            for j in range(i+1, len(synced_output)):
                if synced_output[j]["start"] != -1:
                    next_start = synced_output[j]["start"]
                    break
            
            # squeeze the missing word exactly into the gap
            item["start"] = prev_end
            item["end"] = min(prev_end + 0.2, next_start)
    
    return synced_output

def _fallback_spacing(text: str, duration: float) -> list[dict]:
    """If Whisper fails, fallback to simple proportional spacing."""
    words = text.split()
    if not words:
        return []
    
    dur_per_word = duration / len(words)
    out = []
    t = 0.0
    for w in words:
        out.append({
            "word": w,
            "start": t,
            "end": t + dur_per_word
        })
        t += dur_per_word
    return out
