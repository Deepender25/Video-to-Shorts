import json
import re
import time
import google.generativeai as genai
from config import GEMINI_API_KEY, GEMINI_MODEL, CHUNK_MINUTES, MAX_RETRIES


SYSTEM_PROMPT = """You are an expert short-form video editor. You will receive a timestamped dialogue transcript from a YouTube video. Your task is to find the BEST segments that would work as standalone viral short-form videos (Reels, Shorts, TikTok).

## What Makes a Great Short-Form Clip

1. **Strong hook in the first 3 seconds** — the opening line must grab attention instantly
2. **Self-contained story or idea** — the clip must make complete sense on its own, without needing context from before or after
3. **Emotional punch** — funny moments, surprising facts, controversial opinions, motivational statements, or relatable stories
4. **Clean start and end** — don't cut mid-sentence; start at the beginning of a thought and end at a natural conclusion

## What to AVOID

- Greetings, introductions, "hey guys", "namaste", "welcome to"
- Outros, subscribe reminders, "like share and subscribe"
- Incomplete thoughts or mid-sentence cuts
- Segments that only make sense with visual context that the audio alone cannot convey
- Filler talk with no substance

## Rules

- Each clip MUST be between 20 and 60 seconds long
- Timestamps MUST come directly from the transcript — do NOT invent or approximate timestamps
- Use the EXACT start time from the beginning of the first dialogue line in your clip
- Use the EXACT end time from the last dialogue line in your clip
- Clips MUST NOT overlap with each other
- Find between 3 and 8 clips depending on video length and content quality
- If the content has no strong standalone moments, return fewer clips rather than forcing bad ones

## Timestamp Format

The transcript uses MM:SS format (e.g., "1:30" means 1 minute 30 seconds). Convert these to total seconds in your output:
- "0:00" → 0
- "1:30" → 90  
- "12:45" → 765

## Output Format (strict JSON only, no markdown, no commentary)

{
  "clips": [
    {
      "start": <seconds as float>,
      "end": <seconds as float>,
      "title": "<catchy 5-10 word title for Instagram/YouTube>",
      "hook": "<the exact opening line of the clip from the transcript>"
    }
  ]
}"""


def _mmss_to_seconds(mmss: str) -> float:
    """Convert MM:SS string to seconds."""
    parts = mmss.strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    return 0.0


def _chunk_transcript(formatted_text: str, chunk_minutes: int = CHUNK_MINUTES) -> list[str]:
    """
    Split a formatted transcript into time-based chunks.
    Each chunk covers roughly `chunk_minutes` worth of content.
    Works with the compact MM:SS–MM:SS format.
    """
    lines = formatted_text.strip().split("\n")
    if not lines:
        return []

    chunks = []
    current_chunk = []
    chunk_start_time = None

    for line in lines:
        # Extract start time from "[M:SS–M:SS] text" format
        try:
            time_part = line.split("–")[0].replace("[", "").strip()
            start_time = _mmss_to_seconds(time_part)
        except (ValueError, IndexError):
            current_chunk.append(line)
            continue

        if chunk_start_time is None:
            chunk_start_time = start_time

        # Check if we've exceeded the chunk duration
        if start_time - chunk_start_time >= chunk_minutes * 60 and current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = [line]
            chunk_start_time = start_time
        else:
            current_chunk.append(line)

    # Don't forget the last chunk
    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


def _call_gemini(transcript_text: str, chunk_index: int, total_chunks: int) -> list[dict]:
    """
    Send a single transcript chunk to Gemini and parse the response.
    Retries up to MAX_RETRIES on failure.
    """
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)

    chunk_info = ""
    if total_chunks > 1:
        chunk_info = f"\n\nNOTE: This is chunk {chunk_index + 1} of {total_chunks} from a longer video. Focus only on the timestamps in this chunk."

    user_prompt = f"""Analyze the following timestamped dialogue transcript and identify the best clips for short-form video content. Return ONLY valid JSON, no markdown fences, no commentary.{chunk_info}

DIALOGUE TRANSCRIPT:
{transcript_text}"""

    for attempt in range(MAX_RETRIES):
        try:
            response = model.generate_content(
                [
                    {"role": "user", "parts": [{"text": SYSTEM_PROMPT}]},
                    {"role": "model", "parts": [{"text": "Understood. I will carefully analyze the transcript, identify the strongest standalone moments based on the criteria, and return only valid JSON with clip segments using exact timestamps from the transcript."}]},
                    {"role": "user", "parts": [{"text": user_prompt}]},
                ],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=4096,
                ),
            )

            raw = response.text.strip()

            # Strip markdown code fences if present
            if raw.startswith("```"):
                # Remove opening fence (could be ```json or just ```)
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

            data = json.loads(raw)
            clips = data.get("clips", [])

            # Validate each clip has required fields
            valid_clips = []
            for clip in clips:
                if all(k in clip for k in ("start", "end", "title")):
                    valid_clips.append(clip)

            return valid_clips

        except json.JSONDecodeError:
            if attempt < MAX_RETRIES - 1:
                print(f"  ⟳ Gemini returned invalid JSON (attempt {attempt + 1}), retrying...")
                time.sleep(2 ** attempt)
                continue
            raise ValueError(f"Gemini returned invalid JSON after {MAX_RETRIES} attempts.")

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"  ⟳ Gemini API error (attempt {attempt + 1}): {e}, retrying...")
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"Gemini API error after {MAX_RETRIES} attempts: {e}")


def segment_transcript(formatted_text: str) -> list[dict]:
    """
    Segment a formatted transcript into clips using Gemini API.
    Automatically handles long transcripts by chunking.

    Returns:
        list of {start, end, title, hook}
    """
    chunks = _chunk_transcript(formatted_text)
    all_clips = []
    total = len(chunks)

    print(f"  Sending {total} chunk(s) to Gemini...")

    for i, chunk in enumerate(chunks):
        print(f"  Processing chunk {i + 1}/{total}...")
        clips = _call_gemini(chunk, i, total)
        all_clips.extend(clips)
        print(f"  ✓ Got {len(clips)} clips from chunk {i + 1}")

    return all_clips
