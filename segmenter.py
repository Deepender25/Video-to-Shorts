import json
import time
import google.generativeai as genai
from config import GEMINI_API_KEY, GEMINI_MODEL, CHUNK_MINUTES, MAX_RETRIES


SYSTEM_PROMPT = """You are a professional short-form content editor. Your job is to analyze a timestamped transcript and identify the best segments for viral short-form videos (20-60 seconds each).

Rules:
- Each clip MUST be between 20 and 60 seconds long
- Prefer segments with strong hooks that grab attention immediately
- Avoid greetings, introductions, and outros
- Avoid incomplete thoughts — each clip must be self-contained
- Clips should NOT overlap
- Start and end timestamps MUST come directly from the transcript — do NOT invent timestamps
- Return ONLY valid JSON, no commentary or markdown

Output format (strict JSON only):
{
  "clips": [
    {
      "start": <float>,
      "end": <float>,
      "title": "<short engaging title for the clip>",
      "hook": "<opening line that grabs attention>"
    }
  ]
}"""


def _chunk_transcript(formatted_text: str, chunk_minutes: int = CHUNK_MINUTES) -> list[str]:
    """
    Split a formatted transcript into time-based chunks.
    Each chunk covers roughly `chunk_minutes` worth of content.
    """
    lines = formatted_text.strip().split("\n")
    if not lines:
        return []

    chunks = []
    current_chunk = []
    chunk_start_time = None

    for line in lines:
        # Extract start time from "[start - end] text"
        try:
            time_part = line.split("]")[0].replace("[", "").strip()
            start_str = time_part.split("-")[0].strip()
            start_time = float(start_str)
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


def _call_gemini(transcript_text: str) -> list[dict]:
    """
    Send a single transcript chunk to Gemini and parse the response.
    Retries up to MAX_RETRIES on failure.
    """
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)

    user_prompt = f"""Analyze the following timestamped transcript and identify the best clips for short-form video content. Return ONLY valid JSON.

TRANSCRIPT:
{transcript_text}"""

    for attempt in range(MAX_RETRIES):
        try:
            response = model.generate_content(
                [
                    {"role": "user", "parts": [{"text": SYSTEM_PROMPT}]},
                    {"role": "model", "parts": [{"text": "Understood. I will analyze the transcript and return only valid JSON with clip segments."}]},
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
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()

            data = json.loads(raw)
            clips = data.get("clips", [])
            return clips

        except json.JSONDecodeError:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
            raise ValueError(f"Gemini returned invalid JSON after {MAX_RETRIES} attempts.")

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
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

    for chunk in chunks:
        clips = _call_gemini(chunk)
        all_clips.extend(clips)

    return all_clips
