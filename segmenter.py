import json
import re
import time
import google.generativeai as genai
from config import GEMINI_API_KEY, GEMINI_MODEL, CHUNK_MINUTES, MAX_RETRIES


# ── Language-specific prompt blocks ──────────────────────────────────────

_LANG_INSTRUCTIONS = {
    "hi": """
## Title Language: HINGLISH (Hindi + English in Roman Script)

The transcript is in HINDI. You MUST write all titles in **Hinglish** — that means Hindi words written in English/Roman letters, mixed naturally with English words, exactly how Indian creators write titles on Instagram Reels & YouTube Shorts.

### Examples of GREAT Hinglish Titles:
- "Isse Zyada Savage Reply Nahi Dekha Hoga 🔥"
- "Ye Baat Sunke Sabke Hosh Ud Gaye 😱"
- "Pyaar Ka Asli Matlab Kya Hai? 💔"
- "Isne Sabki Band Baja Di 💀"
- "Ye Reality Check Zaroor Suno 🎯"

### BAD titles (DO NOT write like this):
- "A Discussion About Love" ← too boring, too English
- "प्यार के बारे में बात" ← do NOT use Devanagari script
- "Important conversation" ← generic, no emotion
""",
    "en": """
## Title Language: ENGLISH

The transcript is in ENGLISH. Write catchy, scroll-stopping English titles that would go viral on Instagram Reels & YouTube Shorts.

### Examples of GREAT English Titles:
- "He DESTROYED Her Ego in 10 Seconds 💀"
- "This Life Advice Hits DIFFERENT at 3AM 🎯"
- "Nobody Talks About This But It's SO True 😱"
- "The Most Savage Comeback I've Ever Heard 🔥"
- "This Changed My Entire Perspective on Life 💡"

### BAD titles (DO NOT write like this):
- "A conversation about life" ← boring, generic
- "Speaker talks about success" ← description, not a hook
- "Discussion on relationships" ← nobody clicks this
""",
}


SYSTEM_PROMPT = """You are an expert short-form video editor and viral content curator. You will receive a timestamped dialogue transcript from a YouTube video. Your task is to find the BEST segments that would work as standalone viral short-form videos (Reels, Shorts, TikTok).

## What Makes a VIRAL Short-Form Clip

1. **Irresistible hook in the first 3 seconds** — the opening line must make viewers STOP scrolling
2. **Self-contained story or idea** — the clip must make complete sense on its own, zero context needed
3. **Emotional punch** — funny, shocking, controversial, motivational, OR deeply relatable moments
4. **Clean start and end** — start at the BEGINNING of a thought, end at a NATURAL conclusion or punchline
5. **Curiosity gap** — the viewer should feel compelled to watch till the end

## What to AVOID

- Greetings, introductions, "hey guys", "namaste", "welcome to my channel"
- Outros, subscribe reminders, "like share and subscribe"
- Incomplete thoughts or mid-sentence cuts
- Segments that only make sense with visuals the audio alone cannot convey
- Filler talk, repetitive content, or low-energy moments
- Content that needs the previous or next 10 seconds to make sense

## Strict Rules

- Each clip MUST be between 20 and 60 seconds long
- Timestamps MUST come DIRECTLY from the transcript — do NOT invent or guess timestamps
- Use the EXACT start time from the FIRST dialogue line you include in each clip
- Use the EXACT end time from the LAST dialogue line you include in each clip
- Clips MUST NOT overlap with each other
- Find between 3 and 8 clips depending on video length and content quality
- If the content has no strong standalone moments, return FEWER clips rather than forcing bad ones
- EVERY timestamp must be a number in SECONDS (float), NOT in "MM:SS" format

## Timestamp Conversion Reference

The transcript uses [MM:SS–MM:SS] format. Convert to total seconds:
- "0:00" → 0.0
- "0:45" → 45.0
- "1:30" → 90.0
- "2:15" → 135.0
- "5:00" → 300.0
- "12:45" → 765.0

## Title Guidelines

- Titles MUST be 5-12 words, catchy, and scroll-stopping
- Use power words: DESTROYED, SAVAGE, INSANE, SHOCKING, NOBODY, TRUTH
- Add 1 emoji at the end for visual pop
- Write titles that create a CURIOSITY GAP — make people NEED to watch
- NEVER write boring, descriptive titles like "Speaker discusses topic"

{lang_instructions}

## Output Format

Return ONLY a valid JSON object. No markdown fences. No commentary. No explanation.

{{
  "clips": [
    {{
      "start": <seconds as float>,
      "end": <seconds as float>,
      "title": "<catchy viral title>",
      "hook": "<the exact opening line of the clip from the transcript>"
    }}
  ]
}}"""


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


def _strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks that some models prepend."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _repair_json(text: str) -> str:
    """
    Attempt to repair common JSON issues from LLM output:
    - Trailing commas before } or ]
    - Single quotes → double quotes (careful with apostrophes)
    - Unquoted keys
    """
    # Remove trailing commas: ", }" or ", ]"
    text = re.sub(r",\s*([}\]])", r"\1", text)

    # Remove any BOM or zero-width chars
    text = text.replace("\ufeff", "").replace("\u200b", "")

    return text


def _extract_json(text: str) -> str | None:
    """
    Robustly extract a JSON object from Gemini's response.
    Handles markdown fences, thinking blocks, surrounding text,
    common formatting issues, and multiple brace positions.
    """
    # Pre-clean: strip think blocks
    text = _strip_think_blocks(text)

    # Strategy 1: Find JSON inside markdown code fences
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence_match:
        candidate = _repair_json(fence_match.group(1).strip())
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    # Strategy 2: Find every { and try balanced brace extraction
    # Try all { positions, not just the first one
    pos = 0
    while pos < len(text):
        brace_start = text.find("{", pos)
        if brace_start == -1:
            break

        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = _repair_json(text[brace_start:i + 1])
                    try:
                        data = json.loads(candidate)
                        # Verify it has the expected structure
                        if "clips" in data:
                            return candidate
                    except json.JSONDecodeError:
                        pass
                    break

        pos = brace_start + 1

    # Strategy 3: Try the whole text directly (after repair)
    repaired = _repair_json(text)
    try:
        json.loads(repaired)
        return repaired
    except json.JSONDecodeError:
        pass

    # Log what we got for debugging
    preview = text[:500].replace("\n", " ")
    print(f"  ✗ Could not extract JSON. Response preview: {preview}")
    return None


def _get_lang_instructions(subtitle_lang: str) -> str:
    """Get the language-specific prompt block."""
    # Normalize: "hi", "hi-IN" etc → "hi"
    base_lang = subtitle_lang.split("-")[0].lower()
    return _LANG_INSTRUCTIONS.get(base_lang, _LANG_INSTRUCTIONS["en"])


def _call_gemini(
    transcript_text: str,
    chunk_index: int,
    total_chunks: int,
    subtitle_lang: str = "en",
) -> list[dict]:
    """
    Send a single transcript chunk to Gemini and parse the response.
    Uses response_mime_type to force JSON output.
    Retries up to MAX_RETRIES on failure.
    """
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)

    lang_instructions = _get_lang_instructions(subtitle_lang)
    system_prompt = SYSTEM_PROMPT.format(lang_instructions=lang_instructions)

    chunk_info = ""
    if total_chunks > 1:
        chunk_info = f"\n\nNOTE: This is chunk {chunk_index + 1} of {total_chunks} from a longer video. Focus only on the timestamps in this chunk."

    user_prompt = f"""Analyze the following timestamped dialogue transcript and identify the best clips for short-form video content. Return ONLY valid JSON, no markdown fences, no commentary.{chunk_info}

DIALOGUE TRANSCRIPT:
{transcript_text}"""

    for attempt in range(MAX_RETRIES):
        try:
            print(f"  Gemini attempt {attempt + 1}/{MAX_RETRIES}...")

            response = model.generate_content(
                [
                    {"role": "user", "parts": [{"text": system_prompt}]},
                    {"role": "model", "parts": [{"text": "I will analyze the transcript and return only valid JSON with the best viral clips using exact timestamps."}]},
                    {"role": "user", "parts": [{"text": user_prompt}]},
                ],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.4,
                    max_output_tokens=4096,
                    response_mime_type="application/json",
                ),
            )

            raw = response.text.strip()
            print(f"  Gemini response: {len(raw)} chars")

            # With response_mime_type, the output should be clean JSON
            # But still try extraction as a safety net
            json_str = _extract_json(raw)

            if not json_str:
                # Last resort: try raw directly
                try:
                    json.loads(raw)
                    json_str = raw
                except json.JSONDecodeError:
                    raise json.JSONDecodeError(
                        "Could not extract valid JSON from response",
                        raw[:300], 0
                    )

            data = json.loads(json_str)
            clips = data.get("clips", [])

            # Validate each clip has required fields
            valid_clips = []
            for clip in clips:
                if all(k in clip for k in ("start", "end", "title")):
                    # Ensure start/end are numbers
                    try:
                        clip["start"] = float(clip["start"])
                        clip["end"] = float(clip["end"])
                        valid_clips.append(clip)
                    except (ValueError, TypeError):
                        print(f"  ⚠ Skipping clip with non-numeric timestamps: {clip}")
                        continue

            if valid_clips:
                return valid_clips

            # Got JSON but no valid clips — don't retry, just return empty
            print("  ⚠ Gemini returned valid JSON but no usable clips")
            return []

        except json.JSONDecodeError as e:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** (attempt + 1)
                print(f"  ⟳ Invalid JSON (attempt {attempt + 1}): {str(e)[:100]}")
                print(f"    Waiting {wait}s before retry...")
                time.sleep(wait)
                continue
            print(f"  ✗ Gemini returned invalid JSON after {MAX_RETRIES} attempts")
            print(f"    Last response preview: {raw[:300] if 'raw' in dir() else 'N/A'}")
            raise ValueError(
                f"Gemini returned invalid JSON after {MAX_RETRIES} attempts. "
                "The model may be overloaded — try again in a moment."
            )

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** (attempt + 1)
                print(f"  ⟳ Gemini API error (attempt {attempt + 1}): {e}")
                print(f"    Waiting {wait}s before retry...")
                time.sleep(wait)
                continue
            raise RuntimeError(
                f"Gemini API error after {MAX_RETRIES} attempts: {e}"
            )


def segment_transcript(
    formatted_text: str,
    subtitle_lang: str = "en",
) -> list[dict]:
    """
    Segment a formatted transcript into clips using Gemini API.
    Automatically handles long transcripts by chunking.

    Args:
        formatted_text: The compact MM:SS transcript text
        subtitle_lang: Language code (e.g. "hi", "en") for language-aware titles

    Returns:
        list of {start, end, title, hook}
    """
    chunks = _chunk_transcript(formatted_text)
    all_clips = []
    total = len(chunks)

    lang_label = "Hinglish" if subtitle_lang.startswith("hi") else "English"
    print(f"  Sending {total} chunk(s) to Gemini (titles in {lang_label})...")

    for i, chunk in enumerate(chunks):
        print(f"  Processing chunk {i + 1}/{total}...")
        clips = _call_gemini(chunk, i, total, subtitle_lang=subtitle_lang)
        all_clips.extend(clips)
        print(f"  ✓ Got {len(clips)} clips from chunk {i + 1}")

    return all_clips
