import json
import re
import time
import google.generativeai as genai
from config import GEMINI_API_KEY, GEMINI_MODEL, CHUNK_MINUTES, MAX_RETRIES


# ── Language-specific prompt blocks ──────────────────────────────────────

_LANG_INSTRUCTIONS = {
    "hi": """
## Title Language: HINGLISH (Hindi + English in Roman Script)

The transcript is in HINDI. You MUST write all titles in **Hinglish** — Hindi words in Roman/English letters, mixed with English words, exactly how Indian creators write on Instagram Reels & YouTube Shorts.

### Great Hinglish Title Examples:
- "Isse Zyada Savage Reply Nahi Dekha Hoga 🔥"
- "Ye Baat Sunke Sabke Hosh Ud Gaye 😱"
- "Pyaar Ka Asli Matlab Kya Hai? 💔"
- "Isne Sabki Band Baja Di 💀"
- "Ye Reality Check Zaroor Suno 🎯"
- "Ek Kahani Jo Dil Chhu Legi ❤️"

### BAD titles (NEVER write like this):
- "A Discussion About Love" ← too English, too boring
- "प्यार के बारे में बात" ← NO Devanagari script ever
- "Important conversation" ← generic, zero emotion
""",
    "en": """
## Title Language: ENGLISH

Write catchy, scroll-stopping English titles for Instagram Reels & YouTube Shorts.

### Great English Title Examples:
- "He DESTROYED Her Ego in 10 Seconds 💀"
- "This Life Advice Hits DIFFERENT at 3AM 🎯"
- "Nobody Talks About This But It's SO True 😱"
- "The Most Savage Comeback I've Ever Heard 🔥"
- "This Story Will Change How You Think Forever 💡"
- "Wait For The Plot Twist at The End 😮"

### BAD titles (NEVER write like this):
- "A conversation about life" ← boring, generic
- "Speaker talks about success" ← description, not a hook
- "Discussion on relationships" ← nobody clicks this
""",
}


SYSTEM_PROMPT = """You are a world-class short-form video editor who creates VIRAL YouTube Shorts and Instagram Reels. You specialize in **storytelling** — turning long videos into compelling mini-stories that keep viewers hooked from first to last second.

## Your Superpower: Storytelling Through Compilation

You don't just cut random interesting moments. You CREATE STORIES by:

1. **Compiling multiple segments** from different parts of the video into one cohesive short
2. **Building narrative arcs**: Hook → Context → Build-up → Climax/Payoff
3. **Connecting the dots**: Taking related moments scattered across the video and weaving them into one powerful short

## Two Types of Shorts You Create

### Type 1: Single-Segment Shorts (15–60 seconds)
A continuous clip that naturally tells a complete story on its own.
- Use when a single moment is already powerful and self-contained
- Still needs a strong hook and natural conclusion

### Type 2: Compiled Shorts (30–120 seconds) ⭐ PREFERRED
Multiple segments stitched together to create a BETTER story than any single segment could tell.
- **Combine 2-4 segments** from different parts of the video
- Each segment must flow naturally into the next (same topic/theme)
- The combined short must feel like ONE cohesive narrative
- Total duration of all segments combined: 30 to 120 seconds

**Example**: For a video about someone's life story:
- Segment 1 (0:30–0:55): The struggle/problem they faced
- Segment 2 (3:15–3:45): The turning point/realization
- Segment 3 (7:00–7:20): The result/transformation
→ Together = a 75-second mini-documentary that tells a complete arc

## What Makes Content Go VIRAL

1. **Irresistible hook** — first 3 seconds must STOP the scroll
2. **Emotional journey** — take viewers through feelings (curiosity → shock, sadness → hope, confusion → clarity)
3. **Payoff at the end** — every short needs a satisfying conclusion or punchline
4. **Relatability** — moments viewers can see themselves in
5. **Curiosity gap** — create the NEED to watch till the end

## What to AVOID

- Greetings, intros, "hey guys", "namaste", "welcome"
- Outros, subscribe reminders, "like share subscribe"
- Mid-sentence cuts — always start AND end at natural speech boundaries
- Segments that need visual context the audio can't convey
- Filler, repetition, or low-energy moments
- Shorts where the segments feel disconnected or jarring when combined

## STRICT RULES

1. Each individual segment must be at least 8 seconds long
2. Total short duration (all segments combined) must be 15–120 seconds
3. Timestamps MUST come DIRECTLY from the transcript — NEVER invent timestamps
4. Use the EXACT start time from the first line of each segment
5. Use the EXACT end time from the last line of each segment
6. Segments within a short must be from the SAME topic/theme
7. Shorts MUST NOT have overlapping segments with other shorts
8. Find 2–6 shorts depending on content quality
9. PREFER compiled multi-segment shorts over single-segment cuts
10. Every timestamp must be a **number in SECONDS** (float), NOT "MM:SS"

## Timestamp Conversion

Transcript uses [MM:SS–MM:SS]. Convert to seconds:
- "0:00" → 0.0, "0:45" → 45.0, "1:30" → 90.0
- "2:15" → 135.0, "5:00" → 300.0, "12:45" → 765.0

## Title Rules

- 5–12 words, catchy, scroll-stopping
- Use power words: DESTROYED, SAVAGE, INSANE, SHOCKING, NOBODY, TRUTH
- Add 1 emoji at the end
- Create a CURIOSITY GAP — make people NEED to watch
- NEVER write boring descriptive titles

{lang_instructions}

## Output Format (STRICT JSON, nothing else)

{{
  "clips": [
    {{
      "title": "<viral title>",
      "hook": "<exact opening line from transcript>",
      "segments": [
        {{"start": <seconds>, "end": <seconds>}},
        {{"start": <seconds>, "end": <seconds>}}
      ]
    }}
  ]
}}

Each clip MUST have a "segments" array (even single-segment clips should have exactly one entry in the array)."""


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
    """
    lines = formatted_text.strip().split("\n")
    if not lines:
        return []

    chunks = []
    current_chunk = []
    chunk_start_time = None

    for line in lines:
        try:
            time_part = line.split("–")[0].replace("[", "").strip()
            start_time = _mmss_to_seconds(time_part)
        except (ValueError, IndexError):
            current_chunk.append(line)
            continue

        if chunk_start_time is None:
            chunk_start_time = start_time

        if start_time - chunk_start_time >= chunk_minutes * 60 and current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = [line]
            chunk_start_time = start_time
        else:
            current_chunk.append(line)

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


def _strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks that some models prepend."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _repair_json(text: str) -> str:
    """Fix common JSON issues from LLM output."""
    text = re.sub(r",\s*([}\]])", r"\1", text)  # trailing commas
    text = text.replace("\ufeff", "").replace("\u200b", "")  # invisible chars
    return text


def _extract_json(text: str) -> str | None:
    """
    Robustly extract a JSON object from Gemini's response.
    Handles markdown fences, thinking blocks, and multiple brace positions.
    """
    text = _strip_think_blocks(text)

    # Strategy 1: JSON inside markdown code fences
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence_match:
        candidate = _repair_json(fence_match.group(1).strip())
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    # Strategy 2: Find balanced { } blocks, prefer ones with "clips" key
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
                        if "clips" in data:
                            return candidate
                    except json.JSONDecodeError:
                        pass
                    break

        pos = brace_start + 1

    # Strategy 3: whole text directly
    repaired = _repair_json(text)
    try:
        json.loads(repaired)
        return repaired
    except json.JSONDecodeError:
        pass

    preview = text[:500].replace("\n", " ")
    print(f"  ✗ Could not extract JSON. Response preview: {preview}")
    return None


def _normalize_clips(clips: list[dict]) -> list[dict]:
    """
    Normalize clip format: ensure every clip has a `segments` array.
    Handles both old format {start, end} and new format {segments: [...]}.
    """
    normalized = []
    for clip in clips:
        if "segments" in clip and isinstance(clip["segments"], list):
            # New format — validate each segment has start/end
            valid_segs = []
            for seg in clip["segments"]:
                try:
                    valid_segs.append({
                        "start": float(seg["start"]),
                        "end": float(seg["end"]),
                    })
                except (KeyError, ValueError, TypeError):
                    continue
            if valid_segs:
                normalized.append({
                    "title": str(clip.get("title", "Untitled")),
                    "hook": str(clip.get("hook", "")),
                    "segments": valid_segs,
                })
        elif "start" in clip and "end" in clip:
            # Old format — convert to segments array
            try:
                normalized.append({
                    "title": str(clip.get("title", "Untitled")),
                    "hook": str(clip.get("hook", "")),
                    "segments": [{
                        "start": float(clip["start"]),
                        "end": float(clip["end"]),
                    }],
                })
            except (ValueError, TypeError):
                continue

    return normalized


def _get_lang_instructions(subtitle_lang: str) -> str:
    """Get the language-specific prompt block."""
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
    Returns normalized clips with segments arrays.
    """
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)

    lang_instructions = _get_lang_instructions(subtitle_lang)
    system_prompt = SYSTEM_PROMPT.format(lang_instructions=lang_instructions)

    chunk_info = ""
    if total_chunks > 1:
        chunk_info = (
            f"\n\nNOTE: This is chunk {chunk_index + 1} of {total_chunks} "
            "from a longer video. Focus only on timestamps in this chunk. "
            "Prefer compiled multi-segment shorts that tell a story arc."
        )

    user_prompt = f"""Analyze this transcript and create the most VIRAL short-form video content possible. Prefer COMPILED shorts that combine multiple segments into a storytelling arc. Return ONLY valid JSON.{chunk_info}

DIALOGUE TRANSCRIPT:
{transcript_text}"""

    for attempt in range(MAX_RETRIES):
        try:
            print(f"  Gemini attempt {attempt + 1}/{MAX_RETRIES}...")

            response = model.generate_content(
                [
                    {"role": "user", "parts": [{"text": system_prompt}]},
                    {"role": "model", "parts": [{"text": "I will analyze the transcript for storytelling arcs, find the best segments to compile into viral shorts, and return only valid JSON with the segments array format."}]},
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

            json_str = _extract_json(raw)

            if not json_str:
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

            # Normalize to segments format
            normalized = _normalize_clips(clips)

            if normalized:
                # Log what we got
                for c in normalized:
                    seg_count = len(c["segments"])
                    total_dur = sum(s["end"] - s["start"] for s in c["segments"])
                    print(f"    → \"{c['title']}\" ({seg_count} segment(s), {total_dur:.0f}s)")
                return normalized

            print("  ⚠ Gemini returned valid JSON but no usable clips")
            return []

        except json.JSONDecodeError as e:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** (attempt + 1)
                print(f"  ⟳ Invalid JSON (attempt {attempt + 1}): {str(e)[:100]}")
                print(f"    Waiting {wait}s before retry...")
                time.sleep(wait)
                continue
            print(f"  ✗ Invalid JSON after {MAX_RETRIES} attempts")
            raise ValueError(
                f"Gemini returned invalid JSON after {MAX_RETRIES} attempts. "
                "Try again in a moment."
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
    Segment a formatted transcript into compiled shorts using Gemini API.

    Returns:
        list of {title, hook, segments: [{start, end}, ...]}
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
        print(f"  ✓ Got {len(clips)} shorts from chunk {i + 1}")

    return all_clips
