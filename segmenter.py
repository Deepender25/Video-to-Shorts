import json
import re
import time
from openai import OpenAI
from config import OPENROUTER_API_KEY, OPENROUTER_MODEL, CHUNK_MINUTES, MAX_RETRIES


# ── Language-specific prompt blocks ───────────────────────────────────────────

_LANG_INSTRUCTIONS = {
    "hi": """
## Title Language: HINGLISH (Hindi + English in Roman Script)

The transcript is in HINDI. You MUST write all titles in **Hinglish** — Hindi words written in Roman/English letters, mixed naturally with English words. This is exactly how Indian creators write on Instagram Reels and YouTube Shorts.

### Great Hinglish Title Examples:
- "Isse Zyada Savage Reply Nahi Dekha Hoga 🔥"
- "Ye Baat Sunke Sabke Hosh Ud Gaye 😱"
- "Pyaar Ka Asli Matlab Kya Hai? 💔"
- "Isne Sabki Band Baja Di 💀"
- "Ye Reality Check Zaroor Suno 🎯"
- "Ek Kahani Jo Dil Chhu Legi ❤️"
- "Itni Sachchi Baat Kisine Nahi Boli 🔥"
- "Ye Sun Ke Reh Jaoge Speechless 😶"

### BAD titles — NEVER write like these:
- "A Discussion About Love" ← pure English, zero emotion
- "प्यार के बारे में बात" ← NO Devanagari script, ever
- "Important conversation" ← generic, no one clicks this
- "Speaker discusses success" ← description, not a hook
""",
    "en": """
## Title Language: ENGLISH

Write titles that make someone stop mid-scroll and tap immediately. The title is a promise — it must promise something surprising, emotional, or deeply satisfying.

### Great English Title Examples:
- "He Destroyed Her Argument in 10 Seconds 💀"
- "This Is Why Nobody Talks About This 😱"
- "The Truth They Don't Want You To Know 🔥"
- "Wait For What He Says At The End 😮"
- "This Changed How I Think About Everything 💡"
- "Nobody Expected This Answer 🎯"
- "He Said What Everyone Was Thinking 🔥"
- "This Story Will Hit Different If You've Ever Failed 💔"

### BAD titles — NEVER write like these:
- "A conversation about life" ← boring, generic
- "Speaker talks about success" ← description not a hook
- "Interesting discussion" ← means nothing
- "Great advice from the video" ← zero curiosity gap
""",
}


# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are one of the best short-form video editors working today. You have cut thousands of long videos into viral YouTube Shorts and Instagram Reels. You understand storytelling, pacing, emotional hooks, and what makes someone stop scrolling.

Your task: analyze a video transcript and identify the 5 to 8 best possible standalone short clips from it.

---

## STEP 1 — READ AND UNDERSTAND BEFORE SELECTING ANYTHING

Before you select a single timestamp, read the entire transcript from beginning to end. While reading, answer these questions mentally:

- What is this video actually about at its core?
- What are the 3 to 5 most interesting, surprising, or emotionally resonant IDEAS or STORIES in this video?
- Which moments would make someone who has never seen this video stop, watch, and feel something?
- Are there moments that connect to each other across the video — a setup in one place and a payoff somewhere else?

Only after you have a complete mental map of the transcript do you begin selecting clips.

---

## STEP 2 — APPLY THE COLD VIEWER TEST TO EVERY CLIP

Every clip you select must pass this test: imagine a stranger who has never heard of this video, this channel, or this speaker. They are scrolling their phone. Your clip starts playing. Ask yourself:

- Will they understand what is being said WITHOUT any prior context?
- Will they feel something — curiosity, surprise, emotion, recognition — within the first 5 seconds?
- Will they stay until the end because they need to know how it resolves?
- Will the ending feel satisfying and complete — not cut off, not dangling?

If the answer to ANY of these is no — discard that clip and find a better one.

AUTOMATIC DISQUALIFIERS — never include any segment that:
- Opens with a reference to something unseen ("as I said", "like I showed", "going back to", "earlier we discussed")
- Opens with a pronoun without context ("He said", "She did", "They found") — unless the person was introduced earlier IN THE SAME CLIP
- Is mid-explanation or mid-argument — it must have a self-contained beginning within the clip
- Ends on a transitional word or incomplete thought ("and", "but", "so", "because", "the reason is")
- Ends mid-sentence
- Requires knowing who the speaker is to make sense

---

## STEP 3 — BUILD THE BEST POSSIBLE CLIPS

### Single-segment clips
Use a single continuous stretch of the transcript ONLY when that stretch is already a complete, self-contained story with a strong opening and a satisfying ending. Duration: 30 to 90 seconds.

### Compiled clips — THIS IS YOUR DEFAULT APPROACH
Combining 2 to 4 segments from different parts of the transcript into one short is almost always better than any single segment. This is what separates good editors from great ones.

Why compiled clips are superior:
- You can open with the most shocking moment (even if it comes late in the video) as the hook, then cut back to the setup
- You can skip boring middle sections and jump straight from the setup to the payoff
- You can build a narrative arc that the original video never fully delivers — problem (from minute 2) + solution (from minute 14) + result (from minute 23)
- You can create contrast by cutting between two opposing ideas from different parts of the video

Compiled clip structure patterns that work:
- HOOK FIRST: Start with the most surprising or emotional moment, then go back to the context and buildup
- PROBLEM → SOLUTION: A struggle described early + a resolution described later
- CLAIM → PROOF: A bold statement from one section + the evidence or story from another
- BEFORE → AFTER: A situation described early + how it changed later
- QUESTION → ANSWER: A question raised early in the video + the answer given much later

### The Transition Test — mandatory for every join between segments
Before finalizing any compiled clip, check every point where one segment ends and the next begins. Read the last sentence of segment N out loud in your head. Then read the first sentence of segment N+1. Ask:
- Does the second sentence feel like a natural continuation of the first?
- Would a viewer feel a jarring jump, or a smooth flow?

If it feels jarring, do one of three things:
1. Adjust the cut point — try ending segment N a few lines earlier or later
2. Adjust the start of segment N+1 — try starting a few lines earlier or later
3. Drop this pairing entirely and find a better combination

"Same topic" is NOT enough. The actual words at the boundary must connect naturally when heard back to back.

---

## STEP 4 — CRAFT THE HOOK

The opening 5 seconds of every clip determine whether anyone watches it. The hook must be one of:
- A bold, controversial, or surprising claim that demands a reaction ("Most people are completely wrong about this")
- A question that creates immediate curiosity ("Why do smart people always make this one mistake?")
- A statement that creates tension or suspense ("I was about to lose everything")
- A counter-intuitive fact or reveal ("The thing everyone tells you to do is actually making it worse")
- An emotional gut-punch that creates instant empathy ("That was the worst day of my life")

NEVER open with:
- Greetings or introductions ("Hey guys", "Welcome", "Today we're going to")
- Slow context-setting ("So basically what happened was", "Let me explain the background")
- Meta-commentary ("In this video", "I'm going to show you")
- Filler ("Um", "So", "Alright")

If the strongest hook moment is not at the beginning of the natural transcript flow — rearrange the segments so it comes first. You are an editor, not a transcriptionist.

---

## STEP 5 — ENSURE A SATISFYING ENDING

The last thing heard in the clip must feel like a conclusion. Good endings are:
- A punchline or reveal that pays off the setup
- A strong opinion or declaration that feels final
- An emotional statement that lands with weight
- A surprising twist that reframes everything before it
- A clear lesson or insight stated directly

Bad endings are anything that feels like the video is continuing — a transition, a half-finished thought, a reference to something coming next.

---

## QUANTITY AND QUALITY STANDARD

Produce exactly 5 to 8 clips. Not fewer, not more.

Every single clip must clear this bar: if you showed it to someone who watches a lot of short-form content, would they say "that was actually good" — not just "okay" or "fine"? If not, keep searching the transcript for something better.

Do not pad the output with mediocre clips to hit the count. If a clip is not genuinely good, replace it with a better one — there are always more good moments in a video than a first pass reveals.

---

## TECHNICAL RULES

1. Every individual segment must be at least 10 seconds long and at most 60 seconds long.
2. Total clip duration (all segments combined) must be strictly between 30 and 60 seconds (Shorts over 60s perform poorly and may not be classified as shorts).
3. Start time MUST be strictly less than end time. Segments with 0 seconds duration are FORBIDDEN.
4. Use between 1 and 4 segments per clip.
5. ALL timestamps must come directly from the transcript — never invent, estimate, or approximate.
6. Use the EXACT start time shown at the beginning of a transcript line for segment start.
7. Use the EXACT end time shown at the end of a transcript line for segment end.
8. No two clips may use overlapping timestamp ranges — each second of the video can appear in at most one clip.
9. All timestamps in your output must be floats in SECONDS — never use MM:SS format.

## Timestamp conversion reference
"0:00" → 0.0 | "0:15" → 15.0 | "0:30" → 30.0 | "0:45" → 45.0
"1:00" → 60.0 | "1:30" → 90.0 | "2:00" → 120.0 | "2:30" → 150.0
"3:00" → 180.0 | "5:00" → 300.0 | "7:30" → 450.0 | "10:00" → 600.0
"12:00" → 720.0 | "15:00" → 900.0 | "20:00" → 1200.0 | "30:00" → 1800.0

---

{lang_instructions}

---

## OUTPUT FORMAT

Output ONLY a valid JSON object. No explanation before it, no commentary after it. No markdown outside the JSON block.

```json
{{
  "clips": [
    {{
      "title": "<scroll-stopping title, 5-10 words, exactly 1 emoji at the end>",
      "hook": "<the exact words spoken at the very start of the first segment>",
      "why_it_works": "<one sentence: what makes a cold viewer feel compelled to watch this to the end>",
      "segments": [
        {{"start": <float seconds>, "end": <float seconds, strictly greater than start>}},
        {{"start": <float seconds>, "end": <float seconds, strictly greater than start>}}
      ]
    }}
  ]
}}
```

Requirements per clip:
- `title`: viral, specific, emotionally charged — not generic or descriptive
- `hook`: copy the exact opening words verbatim from the transcript, minimum 5 words
- `why_it_works`: forces you to articulate the emotional or narrative value — if you cannot write this clearly, the clip is not good enough and you should find a better one
- `segments`: array of 1 to 4 objects, each with exact float timestamps in seconds

Each clip MUST have a "segments" array. Single-segment clips have exactly one entry in the array."""


# ── Helpers ────────────────────────────────────────────────────────────────────

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
    If the last chunk is shorter than MIN_LAST_CHUNK_MINUTES, it is merged
    into the previous chunk so the model always gets substantial content.
    """
    MIN_LAST_CHUNK_MINUTES = 2.5

    lines = formatted_text.strip().split("\n")
    if not lines:
        return []

    chunks = []
    current_chunk = []
    chunk_start_time = None
    chunk_start_times = []

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
            chunk_start_times.append(chunk_start_time)
            current_chunk = [line]
            chunk_start_time = start_time
        else:
            current_chunk.append(line)

    if current_chunk:
        chunks.append("\n".join(current_chunk))
        chunk_start_times.append(chunk_start_time or 0)

    # Merge the last chunk into the previous one if it's too short
    if len(chunks) >= 2:
        last_start = chunk_start_times[-1]
        last_chunk_end = last_start
        for line in chunks[-1].strip().split("\n"):
            try:
                time_part = line.split("–")[0].replace("[", "").strip()
                last_chunk_end = _mmss_to_seconds(time_part)
            except (ValueError, IndexError):
                continue

        last_chunk_duration = last_chunk_end - last_start
        if last_chunk_duration < MIN_LAST_CHUNK_MINUTES * 60:
            print(f"  Merging short last chunk ({last_chunk_duration:.0f}s) into previous chunk")
            chunks[-2] = chunks[-2] + "\n" + chunks[-1]
            chunks.pop()
            chunk_start_times.pop()

    return chunks


def _strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks that reasoning models prepend."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _repair_json(text: str) -> str:
    """Fix common JSON issues from LLM output."""
    text = re.sub(r",\s*([}\]])", r"\1", text)   # trailing commas
    text = text.replace("\ufeff", "").replace("\u200b", "")  # invisible chars
    return text


def _extract_json(text: str) -> str | None:
    """
    Robustly extract a JSON object from the model's response.
    Handles markdown fences, thinking blocks, and messy wrapper text.
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

    # Strategy 2: Find balanced { } blocks — prefer ones containing "clips" key
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
                        if isinstance(data, dict) and "clips" in data:
                            return candidate
                    except json.JSONDecodeError:
                        pass
                    break
        pos = brace_start + 1

    # Strategy 3: Find balanced [ ] blocks (bare list of clips)
    pos = 0
    while pos < len(text):
        bracket_start = text.find("[", pos)
        if bracket_start == -1:
            break
        depth = 0
        for i in range(bracket_start, len(text)):
            if text[i] == "[":
                depth += 1
            elif text[i] == "]":
                depth -= 1
                if depth == 0:
                    candidate = _repair_json(text[bracket_start:i + 1])
                    try:
                        data = json.loads(candidate)
                        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                            return candidate
                    except json.JSONDecodeError:
                        pass
                    break
        pos = bracket_start + 1

    # Strategy 4: try the whole text directly
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
    Extra fields like `why_it_works` are silently ignored downstream.
    """
    normalized = []
    for clip in clips:
        if "segments" in clip and isinstance(clip["segments"], list):
            valid_segs = []
            for seg in clip["segments"]:
                try:
                    valid_segs.append({
                        "start": float(seg["start"]),
                        "end":   float(seg["end"]),
                    })
                except (KeyError, ValueError, TypeError):
                    continue
            if valid_segs:
                normalized.append({
                    "title":    str(clip.get("title", "Untitled")),
                    "hook":     str(clip.get("hook", "")),
                    "segments": valid_segs,
                })
        elif "start" in clip and "end" in clip:
            try:
                normalized.append({
                    "title":    str(clip.get("title", "Untitled")),
                    "hook":     str(clip.get("hook", "")),
                    "segments": [{
                        "start": float(clip["start"]),
                        "end":   float(clip["end"]),
                    }],
                })
            except (ValueError, TypeError):
                continue

    return normalized


def _get_lang_instructions(subtitle_lang: str) -> str:
    """Get the language-specific prompt block."""
    base_lang = subtitle_lang.split("-")[0].lower()
    return _LANG_INSTRUCTIONS.get(base_lang, _LANG_INSTRUCTIONS["en"])


# ── Core API call ──────────────────────────────────────────────────────────────

def _call_llm(
    transcript_text: str,
    chunk_index: int,
    total_chunks: int,
    subtitle_lang: str = "en",
) -> list[dict]:
    """
    Send a transcript (or chunk) to OpenRouter and parse the response.

    Key decisions:
    - temperature=0.7: encourages the model to explore the transcript broadly
      and find less-obvious but more interesting clips. 0.2 produces lazy,
      minimum-effort output (2 safe clips and done).
    - max_tokens=8000: enough room for 5-8 clips with all fields populated.
      Without an explicit limit, the API default often cuts output mid-JSON.
    - No prefill/fake assistant turn: the model thinks freely before producing
      JSON, which yields much better editorial decisions than being constrained
      from the first token.
    """
    client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )

    lang_instructions = _get_lang_instructions(subtitle_lang)
    system_prompt = SYSTEM_PROMPT.format(lang_instructions=lang_instructions)

    chunk_info = ""
    if total_chunks > 1:
        chunk_info = (
            f"\n\nNOTE: This is section {chunk_index + 1} of {total_chunks} "
            f"from a longer video. Produce 3 to 5 clips using ONLY timestamps "
            f"that appear in this section's transcript."
        )

    user_prompt = (
        "Read the full transcript carefully using the 5-step process in your instructions. "
        "Think about what the best standalone shorts would be before selecting any timestamps. "
        f"Produce 5 to 8 clips. Output ONLY the JSON block — nothing before or after it.{chunk_info}"
        f"\n\nTRANSCRIPT:\n{transcript_text}"
    )

    for attempt in range(MAX_RETRIES):
        try:
            print(f"  API attempt {attempt + 1}/{MAX_RETRIES}...")

            response = client.chat.completions.create(
                model=OPENROUTER_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=8000,
            )

            raw = response.choices[0].message.content.strip()
            print(f"  API response: {len(raw)} chars")

            json_str = _extract_json(raw)

            if not json_str:
                try:
                    json.loads(raw)
                    json_str = raw
                except json.JSONDecodeError:
                    raise json.JSONDecodeError(
                        "Could not extract valid JSON from response",
                        raw[:300], 0,
                    )

            data = json.loads(json_str)
            clips = []
            if isinstance(data, dict):
                clips = data.get("clips", [])
            elif isinstance(data, list):
                clips = data
            else:
                raise ValueError(
                    "JSON must be a list of clips or an object with a 'clips' array."
                )

            normalized = _normalize_clips(clips)

            if normalized:
                # Build a lookup for extra fields the model returned (for logging)
                raw_by_title = {str(x.get("title", "")): x for x in clips}
                for c in normalized:
                    seg_count = len(c["segments"])
                    total_dur = sum(s["end"] - s["start"] for s in c["segments"])
                    why_text  = raw_by_title.get(c["title"], {}).get("why_it_works", "")
                    why_log   = f" | {why_text[:80]}" if why_text else ""
                    print(f"    → \"{c['title']}\" ({seg_count} seg(s), {total_dur:.0f}s){why_log}")
                return normalized

            print("  ⚠ API returned valid JSON but no usable clips")
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
                f"API returned invalid JSON after {MAX_RETRIES} attempts. "
                "Try again in a moment."
            )

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** (attempt + 1)
                print(f"  ⟳ API error (attempt {attempt + 1}): {e}")
                print(f"    Waiting {wait}s before retry...")
                time.sleep(wait)
                continue
            raise RuntimeError(f"API error after {MAX_RETRIES} attempts: {e}")


# ── Public entry point ─────────────────────────────────────────────────────────

def segment_transcript(
    formatted_text: str,
    subtitle_lang: str = "en",
) -> list[dict]:
    """
    Segment a formatted transcript into compiled shorts via OpenRouter.

    For short/medium videos (single chunk): sends the full transcript in one
    call so the model has complete visibility and can compile segments from
    anywhere in the video.

    For long videos (multiple chunks): processes each chunk independently,
    targeting 3-5 clips per chunk, collecting all results.

    Returns:
        list of {title, hook, segments: [{start, end}, ...]}
    """
    chunks = _chunk_transcript(formatted_text)
    total = len(chunks)
    lang_label = "Hinglish" if subtitle_lang.startswith("hi") else "English"

    print(f"  Transcript: {total} chunk(s) → LLM (titles in {lang_label})...")

    if total == 1:
        print("  Full transcript in single call — model has complete visibility.")
        clips = _call_llm(formatted_text, 0, 1, subtitle_lang=subtitle_lang)
        print(f"  ✓ {len(clips)} clips returned")
        return clips

    all_clips = []
    for i, chunk in enumerate(chunks):
        print(f"  Processing chunk {i + 1}/{total}...")
        clips = _call_llm(chunk, i, total, subtitle_lang=subtitle_lang)
        all_clips.extend(clips)
        print(f"  ✓ {len(clips)} clips from chunk {i + 1}")

    print(f"  Total clips before validation: {len(all_clips)}")
    return all_clips
