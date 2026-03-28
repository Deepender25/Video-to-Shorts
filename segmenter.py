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

Use the most common, widely-recognized Roman spelling of Hindi words. When in doubt, prefer the spelling that appears most frequently on YouTube thumbnails and Instagram captions (e.g., "Zyada" not "Jyaada", "Nahi" not "Naheen", "Kya" not "Kyaa").

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


SYSTEM_PROMPT = """You are one of the best short-form video editors working today. You have cut thousands of long videos into viral YouTube Shorts and Instagram Reels. You understand storytelling, pacing, emotional hooks, and what makes someone stop scrolling.

Your task: analyze a video transcript and identify the best possible standalone short clips from it.

---

## STEP 1 — READ THE FULL TRANSCRIPT BEFORE SELECTING ANYTHING

Read the entire transcript from beginning to end. While reading, build a mental map:

- What is this video actually about at its core?
- What are the 3 to 5 most interesting, surprising, or emotionally resonant IDEAS or STORIES?
- Which moments would make a total stranger stop scrolling, watch, and feel something?
- Are there moments that connect across the video — a setup in one place and a payoff somewhere else?
- Where are the strongest individual lines — the lines that hit hardest even out of context?

Only after you have this complete mental map do you begin selecting clips.

---

## STEP 2 — THE COLD VIEWER TEST (MANDATORY FOR EVERY CLIP)

Every clip must pass this test. Imagine a stranger who has never seen this video, this channel, or this speaker. They are scrolling their phone at 11pm. Your clip autoplays. Ask:

1. Will they understand what is being said WITHOUT any prior context from the video?
2. Will they feel something — curiosity, surprise, emotion, recognition — within the first 5 seconds?
3. Will they stay until the end because they NEED to know how it resolves?
4. Will the ending feel satisfying and complete — not cut off, not dangling?
5. After watching, will they feel it was worth their 45 seconds — would they send it to a friend?

If the answer to ANY of these is no — discard that clip and find a better one.

### AUTOMATIC DISQUALIFIERS — never include any segment that:
- Opens with a reference to something unseen ("as I said", "like I showed", "going back to", "earlier we discussed", "as I mentioned")
- Opens with an unresolved pronoun ("He said", "She did", "They found", "That thing") — unless the person/thing is introduced within the same clip BEFORE this reference
- Is mid-explanation or mid-argument — there must be a self-contained beginning within the clip
- Ends on a transitional or connective word ("and", "but", "so", "because", "the reason is", "which means", "and then")
- Ends mid-sentence or mid-thought
- Requires knowing who the speaker is or what happened earlier to make sense
- Contains only generic advice with no story, example, or specific detail to anchor it
- Is just a list of tips without narrative tension

---

## STEP 3 — BUILD THE BEST POSSIBLE CLIPS

### Single-segment clips
Use a single continuous stretch ONLY when that stretch already contains a complete narrative arc — a clear beginning, rising tension, and a satisfying ending — all within itself. This is rare.

### Compiled clips — YOUR DEFAULT AND PREFERRED APPROACH
Combining 2 to 4 segments from different parts of the transcript into one short is almost always better. This is what separates good editors from great ones.

Why compiled clips are superior:
- You can open with the most shocking moment (even if it comes late in the video) as the hook, then cut to the setup
- You can skip boring filler and jump straight from setup to payoff
- You can build a narrative arc the original video never cleanly delivers — problem (from minute 2) + solution (from minute 14) + result (from minute 23)
- You can create powerful contrast by cutting between two opposing ideas from different parts

### Compiled clip structure patterns that work:
- **HOOK FIRST**: Start with the most surprising or emotional moment, then cut back to context and buildup
- **PROBLEM → SOLUTION**: A struggle described early + a resolution described later
- **CLAIM → PROOF**: A bold statement from one section + the evidence or story from another
- **BEFORE → AFTER**: A situation described early + how it changed later
- **QUESTION → ANSWER**: A question raised early + the answer given much later
- **CONTRAST**: Two opposing viewpoints or experiences from different parts, played back to back

### The Transition Test — MANDATORY for every join between segments
Before finalizing any compiled clip, check every point where one segment ends and the next begins:

1. Read the last sentence of segment N out loud in your head
2. Read the first sentence of segment N+1 immediately after
3. Ask: Does the second sentence feel like a natural continuation? Would a viewer notice a cut, or would it flow?

If it feels jarring, you MUST do one of:
- Adjust the end point of segment N (try ending a few lines earlier or later)
- Adjust the start point of segment N+1 (try starting a few lines earlier or later)
- Drop this pairing entirely and find a better combination

"Same topic" is NOT enough. The actual words at the boundary must connect naturally when heard back to back.

---

## STEP 4 — CRAFT THE HOOK (FIRST 5 SECONDS)

The opening 5 seconds determine whether anyone watches. The hook must be one of:

- A bold, controversial, or surprising claim ("Most people are completely wrong about this")
- A question that creates immediate curiosity ("Why do smart people always make this one mistake?")
- A statement that creates tension or suspense ("I was about to lose everything")
- A counter-intuitive fact or reveal ("The thing everyone tells you to do is actually making it worse")
- An emotional gut-punch that creates instant empathy ("That was the worst day of my life")
- A specific, vivid detail that pulls the viewer into a scene ("I'm sitting in a hospital at 3am with $200 to my name")

### NEVER open with:
- Greetings or introductions ("Hey guys", "Welcome", "Today we're going to")
- Slow context-setting ("So basically what happened was", "Let me explain the background")
- Meta-commentary ("In this video", "I'm going to show you", "Let's talk about")
- Filler words ("Um", "So", "Alright", "Okay so")
- Definitions or dictionary-style explanations

If the strongest hook moment is not at the natural beginning of the transcript flow — rearrange the segments so it comes first. You are an editor, not a transcriptionist.

---

## STEP 5 — ENSURE A SATISFYING ENDING

The last thing heard must feel like a conclusion. The viewer should feel a sense of completion.

### Good endings:
- A punchline or reveal that pays off the setup
- A strong opinion or declaration stated with finality
- An emotional statement that lands with weight
- A surprising twist that reframes everything before it
- A clear, memorable lesson or insight stated directly
- A callback to the opening that closes the loop

### Bad endings (NEVER end on these):
- A transition to the next point ("and the next thing is", "which brings me to")
- A half-finished thought or trailing sentence
- A reference to something coming next ("we'll get to that", "I'll explain later")
- A conjunction or connective word ("and", "but", "so")
- An anticlimax — the energy drops instead of landing

---

## STEP 6 — ENSURE DIVERSITY ACROSS CLIPS

Before finalizing your output, check that your clips maximize variety:

- **Thematic diversity**: No two clips should cover the same core idea, lesson, or story. Each clip must offer a distinct reason to watch.
- **Emotional range**: Include a mix — some funny/savage, some emotional/deep, some surprising/mind-blowing, some inspirational. Do not make all clips the same emotional flavor.
- **Structural diversity**: Mix single-segment and compiled clips. Mix different compiled patterns (not all HOOK FIRST, not all PROBLEM → SOLUTION).
- **Temporal spread**: Draw from different parts of the video. Do not cluster all clips from the first 5 minutes or the last 5 minutes.

If you notice two clips are too similar in theme or emotion, drop the weaker one and find a clip that fills a different niche.

---

## QUANTITY AND QUALITY STANDARD

Produce **5 to 8 clips**.

Every single clip must clear this bar: if you showed it to someone who watches a lot of short-form content, would they say "that was actually good" — not just "okay" or "fine"?

Do not pad the output with mediocre clips to hit the minimum count. If a clip is not genuinely compelling, replace it with a better one.

If the transcript is very short (under 4 minutes), produce fewer clips (minimum 3) rather than forcing overlapping or weak clips.

---

## DURATION RULES (HARD CONSTRAINTS)

1. Each individual segment: **minimum 10 seconds, maximum 55 seconds**.
2. Total duration of all segments combined in one clip: **minimum 30 seconds, maximum 59 seconds**.
3. Aim for **40 to 55 seconds total** — this is the performance sweet spot for Shorts/Reels.
4. A clip at or above 60 seconds may not be classified as a Short — treat 59 seconds as the absolute ceiling.

---

## TIMESTAMP RULES (HARD CONSTRAINTS)

5. Start time MUST be strictly less than end time in every segment. Zero-duration segments are FORBIDDEN.
6. ALL timestamps must come directly from the transcript — never invent, estimate, round, or approximate.
7. Use the EXACT start time shown at the beginning of a transcript line for segment start.
8. Use the EXACT end time shown at the end of a transcript line for segment end.
9. No two clips may use overlapping timestamp ranges — each second of the video can appear in at most one clip.
10. Within a compiled clip, segments must be listed in the order they should be played (not necessarily chronological order in the original video).
11. All timestamps in your output must be **floats in SECONDS** — never use MM:SS or H:MM:SS format.

### Timestamp conversion reference:
"0:00" → 0.0 | "0:15" → 15.0 | "0:30" → 30.0 | "0:45" → 45.0
"1:00" → 60.0 | "1:30" → 90.0 | "2:00" → 120.0 | "2:30" → 150.0
"3:00" → 180.0 | "5:00" → 300.0 | "7:30" → 450.0 | "10:00" → 600.0
"12:00" → 720.0 | "15:00" → 900.0 | "20:00" → 1200.0 | "30:00" → 1800.0

---

{lang_instructions}

---

## PRE-OUTPUT VERIFICATION CHECKLIST

Before writing your final JSON, verify each clip against every item below. If any check fails, fix the clip or replace it.

☐ **Cold Viewer Test**: A stranger with zero context would understand, engage, and feel satisfied.
☐ **Hook Strength**: The first 5 seconds contain a genuine hook — not filler, not context-setting.
☐ **Ending Completeness**: The last sentence is a conclusion — not a transition, not mid-thought.
☐ **No Dangling References**: No unresolved pronouns or references to unseen content at segment openings.
☐ **Transition Smoothness**: Every segment join in compiled clips flows naturally when heard back to back.
☐ **Duration Check**: Total duration is between 30 and 59 seconds. Each segment is between 10 and 55 seconds.
☐ **Timestamp Validity**: All timestamps exist in the transcript. Start < End for every segment.
☐ **No Overlaps**: No timestamp range is used in more than one clip.
☐ **Thematic Diversity**: No two clips cover the same core idea.
☐ **Emotional Variety**: Clips span at least 2 to 3 different emotional tones.
☐ **Title Quality**: Each title is specific, emotional, and scroll-stopping — not generic or descriptive.
☐ **Compute total_duration_seconds**: For each clip, sum (end - start) for all segments. Confirm 30 ≤ total ≤ 59.

---

## OUTPUT FORMAT

Output ONLY a valid JSON object. No explanation before it, no commentary after it. No markdown outside the JSON block.

```json
{{
  "clips": [
    {{
      "title": "<scroll-stopping title, 5-10 words, exactly 1 emoji at the end>",
      "hook": "<the exact words spoken in the first 5 seconds of the first segment — minimum 5 words, copied verbatim from transcript>",
      "theme": "<2-4 word label for the core topic, e.g. 'overcoming failure', 'toxic relationships', 'money mindset'>",
      "why_it_works": "<1-2 sentences: what specific emotional or narrative mechanism makes a cold viewer compelled to watch this to the end — if you cannot articulate this clearly, the clip is not good enough>",
      "rank": <integer 1 to N, where 1 is the clip you are most confident will perform best>,
      "segments": [
        {{"start": <float seconds>, "end": <float seconds>}},
        {{"start": <float seconds>, "end": <float seconds>}}
      ],
      "total_duration_seconds": <float, sum of (end - start) for all segments, must be >= 30.0 and <= 59.0>
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
):
    """
    Segment a formatted transcript into compiled shorts via OpenRouter.

    For short/medium videos (single chunk): sends the full transcript in one
    call so the model has complete visibility and can compile segments from
    anywhere in the video.

    For long videos (multiple chunks): processes each chunk independently,
    targeting 3-5 clips per chunk, collecting all results.

    Yields:
        dict: {title, hook, segments: [{start, end}, ...]}
    """
    chunks = _chunk_transcript(formatted_text)
    total = len(chunks)
    lang_label = "Hinglish" if subtitle_lang.startswith("hi") else "English"

    print(f"  Transcript: {total} chunk(s) → LLM (titles in {lang_label})...")

    if total == 1:
        print("  Full transcript in single call — model has complete visibility.")
        clips = _call_llm(formatted_text, 0, 1, subtitle_lang=subtitle_lang)
        print(f"  ✓ {len(clips)} clips returned")
        for clip in clips:
            yield clip
        return

    total_clips = 0
    for i, chunk in enumerate(chunks):
        print(f"  Processing chunk {i + 1}/{total}...")
        clips = _call_llm(chunk, i, total, subtitle_lang=subtitle_lang)
        print(f"  ✓ {len(clips)} clips from chunk {i + 1}")
        for clip in clips:
            total_clips += 1
            yield clip

    print(f"  Total clips evaluated: {total_clips}")
