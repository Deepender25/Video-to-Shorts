"""
subtitles.py - Karaoke subtitle engine for Shorts output.

Design principles (v4 - clean):
  - ASS format via FFmpeg `ass` filter (libass handles all Unicode scripts)
  - fontsdir ALWAYS points to C:/Windows/Fonts so libass finds Nirmala UI
    for Devanagari/Hindi without any fontconfig.conf trickery
  - Alignment=2 (bottom-center) + MarginV=180 - simple, proven, stable
  - MAX_WORDS=4 per entry so every subtitle fits on exactly ONE line
    (tested: 4 Hindi words at font 52 = ~800px < 1080px canvas width)
  - WrapStyle=1 (soft-wrap) as a safety net if a word is unusually long
  - MIN_DISPLAY_S=0.9 so no chunk flashes for less than 0.9 seconds
"""

import os
import re


# ── Canvas constants ───────────────────────────────────────────────────────────

CANVAS_W = 1080
CANVAS_H = 1920

# Tuning knobs
MAX_WORDS     = 4    # max words per subtitle entry (guarantees single-line)
MIN_DISPLAY_S = 0.9  # minimum display duration per entry (seconds)
FONT_SIZE     = 54   # font size in ASS units (scaled to 1920px height)

# Alignment=2 = bottom-center; MarginV from the BOTTOM of the canvas.
# 1920 - 300 = 1620px from top -> well inside the lower black bar, noticeably higher.
MARGIN_V = 300

# Windows system fonts directory — passed as fontsdir to libass so it can
# find Nirmala UI (Devanagari, Bengali, Tamil, Telugu…) and Arial without
# any fontconfig.conf.
_WIN_FONTS_DIR = r"C:\Windows\Fonts"

# Font ordered by Unicode coverage
_FONT_NAME = "Nirmala UI"   # Nirmala.ttc — ships with Windows 8+, covers most scripts
_FONT_FALLBACK = "Arial"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_font_name() -> str:
    """Return the logical font name to embed in the ASS style."""
    # Check if Nirmala.ttc / NirmalaUI.ttf exists
    for fname in ("Nirmala.ttc", "NirmalaUI.ttf", "NirmalaUIB.ttf"):
        if os.path.exists(os.path.join(_WIN_FONTS_DIR, fname)):
            return _FONT_NAME
    return _FONT_FALLBACK


def _strip_tags(text: str) -> str:
    """Remove HTML/SRT/ASS tags and normalize whitespace."""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\{[^}]+\}", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _seconds_to_ass(s: float) -> str:
    """Convert seconds to ASS timestamp H:MM:SS.cs"""
    s = max(s, 0.0)
    h  = int(s // 3600)
    m  = int((s % 3600) // 60)
    sc = s % 60
    cs = int(round((sc - int(sc)) * 100))
    return f"{h}:{m:02d}:{int(sc):02d}.{cs:02d}"


def _ffmpeg_path(path: str) -> str:
    """
    Convert a Windows path to FFmpeg filter-string-safe format.
    Forward slashes + colon-after-drive-letter escaped as \\:
    """
    path = path.replace("\\", "/")
    if len(path) >= 2 and path[1] == ":":
        path = path[0] + "\\:" + path[2:]
    return path


# ── Public API ─────────────────────────────────────────────────────────────────

def filter_srt_for_clip(
    words: list[dict],
    clip_start: float,
    clip_end: float,
    time_offset: float = 0.0,
) -> list[dict]:
    """
    Groups precise word-level timings into karaoke chunks (maximum MAX_WORDS),
    and enforces non-overlapping time windows for the ASS engine.

    Args:
        words:         List of words with exact timings: [{"word": "hello", "start": 0.5, "end": 1.0}, ...]
                       These times are expected to be relative to the CLIP START (0.0).
        clip_start:    (Legacy/Unused for absolute bounds if words are already local, kept for compat)
        clip_end:      (Legacy/Unused for absolute bounds if words are already local, kept for compat)
        time_offset:   extra offset for multi-segment clips

    Returns:
        list of {start, end, text, word_timings} in clip-local seconds, non-overlapping
    """
    if not words:
        return []

    raw_chunks = []
    
    # words is already a flattened sequence of chronological words
    for i in range(0, len(words), MAX_WORDS):
        chunk_words = words[i : i + MAX_WORDS]
        if not chunk_words:
            continue
            
        chunk_start = chunk_words[0]["start"] + time_offset
        chunk_end = chunk_words[-1]["end"] + time_offset
        
        # Ensure minimum display time for very fast speech so it doesn't flicker invisibly
        chunk_dur = max(chunk_end - chunk_start, MIN_DISPLAY_S)
        chunk_end = chunk_start + chunk_dur
        
        text = " ".join([w["word"] for w in chunk_words])
        
        # Shift internal word timings for the ASS renderer (which expects relative timings within the chunk)
        # However, write_ass_file now needs to know the exact duration of each word.
        # We'll pass the exact word durations forward.
        timings = []
        for w in chunk_words:
            dur = max(0.01, w["end"] - w["start"])
            timings.append({
                "word": w["word"],
                "duration": dur
            })
            
        raw_chunks.append({
            "start": chunk_start, 
            "end": chunk_end, 
            "text": text,
            "word_timings": timings
        })

    if not raw_chunks:
        return []

    # ── Non-overlap pass ───────────────────────────────────────────────────────
    # Sort by start time, then cap each entry's end at the next entry's start.
    raw_chunks.sort(key=lambda x: x["start"])

    result = []
    for i, chunk in enumerate(raw_chunks):
        if i + 1 < len(raw_chunks):
            capped_end = min(chunk["end"], raw_chunks[i + 1]["start"])
        else:
            capped_end = chunk["end"]

        if capped_end - chunk["start"] >= 0.1:
            result.append({
                "start": chunk["start"],
                "end":   capped_end,
                "text":  chunk["text"],
                "word_timings": chunk["word_timings"]
            })

    return result



def write_ass_file(
    subtitle_entries: list[dict],
    output_path: str,
    font_size: int = FONT_SIZE,
) -> bool:
    """
    Write a libass-compatible ASS subtitle file.

    Font name is set to 'Nirmala UI' which covers Devanagari, Tamil, Telugu,
    Bengali, Gujarati, Gurmukhi, Latin, and more. libass finds it via fontsdir
    (see ass_vf_filter).

    Style:
      - Bold white text with 3px black outline (no background box)
      - Bottom-center alignment (Alignment=2) — simple and proven
      - MarginV=MARGIN_V from the bottom edge

    Returns True if written, False if no entries.
    """
    if not subtitle_entries:
        return False

    font_name = _get_font_name()

    # ASS colour: &HAABBGGRR  (alpha, blue, green, red — reverse of RGB!)
    # Yellow = R=255,G=255,B=0  -> &H00_00_FF_FF
    yellow    = "&H0000FFFF"   # bright yellow — currently spoken word
    white     = "&H00FFFFFF"   # white — other words (not yet spoken)
    black_out = "&H00000000"   # black border
    shadow    = "&HA0000000"   # semi-transparent black shadow

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {CANVAS_W}\n"
        f"PlayResY: {CANVAS_H}\n"
        "ScaledBorderAndShadow: yes\n"
        "WrapStyle: 1\n"          # soft word-wrap — safety net for unexpectedly long words
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        # Fields: Name, Font, Size, Primary, Secondary, Outline, Back,
        #         Bold, Italic, Uline, Strike, ScX, ScY, Spc, Angle,
        #         BdSt, Outl, Shad, Align, ML, MR, MV, Enc
        f"Style: Default,{font_name},{font_size},"
        f"{yellow},{white},{black_out},{shadow},"
        "-1,0,0,0,100,100,0,0,"     # Bold=-1(on), rest off, scale 100%
        "1,3,1,"                    # BorderStyle=1, Outline=3px, Shadow=1px
        f"2,30,30,{MARGIN_V},1\n"   # Alignment=2 (bot-center), MarginL/R=30, MarginV
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    lines = [header]
    for entry in subtitle_entries:
        t_start = _seconds_to_ass(entry["start"])
        t_end   = _seconds_to_ass(entry["end"])

        # Build ASS \k karaoke tags: each word highlighted in yellow (Primary)
        # sequentially while all other words stay white (Secondary).
        # Duration of the entry is divided equally among words.
        karaoke_parts = []
        
        # Determine actual centisecond length for ASS tagging
        # We might have less actual word duration than total entry duration due to the MIN_DISPLAY_S stretching
        # The remaining time stays on the last word
        
        entry_dur_cs = int(round((entry["end"] - entry["start"]) * 100))
        used_cs = 0
        
        for i, word_info in enumerate(entry.get("word_timings", [])):
            word = word_info["word"]
            dur_cs = int(round(word_info["duration"] * 100))
            
            # If this is the last word in the chunk, stretch it to fill the remaining chunk time
            # (Ensures the yellow highlight stays until the subtitle disappears)
            if i == len(entry.get("word_timings", [])) - 1:
                dur_cs = max(dur_cs, entry_dur_cs - used_cs)
                
            used_cs += dur_cs
            
            safe_word = word.replace("{", "\\{").replace("}", "\\}")
            karaoke_parts.append(f"{{\\k{dur_cs}}}{safe_word}")

        karaoke_text = " ".join(karaoke_parts)
        lines.append(f"Dialogue: 0,{t_start},{t_end},Default,,0,0,0,,{karaoke_text}")

    with open(output_path, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(lines))

    return True


def ass_vf_filter(ass_path: str) -> str:
    """
    Build the FFmpeg -vf fragment for ASS subtitle rendering.

    Always passes fontsdir pointing to C:/Windows/Fonts so libass finds
    Nirmala UI (Devanagari), Arial, and all other system fonts without
    needing fontconfig.conf setup.
    """
    safe_ass = _ffmpeg_path(ass_path)

    # Include system fonts dir so libass can find Nirmala UI for Devanagari
    if os.path.isdir(_WIN_FONTS_DIR):
        safe_fonts = _ffmpeg_path(_WIN_FONTS_DIR)
        return f"ass='{safe_ass}':fontsdir='{safe_fonts}'"

    return f"ass='{safe_ass}'"


# ── Compatibility stubs (old callers) ──────────────────────────────────────────

def ass_filter_with_fonts(ass_path: str, _fonts_dir: str = "") -> str:
    """Compatibility alias — use ass_vf_filter instead."""
    return ass_vf_filter(ass_path)


def ass_filter_path(path: str) -> str:
    """Legacy helper — kept for any direct callers."""
    return _ffmpeg_path(path)


def build_drawtext_filter(subtitle_entries: list[dict], font_size: int = FONT_SIZE) -> str:
    """Legacy stub — no longer used."""
    return ""
