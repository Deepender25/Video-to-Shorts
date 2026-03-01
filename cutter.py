import os
import subprocess
import shutil
import re
from subtitles import filter_srt_for_clip, write_ass_file, ass_vf_filter


def _sanitize_filename(name: str) -> str:
    """Remove special characters to make a safe filename."""
    safe = re.sub(r'[^\w\s-]', '', name)
    safe = re.sub(r'\s+', ' ', safe).strip()
    return safe.replace(' ', '_')


def cut_clips(
    video_path: str,
    clips: list[dict],
    output_dir: str,
    srt_segments: list[dict] | None = None,
) -> list[dict]:
    """
    Cut video into shorts using FFmpeg.

    Supports single-segment and multi-segment (compiled) clips.
    If srt_segments is provided, karaoke subtitles (3 words at a time) are
    burned into the black padding zone below the main video using an ASS
    subtitle file — which correctly renders all Unicode scripts including
    Devanagari, Tamil, Arabic, etc.

    Args:
        video_path:    path to the source video
        clips:         list of clips with 'segments' array [{start, end}, ...]
        output_dir:    directory to write output files
        srt_segments:  parsed SRT [{start, end, text}] for subtitle burning

    Returns:
        clips list with added 'output_path' and 'filename' keys
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for i, clip in enumerate(clips, 1):
        segments = clip.get("segments", [])
        if not segments:
            continue

        raw_title = clip.get("title") or f"short_{i}"
        if raw_title == "Untitled":
            raw_title = f"short_{i}"

        sanitized_title = _sanitize_filename(raw_title) or f"short_{i}"

        clip_folder = os.path.join(output_dir, sanitized_title)
        os.makedirs(clip_folder, exist_ok=True)

        video_filename = f"{sanitized_title}.mp4"
        video_output_path = os.path.join(clip_folder, video_filename)

        try:
            if len(segments) == 1:
                seg = segments[0]
                
                entries = []
                if srt_segments:
                    # Gather original text for this window
                    text_parts = [
                        s.get("text", "") for s in srt_segments 
                        if s["start"] < seg["end"] and s["end"] > seg["start"]
                    ]
                    orig_text = " ".join(text_parts)
                    
                    # Run precise word alignment via Whisper + SequenceMatcher
                    import sync
                    aligned_words = sync.sync_subtitles(
                        video_path=video_path,
                        clip_start=seg["start"],
                        clip_end=seg["end"],
                        original_youtube_text=orig_text,
                        temp_dir=output_dir
                    )
                    
                    # Convert exact word timings into Karaoke chunk entries
                    entries = filter_srt_for_clip(aligned_words, seg["start"], seg["end"])

                ass_path = os.path.join(output_dir, f"_sub_{i}_0.ass")
                success = _cut_single(video_path, seg, video_output_path, entries, ass_path)
            else:
                success = _cut_and_concat(
                    video_path, segments, video_output_path, output_dir, i,
                    srt_segments=srt_segments,
                )

            if success and os.path.exists(video_output_path) and os.path.getsize(video_output_path) > 0:
                metadata_path = os.path.join(clip_folder, f"{sanitized_title}.txt")
                with open(metadata_path, "w", encoding="utf-8") as f:
                    f.write(f"Title: {clip.get('title')}\n")
                    f.write(f"Hook: {clip.get('hook')}\n")
                    f.write(f"Duration: {clip.get('duration', 0)} seconds\n")

                zip_path_without_ext = clip_folder
                shutil.make_archive(zip_path_without_ext, 'zip', clip_folder)
                try:
                    shutil.rmtree(clip_folder)
                except OSError:
                    pass

                zip_filename = f"{sanitized_title}.zip"
                final_output_path = f"{clip_folder}.zip"

                clip_result = clip.copy()
                clip_result["output_path"] = final_output_path
                clip_result["filename"] = zip_filename
                results.append(clip_result)

                seg_info = f"{len(segments)} segment(s)"
                sub_info = " + subtitles" if srt_segments else ""
                print(f"  ✓ Short {i}: {seg_info}{sub_info} → {zip_filename}")
            else:
                print(f"  ✗ Short {i}: failed to produce output")

        except Exception as e:
            print(f"  ✗ Short {i} error: {e}")
            continue

    return results


# ── Base video filter (scale + letterbox pad to 9:16) ─────────────────────────

_BASE_VF = (
    "scale=1080:1920:force_original_aspect_ratio=decrease,"
    "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black"
)


def _build_vf(ass_path: str | None) -> str:
    """Return the full -vf filter chain, optionally with ASS subtitle overlay."""
    if not ass_path or not os.path.exists(ass_path):
        return _BASE_VF
    return f"{_BASE_VF},{ass_vf_filter(ass_path)}"


# ── Single-segment cut ─────────────────────────────────────────────────────────

def _cut_single(
    video_path: str,
    segment: dict,
    output_path: str,
    subtitle_entries: list[dict],
    ass_path: str,
) -> bool:
    """Cut, scale to 9:16, and burn karaoke subtitles into a single segment."""
    has_subs = write_ass_file(subtitle_entries, ass_path)
    vf = _build_vf(ass_path if has_subs else None)
    duration = segment["end"] - segment["start"]

    # Sync fix: -ss before -i = fast input seek; -t as output option = exact duration.
    # This guarantees t=0 in the output aligns to exactly segment["start"] in source.
    # Old approach (-ss -to both before -i + -avoid_negative_ts) could offset by up
    # to one keyframe interval (1-5s), causing subtitle-audio mismatch.
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(segment["start"]),   # fast input seek
        "-i", video_path,
        "-t", str(duration),            # output duration (exact)
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        output_path,
    ]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)

    if proc.returncode != 0 and has_subs:
        stderr = proc.stderr.decode(errors="ignore")
        print(f"    ⚠ Subtitle rendering failed ({stderr[:120].strip()}), retrying without subtitles...")
        cmd_plain = [
            "ffmpeg", "-y",
            "-ss", str(segment["start"]),
            "-i", video_path,
            "-t", str(duration),
            "-vf", _BASE_VF,
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            output_path,
        ]
        proc = subprocess.run(cmd_plain, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)

    try:
        os.remove(ass_path)
    except OSError:
        pass

    return proc.returncode == 0


# ── Multi-segment cut + concat ────────────────────────────────────────────────

def _cut_and_concat(
    video_path: str,
    segments: list[dict],
    output_path: str,
    output_dir: str,
    clip_index: int,
    srt_segments: list[dict] | None = None,
) -> bool:
    """
    Cut multiple segments (each with subtitles burned in at local t=0),
    then concatenate them into one seamless video.
    """
    temp_files = []
    temp_ass_files = []
    concat_list_path = os.path.join(output_dir, f"_concat_{clip_index}.txt")

    try:
        for j, seg in enumerate(segments):
            temp_path = os.path.join(output_dir, f"_temp_{clip_index}_{j}.mp4")
            ass_path  = os.path.join(output_dir, f"_sub_{clip_index}_{j}.ass")
            temp_files.append(temp_path)
            temp_ass_files.append(ass_path)

            entries = []
            if srt_segments:
                text_parts = [
                    s.get("text", "") for s in srt_segments 
                    if s["start"] < seg["end"] and s["end"] > seg["start"]
                ]
                orig_text = " ".join(text_parts)
                import sync
                aligned_words = sync.sync_subtitles(
                    video_path=video_path,
                    clip_start=seg["start"],
                    clip_end=seg["end"],
                    original_youtube_text=orig_text,
                    temp_dir=output_dir
                )
                # No time_offset needed here, as the chunks start at 0 relative to themselves
                entries = filter_srt_for_clip(aligned_words, seg["start"], seg["end"], time_offset=0.0)

            has_subs = write_ass_file(entries, ass_path)
            vf = _build_vf(ass_path if has_subs else None)
            seg_duration = seg["end"] - seg["start"]

            cmd = [
                "ffmpeg", "-y",
                "-ss", str(seg["start"]),   # fast input seek
                "-i", video_path,
                "-t", str(seg_duration),    # exact output duration
                "-vf", vf,
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart",
                temp_path,
            ]

            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)

            if proc.returncode != 0:
                stderr = proc.stderr.decode(errors="ignore")
                if has_subs:
                    print(f"    ⚠ Subtitle render failed on segment {j+1}, retrying without subtitles...")
                    cmd_plain = [
                        "ffmpeg", "-y",
                        "-ss", str(seg["start"]),
                        "-i", video_path,
                        "-t", str(seg_duration),
                        "-vf", _BASE_VF,
                        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                        "-c:a", "aac", "-b:a", "192k",
                        "-movflags", "+faststart",
                        temp_path,
                    ]
                    proc = subprocess.run(cmd_plain, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)

                if proc.returncode != 0:
                    print(f"    ✗ Segment {j+1} cut failed: {proc.stderr.decode(errors='ignore')[:200]}")
                    return False

        # Write concat list
        with open(concat_list_path, "w", encoding="utf-8") as f:
            for temp_path in temp_files:
                safe_path = temp_path.replace("\\", "/")
                f.write(f"file '{safe_path}'\n")

        # Stream-copy concat (segments already encoded uniformly)
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_list_path,
            "-c", "copy",
            "-movflags", "+faststart",
            output_path,
        ]

        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
        if proc.returncode != 0:
            print(f"    ✗ Concat failed: {proc.stderr.decode(errors='ignore')[:200]}")
            return False

        return True

    finally:
        for p in temp_files + temp_ass_files + [concat_list_path]:
            try:
                os.remove(p)
            except OSError:
                pass
