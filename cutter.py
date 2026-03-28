import os
import subprocess
import shutil
import re


def _sanitize_filename(name: str) -> str:
    """Remove special characters to make a safe filename."""
    safe = re.sub(r'[^\w\s-]', '', name)
    safe = re.sub(r'\s+', ' ', safe).strip()
    return safe.replace(' ', '_')


def cut_clips(
    video_path: str,
    clips: list[dict],
    output_dir: str,
) -> list[dict]:
    """
    Cut video into shorts using FFmpeg.

    Supports single-segment and multi-segment (compiled) clips.

    Args:
        video_path:    path to the source video
        clips:         list of clips with 'segments' array [{start, end}, ...]
        output_dir:    directory to write output files

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

        # Write directly to output_dir instead of making a subfolder
        video_filename = f"{sanitized_title}.mp4"
        video_output_path = os.path.join(output_dir, video_filename)

        try:
            if len(segments) == 1:
                seg = segments[0]
                success = _cut_single(video_path, seg, video_output_path)
            else:
                success = _cut_and_concat(
                    video_path, segments, video_output_path, output_dir, i,
                )

            if success and os.path.exists(video_output_path) and os.path.getsize(video_output_path) > 0:
                clip_result = clip.copy()
                clip_result["output_path"] = video_output_path
                clip_result["filename"] = video_filename
                results.append(clip_result)

                seg_info = f"{len(segments)} segment(s)"
                print(f"  ✓ Short {i}: {seg_info} → {video_filename}")
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


# ── Single-segment cut ─────────────────────────────────────────────────────────

def _cut_single(
    video_path: str,
    segment: dict,
    output_path: str,
) -> bool:
    """Cut, scale to 9:16 for a single segment. No subtitle burning."""
    duration = segment["end"] - segment["start"]

    cmd = [
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

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
    return proc.returncode == 0


# ── Multi-segment cut + concat ────────────────────────────────────────────────

def _cut_and_concat(
    video_path: str,
    segments: list[dict],
    output_path: str,
    output_dir: str,
    clip_index: int,
) -> bool:
    """
    Cut multiple segments and concatenate them into one video.
    No subtitle burning — each segment is just scaled to 9:16.
    """
    temp_files = []
    concat_list_path = os.path.join(output_dir, f"_concat_{clip_index}.txt")

    try:
        for j, seg in enumerate(segments):
            temp_path = os.path.join(output_dir, f"_temp_{clip_index}_{j}.mp4")
            temp_files.append(temp_path)

            seg_duration = seg["end"] - seg["start"]

            cmd = [
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

            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)

            if proc.returncode != 0:
                print(f"    ✗ Segment {j+1} cut failed: {proc.stderr.decode(errors='ignore')[:200]}")
                return False

        # Write concat list
        with open(concat_list_path, "w", encoding="utf-8") as f:
            for temp_path in temp_files:
                safe_path = temp_path.replace("\\", "/")
                f.write(f"file '{safe_path}'\n")

        # Stream-copy concat
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
        for p in temp_files + [concat_list_path]:
            try:
                os.remove(p)
            except OSError:
                pass
