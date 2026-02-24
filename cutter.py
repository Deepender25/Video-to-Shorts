import os
import subprocess
import tempfile
import shutil
import re

def _sanitize_filename(name: str) -> str:
    """Remove weird characters to make a safe filename/foldername."""
    # Keep alphanumeric, spaces, and dashes
    safe = re.sub(r'[^\w\s-]', '', name)
    # Replace multiple spaces with a single space
    safe = re.sub(r'\s+', ' ', safe).strip()
    return safe.replace(' ', '_')


def cut_clips(video_path: str, clips: list[dict], output_dir: str) -> list[dict]:
    """
    Cut video into shorts using FFmpeg.

    Supports both single-segment and multi-segment (compiled) clips.
    - Single segment: cuts and applies 9:16 crop filter
    - Multi segment: cuts each segment with 9:16 crop filter, then concatenates

    Args:
        video_path: path to the source video
        clips: list of clips with 'segments' array [{start, end}, ...]
        output_dir: directory to write output files

    Returns:
        clips list with added 'output_path' and 'filename' keys
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for i, clip in enumerate(clips, 1):
        segments = clip.get("segments", [])
        if not segments:
            continue

        raw_title = clip.get("title")
        if not raw_title or raw_title == "Untitled":
            raw_title = f"short_{i}"
            
        sanitized_title = _sanitize_filename(raw_title)
        if not sanitized_title:
            sanitized_title = f"short_{i}"
            
        clip_folder = os.path.join(output_dir, sanitized_title)
        os.makedirs(clip_folder, exist_ok=True)
        
        video_filename = f"{sanitized_title}.mp4"
        video_output_path = os.path.join(clip_folder, video_filename)

        try:
            if len(segments) == 1:
                # Single segment — cut and crop to 9:16
                success = _cut_single(video_path, segments[0], video_output_path)
            else:
                # Multi-segment — cut + concatenate
                success = _cut_and_concat(video_path, segments, video_output_path, output_dir, i)

            if success and os.path.exists(video_output_path) and os.path.getsize(video_output_path) > 0:
                # Write metadata text file
                metadata_path = os.path.join(clip_folder, f"{sanitized_title}.txt")
                with open(metadata_path, "w", encoding="utf-8") as f:
                    f.write(f"Title: {clip.get('title')}\n")
                    f.write(f"Hook: {clip.get('hook')}\n")
                    f.write(f"Duration: {clip.get('duration', 0)} seconds\n")
                    
                # Zip the folder contents
                zip_path_without_ext = clip_folder
                shutil.make_archive(zip_path_without_ext, 'zip', clip_folder)
                
                # Clean up the unzipped folder to save space
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
                print(f"  ✓ Short {i}: {seg_info} → {zip_filename}")
            else:
                print(f"  ✗ Short {i}: failed to produce output")

        except Exception as e:
            print(f"  ✗ Short {i} error: {e}")
            continue

    return results


def _cut_single(video_path: str, segment: dict, output_path: str) -> bool:
    """Cut a single continuous segment and format as 9:16 Shorts."""
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(segment["start"]),
        "-to", str(segment["end"]),
        "-i", video_path,
        "-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "192k",
        "-avoid_negative_ts", "make_zero",
        "-movflags", "+faststart",
        output_path,
    ]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
    )
    return proc.returncode == 0


def _cut_and_concat(
    video_path: str,
    segments: list[dict],
    output_path: str,
    output_dir: str,
    clip_index: int,
) -> bool:
    """
    Cut multiple segments and concatenate them into one seamless video.

    Uses FFmpeg concat demuxer for lossless joining.
    Steps:
      1. Cut each segment to a temp file (re-encode for consistent format)
      2. Write a concat list file
      3. Concatenate all segments
      4. Clean up temp files
    """
    temp_files = []
    concat_list_path = os.path.join(output_dir, f"_concat_{clip_index}.txt")

    try:
        # Step 1: Cut each segment to a temp file
        # We re-encode to ensure consistent codec/format for seamless concat
        for j, seg in enumerate(segments):
            temp_path = os.path.join(output_dir, f"_temp_{clip_index}_{j}.mp4")
            temp_files.append(temp_path)

            cmd = [
                "ffmpeg",
                "-y",
                "-ss", str(seg["start"]),
                "-to", str(seg["end"]),
                "-i", video_path,
                "-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "18",           # high quality
                "-c:a", "aac",
                "-b:a", "192k",
                "-avoid_negative_ts", "make_zero",
                "-movflags", "+faststart",
                temp_path,
            ]

            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=180,
            )

            if proc.returncode != 0:
                stderr = proc.stderr.decode(errors="ignore")[:200]
                print(f"    ✗ Segment {j + 1} cut failed: {stderr}")
                return False

        # Step 2: Write concat list
        with open(concat_list_path, "w", encoding="utf-8") as f:
            for temp_path in temp_files:
                # FFmpeg concat requires forward slashes or escaped backslashes
                safe_path = temp_path.replace("\\", "/")
                f.write(f"file '{safe_path}'\n")

        # Step 3: Concatenate
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list_path,
            "-c", "copy",             # no re-encode since segments are uniform
            "-movflags", "+faststart",
            output_path,
        ]

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=180,
        )

        if proc.returncode != 0:
            stderr = proc.stderr.decode(errors="ignore")[:200]
            print(f"    ✗ Concat failed: {stderr}")
            return False

        return True

    finally:
        # Step 4: Clean up temp files
        for temp_path in temp_files:
            try:
                os.remove(temp_path)
            except OSError:
                pass
        try:
            os.remove(concat_list_path)
        except OSError:
            pass
