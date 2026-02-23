import os
import subprocess


def cut_clips(video_path: str, clips: list[dict], output_dir: str) -> list[dict]:
    """
    Cut video into clips using FFmpeg with stream copy (no re-encoding).

    Args:
        video_path: path to the source video
        clips: list of validated clips with start/end
        output_dir: directory to write output files

    Returns:
        clips list with added 'output_path' and 'filename' keys
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for i, clip in enumerate(clips, 1):
        filename = f"short_{i}.mp4"
        output_path = os.path.join(output_dir, filename)

        cmd = [
            "ffmpeg",
            "-y",                       # overwrite
            "-ss", str(clip["start"]),   # seek to start
            "-to", str(clip["end"]),     # end time
            "-i", video_path,            # input file
            "-c", "copy",                # stream copy (no re-encode)
            "-avoid_negative_ts", "make_zero",
            output_path,
        ]

        try:
            subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                timeout=120,
            )

            # Verify output file exists and has size
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                clip_result = clip.copy()
                clip_result["output_path"] = output_path
                clip_result["filename"] = filename
                results.append(clip_result)

        except subprocess.CalledProcessError as e:
            # Log but continue with other clips
            print(f"FFmpeg error on clip {i}: {e.stderr.decode()[:200]}")
            continue
        except subprocess.TimeoutExpired:
            print(f"FFmpeg timeout on clip {i}")
            continue

    return results
