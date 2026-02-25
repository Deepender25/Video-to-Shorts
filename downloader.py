import os
import glob
import json
import time
import subprocess
import sys

# Use the venv's yt-dlp binary (not the outdated system-wide one)
YTDLP_BIN = os.path.join(os.path.dirname(sys.executable), "yt-dlp.exe")
if not os.path.exists(YTDLP_BIN):
    YTDLP_BIN = "yt-dlp"  # fallback to system PATH

# Languages to attempt for captions, in priority order
SUBTITLE_LANGS = ["en", "hi", "en-US", "en-GB", "en-IN"]

# Retry settings
MAX_DOWNLOAD_RETRIES = 3
RETRY_BACKOFF_BASE = 3  # seconds

# Format selection strategies — tried in order until one works.
# This prevents "Requested format is not available" errors.
FORMAT_STRATEGIES = [
    # 1. Best quality: separate video+audio merged to mp4
    "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio",
    # 2. Best pre-muxed file (already combined video+audio)
    "best[ext=mp4]/best",
    # 3. Absolute fallback — grab literally anything
    "worst",
]

# Cookie file for age-restricted / authenticated videos
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COOKIES_FILE = os.path.join(_BASE_DIR, "cookies.txt")
if not os.path.isfile(COOKIES_FILE):
    COOKIES_FILE = None  # no cookie file available

# Browser cookie sources to try (in priority order)
BROWSER_COOKIE_SOURCES = ["edge", "chrome", "firefox", "brave"]

# Extra extractor arguments to help with age-restricted videos
# Uses multiple player clients — some bypass restrictions that others don't.
YTDLP_EXTRACTOR_ARGS = ["--extractor-args", "youtube:player_client=web,default"]


def _find_file(directory: str, ext: str) -> str | None:
    """Find the first file with given extension in a directory (case-insensitive)."""
    ext_lower = ext.lower()
    for f in os.listdir(directory):
        if f.lower().endswith(ext_lower):
            return os.path.join(directory, f)
    return None


def _find_srt_file(directory: str) -> str | None:
    """
    Find any SRT file in the directory.
    yt-dlp can name subtitle files in many patterns, so we use glob.
    """
    patterns = ["*.srt"]
    for pattern in patterns:
        matches = glob.glob(os.path.join(directory, pattern))
        if matches:
            return matches[0]
    return None


def _get_video_info(url: str, auth_args: list[str] | None = None) -> dict:
    """
    Get video metadata without downloading. Retries on transient failure.
    auth_args: optional list of CLI args for authentication, e.g.
               ['--cookies', 'cookies.txt'] or ['--cookies-from-browser', 'edge'].
    """
    cmd = [
        YTDLP_BIN,
        "--dump-json",
        "--no-download",
        "--no-playlist",
        "--no-warnings",
        "--no-check-formats",   # Don't verify format availability during info fetch
        *YTDLP_EXTRACTOR_ARGS,  # Use multiple player clients for age-restricted
        url,
    ]
    if auth_args:
        cmd.extend(auth_args)

    for attempt in range(MAX_DOWNLOAD_RETRIES):
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
                text=True,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                return json.loads(proc.stdout)

            stderr = proc.stderr.strip()

            # Distinguish error types
            if "Video unavailable" in stderr or "Private video" in stderr:
                raise ValueError("This video is unavailable or private. Please try a different URL.")
            if "is not a valid URL" in stderr or "Unsupported URL" in stderr:
                raise ValueError("Invalid URL. Please provide a valid YouTube video link.")

            # Auth / age-restricted errors
            if "Sign in to confirm" in stderr or "age" in stderr.lower():
                raise PermissionError("Needs auth")

            # Format errors — often caused by age-restriction hiding formats;
            # treat as a potential auth issue so the fallback can retry with cookies.
            if "Requested format" in stderr or "format is not available" in stderr.lower():
                raise PermissionError("Needs auth (format issue)")

            # "Only images are available" — age-restricted video without proper auth
            if "Only images" in stderr or "no video formats" in stderr.lower():
                raise PermissionError("Needs auth (images only)")

            # Browser cookie lock error / other file issues
            if "Could not copy" in stderr or "locked" in stderr.lower() or "sqlite" in stderr.lower() or "Permission denied" in stderr:
                raise FileNotFoundError(f"Cookie access issue: {stderr[:120]}")

            # Transient error — retry
            if attempt < MAX_DOWNLOAD_RETRIES - 1:
                print(f"  ⟳ Info fetch failed (attempt {attempt + 1}), retrying...")
                time.sleep(RETRY_BACKOFF_BASE * (attempt + 1))
                continue

            raise RuntimeError(f"Failed to get video info: {stderr[:300]}")

        except subprocess.TimeoutExpired:
            if attempt < MAX_DOWNLOAD_RETRIES - 1:
                print(f"  ⟳ Info fetch timed out (attempt {attempt + 1}), retrying...")
                time.sleep(RETRY_BACKOFF_BASE * (attempt + 1))
                continue
            raise RuntimeError("Timed out getting video info. The video may be too long or YouTube is slow.")

    raise RuntimeError("Failed to get video info after all retries.")


def _get_basic_video_info(url: str, auth_args: list[str] | None = None) -> dict | None:
    """
    Fallback: extract minimal metadata using --print when --dump-json fails.
    Returns a partial info dict or None.
    """
    cmd = [
        YTDLP_BIN,
        "--no-download",
        "--no-playlist",
        "--no-warnings",
        "--no-check-formats",
        *YTDLP_EXTRACTOR_ARGS,
        "--print", "%(id)s|||%(title)s|||%(duration)s",
        url,
    ]
    if auth_args:
        cmd.extend(auth_args)

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            text=True,
        )
        stdout = proc.stdout.strip()
        if stdout and "|||" in stdout:
            parts = stdout.split("|||")
            if len(parts) >= 3:
                duration_str = parts[2].strip()
                try:
                    duration = int(float(duration_str))
                except (ValueError, TypeError):
                    duration = 0
                return {
                    "id": parts[0].strip() or "video",
                    "title": parts[1].strip() or "Untitled",
                    "duration": duration,
                }
    except Exception:
        pass
    return None


def _build_auth_strategies() -> list[tuple[str, list[str] | None]]:
    """
    Build a list of (description, auth_args) tuples to try in order.
    Includes: no auth, cookies.txt, and each installed browser.
    """
    strategies: list[tuple[str, list[str] | None]] = [
        ("no cookies", None),
    ]

    if COOKIES_FILE:
        strategies.append(("cookies.txt", ["--cookies", COOKIES_FILE]))

    for browser in BROWSER_COOKIE_SOURCES:
        strategies.append(
            (f"browser cookies ({browser})", ["--cookies-from-browser", browser])
        )

    return strategies


def _get_video_info_with_auth_fallback(url: str) -> tuple[dict, list[str] | None]:
    """
    Tries fetching info with multiple auth strategies in order:
      1. No cookies (fastest, works for non-restricted videos)
      2. cookies.txt (if present)
      3. --cookies-from-browser edge/chrome/firefox/brave

    Returns (info_dict, auth_args_that_worked).
    """
    strategies = _build_auth_strategies()
    last_error = None
    cookie_file_error = None

    for desc, auth_args in strategies:
        try:
            print(f"  Trying auth: {desc}...")
            info = _get_video_info(url, auth_args=auth_args)
            print(f"  ✓ Success with {desc}")
            return info, auth_args
        except PermissionError as e:
            print(f"  ✗ {desc}: needs auth — {e}")
            last_error = e
        except FileNotFoundError as e:
            # Browser is open / cookie file locked — skip this one, try next
            print(f"  ✗ {desc}: cookie access issue (browser may be open)")
            cookie_file_error = e
        except ValueError:
            raise  # Video unavailable, invalid URL — don't retry
        except RuntimeError as e:
            # Transient errors after retries; try next auth strategy
            print(f"  ✗ {desc}: {e}")
            last_error = e

    # All strategies failed. Try _get_basic_video_info as a last resort.
    print("  All auth strategies failed for --dump-json. Trying partial info extraction...")
    for desc, auth_args in strategies:
        basic = _get_basic_video_info(url, auth_args=auth_args)
        if basic:
            print(f"  ✓ Got basic info via {desc}")
            return basic, auth_args

    # Provide a helpful error message
    raise ValueError(
        "This video is age-restricted. YouTube has strict age verification that prevents "
        "automated downloads of some restricted videos, even with valid cookies.\n\n"
        "What you can try:\n"
        "  1. Make sure your cookies.txt is fresh (re-export it from YouTube while logged in)\n"
        "  2. Try a different video that has the same content but isn't age-restricted\n"
        "  3. Close all browser windows and try again (so yt-dlp can read browser cookies)\n\n"
        "Note: Non-age-restricted videos will work perfectly fine!"
    )


def _download_video_cli(url: str, output_dir: str, video_id: str, auth_args: list[str] | None = None) -> str | None:
    """
    Download video using yt-dlp CLI.
    Tries multiple format strategies to avoid 'Requested format is not available'.
    """
    output_template = os.path.join(output_dir, f"{video_id}.%(ext)s")

    for strategy_idx, fmt in enumerate(FORMAT_STRATEGIES):
        # Clean any partial downloads from a previous failed strategy
        for ext in [".mp4", ".webm", ".mkv", ".avi", ".part", ".ytdl"]:
            for old in glob.glob(os.path.join(output_dir, f"*{ext}")):
                try:
                    os.remove(old)
                except OSError:
                    pass

        cmd = [
            YTDLP_BIN,
            "--format", fmt,
            "--merge-output-format", "mp4",
            "--no-playlist",
            "--no-warnings",
            "--no-check-formats",  # Don't pre-check format availability
            *YTDLP_EXTRACTOR_ARGS,
            "--socket-timeout", "30",
            "--retries", "5",
            "--fragment-retries", "5",
            "-o", output_template,
            url,
        ]
        if auth_args:
            cmd.extend(auth_args)

        for attempt in range(MAX_DOWNLOAD_RETRIES):
            try:
                print(f"  Downloading video (strategy {strategy_idx + 1}/{len(FORMAT_STRATEGIES)}, "
                      f"attempt {attempt + 1}, format: {fmt})...")
                proc = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=900,  # 15 min timeout for large videos
                    text=True,
                )
                output = proc.stdout or ""

                # Check for a downloaded video file
                video_file = _find_file(output_dir, ".mp4")
                if video_file and os.path.getsize(video_file) > 0:
                    print(f"  ✓ Downloaded with format strategy {strategy_idx + 1}")
                    return video_file

                for ext in [".webm", ".mkv", ".avi"]:
                    video_file = _find_file(output_dir, ext)
                    if video_file and os.path.getsize(video_file) > 0:
                        print(f"  ✓ Downloaded ({ext}) with format strategy {strategy_idx + 1}")
                        return video_file

                # If the error is about format availability, skip retries
                # and jump to the next format strategy immediately
                if proc.returncode != 0:
                    if "Requested format" in output or "format is not available" in output.lower():
                        print(f"  ✗ Format strategy {strategy_idx + 1} not available, trying next...")
                        break  # break retry loop, go to next strategy
                    if "Only images" in output:
                        print(f"  ✗ Only images available (auth issue), trying next strategy...")
                        break
                    print(f"  ⟳ Download output: {output[-300:]}")

                if attempt < MAX_DOWNLOAD_RETRIES - 1:
                    print(f"  ⟳ No video file found, retrying...")
                    time.sleep(RETRY_BACKOFF_BASE * (attempt + 1))
                    continue

            except subprocess.TimeoutExpired:
                if attempt < MAX_DOWNLOAD_RETRIES - 1:
                    print(f"  ⟳ Download timed out (attempt {attempt + 1}), retrying...")
                    time.sleep(RETRY_BACKOFF_BASE * (attempt + 1))
                    continue
                raise RuntimeError(
                    "Video download timed out. The video may be too large."
                )

    return None


def _download_captions_cli(url: str, output_dir: str, video_id: str, auth_args: list[str] | None = None) -> str | None:
    """
    Download captions using yt-dlp CLI.
    Tries both manually uploaded and auto-generated captions.
    Tries multiple languages in priority order.
    """
    # Strategy 1: Try manually uploaded subtitles first (higher quality)
    for lang in SUBTITLE_LANGS:
        srt = _try_caption_download(
            url, output_dir, video_id, lang, auto=False, auth_args=auth_args
        )
        if srt:
            print(f"  ✓ Got manual captions ({lang}): {os.path.basename(srt)}")
            return srt

    # Strategy 2: Try auto-generated captions
    for lang in SUBTITLE_LANGS:
        srt = _try_caption_download(
            url, output_dir, video_id, lang, auto=True, auth_args=auth_args
        )
        if srt:
            print(f"  ✓ Got auto captions ({lang}): {os.path.basename(srt)}")
            return srt

    return None


def _try_caption_download(
    url: str, output_dir: str, video_id: str, lang: str, auto: bool,
    auth_args: list[str] | None = None
) -> str | None:
    """Attempt to download captions for a specific language."""
    kind = "auto" if auto else "manual"
    print(f"  Trying {kind} captions: {lang}...")

    # Clean any previous subtitle files to avoid confusion
    for f in glob.glob(os.path.join(output_dir, "*.srt")):
        try:
            os.remove(f)
        except OSError:
            pass
    for f in glob.glob(os.path.join(output_dir, "*.vtt")):
        try:
            os.remove(f)
        except OSError:
            pass

    cmd = [
        YTDLP_BIN,
        "--skip-download",
        "--no-playlist",
        "--no-warnings",
        "--no-check-formats",
        *YTDLP_EXTRACTOR_ARGS,
        "--sub-lang", lang,
        "--convert-subs", "srt",
        "-o", os.path.join(output_dir, f"{video_id}.%(ext)s"),
        url,
    ]
    if auth_args:
        cmd.extend(auth_args)

    if auto:
        cmd.insert(2, "--write-auto-subs")
    else:
        cmd.insert(2, "--write-subs")

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=60,
            text=True,
        )
        output = proc.stdout or ""

        # Check if an SRT was produced
        srt = _find_srt_file(output_dir)
        if srt and os.path.getsize(srt) > 50:  # at least 50 bytes = real content
            return srt

        # Rate limited — wait before next attempt
        if "429" in output or "Too Many Requests" in output:
            print(f"  ✗ Rate limited for '{lang}', waiting 5s...")
            time.sleep(5)
        else:
            time.sleep(0.5)

    except subprocess.TimeoutExpired:
        print(f"  ✗ Timeout for '{lang}' ({kind})")
    except Exception as e:
        print(f"  ✗ Error for '{lang}' ({kind}): {e}")

    return None


def download_video(url: str, output_dir: str) -> dict:
    """
    Download a YouTube video and its captions.
    Uses yt-dlp CLI with robust retries, format fallbacks, and multi-strategy auth.

    Returns:
        dict with keys: video_path, srt_path, title, duration, video_id

    Raises:
        ValueError: for user-fixable issues (bad URL, private video, no captions)
        RuntimeError: for infrastructure failures (network, timeout)
        FileNotFoundError: when download succeeds but files are missing
    """
    # Clean previous downloads
    if os.path.exists(output_dir):
        for f in glob.glob(os.path.join(output_dir, "*")):
            try:
                os.remove(f)
            except OSError:
                pass
    os.makedirs(output_dir, exist_ok=True)

    # ── Get video metadata ──
    print("Getting video info...")
    info, auth_args = _get_video_info_with_auth_fallback(url)
    video_id = info.get("id", "video")
    title = info.get("title", "Untitled")
    duration = info.get("duration", 0)
    print(f"  Title: {title}")
    print(f"  Duration: {duration}s")
    if auth_args:
        print(f"  Authentication: {auth_args}")
    else:
        print(f"  Authentication: none required")

    # Sanity check: reject extremely short or extremely long videos early
    if duration and duration < 15:
        raise ValueError(
            "Video is too short (under 15 seconds). "
            "Please provide a longer video."
        )
    if duration and duration > 14400:  # 4 hours
        raise ValueError(
            "Video is too long (over 4 hours). "
            "Please provide a shorter video."
        )

    # ── Download video ──
    video_path = _download_video_cli(url, output_dir, video_id, auth_args=auth_args)
    if video_path is None:
        raise FileNotFoundError(
            "Video file could not be downloaded. "
            "The video may be age-restricted or region-locked. "
            "Try closing all browser windows and retrying, or provide a fresh cookies.txt."
        )
    print(f"  ✓ Video ready: {os.path.basename(video_path)}")

    # ── Download captions ──
    print("Downloading captions...")
    srt_path = _download_captions_cli(url, output_dir, video_id, auth_args=auth_args)
    if srt_path is None:
        raise ValueError(
            "No captions available for this video (neither manual nor auto-generated). "
            "This tool requires videos with captions. Please try a different video."
        )

    # Detect subtitle language from filename (e.g. "video.hi.srt" → "hi")
    subtitle_lang = "en"  # default
    if srt_path:
        basename = os.path.basename(srt_path)
        # Pattern: videoId.LANG.srt  (e.g. dQw4w9WgXcQ.hi.srt)
        parts = basename.rsplit(".", 2)
        if len(parts) >= 3:
            detected = parts[-2].lower()
            if len(detected) <= 5:  # valid lang codes: "en", "hi", "en-US", etc.
                subtitle_lang = detected
    print(f"  Detected subtitle language: {subtitle_lang}")

    return {
        "video_path": video_path,
        "srt_path": srt_path,
        "title": title,
        "duration": duration,
        "video_id": video_id,
        "subtitle_lang": subtitle_lang,
    }
