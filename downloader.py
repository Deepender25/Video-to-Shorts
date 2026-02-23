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


# Cookie file for age-restricted / authenticated videos
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COOKIES_FILE = os.path.join(_BASE_DIR, "cookies.txt")
if not os.path.isfile(COOKIES_FILE):
    COOKIES_FILE = None  # no cookie file available


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


def _get_video_info(url: str, use_cookies_file: bool = False) -> dict:
    """Get video metadata without downloading. Retries on transient failure."""
    cmd = [
        YTDLP_BIN,
        "--dump-json",
        "--no-download",
        "--no-playlist",
        "--no-warnings",
        url,
    ]
    if use_cookies_file and COOKIES_FILE:
        cmd.extend(["--cookies", COOKIES_FILE])

    for attempt in range(MAX_DOWNLOAD_RETRIES):
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=45,
                text=True,
            )
            if proc.returncode == 0:
                return json.loads(proc.stdout)

            stderr = proc.stderr.strip()

            # Distinguish error types
            if "Video unavailable" in stderr or "Private video" in stderr:
                raise ValueError("This video is unavailable or private. Please try a different URL.")
            if "is not a valid URL" in stderr or "Unsupported URL" in stderr:
                raise ValueError("Invalid URL. Please provide a valid YouTube video link.")
                
            # Auth errors
            if "Sign in to confirm" in stderr or "age" in stderr.lower():
                raise PermissionError("Needs auth")
                
            # Browser cookie lock error / other file issues
            if "Could not copy" in stderr or "locked" in stderr.lower() or "sqlite" in stderr.lower() or "Permission denied" in stderr:
                raise FileNotFoundError(f"Cookie access issue: {stderr[:100]}")

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


def _get_video_info_with_auth_fallback(url: str) -> tuple[dict, bool]:
    """Tries fetching info without cookies, then falls back to cookies.txt if age-restricted."""
    # 1. Try without cookies first (fastest)
    try:
        return _get_video_info(url, use_cookies_file=False), False
    except PermissionError:
        pass  # Needs auth
        
    # 2. Try with cookies.txt for age-restricted videos
    print("  Video requires age verification. Attempting to use cookies.txt...")
    
    if not COOKIES_FILE:
        raise ValueError(
            "This video requires age verification. You must provide a cookies file.\n"
            "1. Install the 'Get cookies.txt LOCALLY' extension in Chrome/Edge.\n"
            "2. Go to YouTube and log in.\n"
            "3. Click the extension to download 'cookies.txt'.\n"
            "4. Place 'cookies.txt' in your Video-to-Shorts project folder.\n"
            "Then try again!"
        )
        
    try:
        return _get_video_info(url, use_cookies_file=True), True
    except PermissionError:
        raise ValueError(
            "This video requires age verification, but your cookies.txt seems to be expired. "
            "Please delete your old cookies.txt, log into YouTube, export a fresh cookies.txt "
            "using the 'Get cookies.txt LOCALLY' extension, and put it in your project folder."
        )
    except FileNotFoundError as e:
        raise ValueError(f"Cookie file error: {e}")


def _download_video_cli(url: str, output_dir: str, video_id: str, use_cookies_file: bool) -> str | None:
    """Download video using yt-dlp CLI with retry logic."""
    output_template = os.path.join(output_dir, f"{video_id}.%(ext)s")

    cmd = [
        YTDLP_BIN,
        "--format", "bestvideo+bestaudio/best",
        "--merge-output-format", "mp4",
        "--no-playlist",
        "--no-warnings",
        "--socket-timeout", "30",
        "--retries", "5",
        "--fragment-retries", "5",
        "-o", output_template,
        url,
    ]
    if use_cookies_file and COOKIES_FILE:
        cmd.extend(["--cookies", COOKIES_FILE])

    for attempt in range(MAX_DOWNLOAD_RETRIES):
        try:
            print(f"  Downloading video (attempt {attempt + 1})...")
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=900,  # 15 min timeout for large videos
                text=True,
            )
            output = proc.stdout or ""
            if proc.returncode != 0:
                print(f"  ⟳ Download output: {output[-300:]}")

            # Check for mp4 file
            video_file = _find_file(output_dir, ".mp4")
            if video_file and os.path.getsize(video_file) > 0:
                return video_file

            # Check for any video file (webm, mkv, etc.)
            for ext in [".webm", ".mkv", ".avi"]:
                video_file = _find_file(output_dir, ext)
                if video_file and os.path.getsize(video_file) > 0:
                    return video_file

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


def _download_captions_cli(url: str, output_dir: str, video_id: str, use_cookies_file: bool) -> str | None:
    """
    Download captions using yt-dlp CLI.
    Tries both manually uploaded and auto-generated captions.
    Tries multiple languages in priority order.
    """
    # Strategy 1: Try manually uploaded subtitles first (higher quality)
    for lang in SUBTITLE_LANGS:
        srt = _try_caption_download(
            url, output_dir, video_id, lang, auto=False, use_cookies_file=use_cookies_file
        )
        if srt:
            print(f"  ✓ Got manual captions ({lang}): {os.path.basename(srt)}")
            return srt

    # Strategy 2: Try auto-generated captions
    for lang in SUBTITLE_LANGS:
        srt = _try_caption_download(
            url, output_dir, video_id, lang, auto=True, use_cookies_file=use_cookies_file
        )
        if srt:
            print(f"  ✓ Got auto captions ({lang}): {os.path.basename(srt)}")
            return srt

    return None


def _try_caption_download(
    url: str, output_dir: str, video_id: str, lang: str, auto: bool, use_cookies_file: bool
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
        "--sub-lang", lang,
        "--convert-subs", "srt",
        "-o", os.path.join(output_dir, f"{video_id}.%(ext)s"),
        url,
    ]
    if use_cookies_file and COOKIES_FILE:
        cmd.extend(["--cookies", COOKIES_FILE])

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
    Uses yt-dlp CLI with robust retries and fallbacks.

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
    info, used_cookies = _get_video_info_with_auth_fallback(url)
    video_id = info.get("id", "video")
    title = info.get("title", "Untitled")
    duration = info.get("duration", 0)
    print(f"  Title: {title}")
    print(f"  Duration: {duration}s")
    if used_cookies:
        print(f"  Authentication: using cookies.txt")
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
    video_path = _download_video_cli(url, output_dir, video_id, used_cookies)
    if video_path is None:
        raise FileNotFoundError(
            "Video file could not be downloaded. "
            "Please check the URL and try again."
        )
    print(f"  ✓ Video ready: {os.path.basename(video_path)}")

    # ── Download captions ──
    print("Downloading captions...")
    srt_path = _download_captions_cli(url, output_dir, video_id, used_cookies)
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
