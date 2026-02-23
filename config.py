import os
from dotenv import load_dotenv

load_dotenv()

# ── API ──────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ── Paths ────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOADS_DIR = os.path.join(BASE_DIR, "downloads")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(DOWNLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ── Clip constraints ────────────────────────────────
MIN_CLIP_DURATION = 20   # seconds
MAX_CLIP_DURATION = 60   # seconds

# ── Gemini settings ─────────────────────────────────
GEMINI_MODEL = "gemini-2.0-flash"
CHUNK_MINUTES = 7        # minutes per transcript chunk for long videos
MAX_RETRIES = 3          # retries for API calls
