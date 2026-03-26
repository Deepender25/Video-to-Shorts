import os
from dotenv import load_dotenv

load_dotenv()

# ── API ──────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# ── Paths ────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOADS_DIR = os.path.join(BASE_DIR, "downloads")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(DOWNLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ── Clip constraints ────────────────────────────────
MIN_CLIP_DURATION = 15   # seconds (per segment)
MAX_CLIP_DURATION = 150  # seconds (total short duration, up to 2.5 min)

# ── OpenRouter settings ─────────────────────────────
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "minimax/minimax-m2.5:free")
CHUNK_MINUTES = 7        # minutes per transcript chunk for long videos
MAX_RETRIES = 3          # retries for API calls
