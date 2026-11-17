"""Central place for paths and settings. All paths are relative to the repo root,
so the app runs the same on your laptop, in Colab and on Streamlit Community Cloud."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
NLTK_DIR = ROOT / "nltk_data"
# The dataset pack keeps sample_images/ next to data/ (Project_01/sample_images).
SAMPLE_IMAGES = ROOT / "sample_images" if (ROOT / "sample_images").exists() else DATA / "sample_images"

for _d in (MODELS, REPORTS):
    _d.mkdir(exist_ok=True)

# Groq Free Plan model (proposal Section 8). Override with env var or Streamlit secret.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_MODEL = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)

SEED = 42
MAX_COMMENT_CHARS = 1000

# Ranges used if meta.json does not list them (UCI Student Performance dictionary)
FALLBACK_RANGES = {
    "G1": (0, 20), "G2": (0, 20), "G3": (0, 20),
    "studytime": (1, 4), "failures": (0, 4),
}
UNITS = {
    "G1": "points out of 20", "G2": "points out of 20",
    "studytime": "category 1-4 (1 = <2 h, 2 = 2-5 h, 3 = 5-10 h, 4 = >10 h per week)",
    "failures": "number of past class failures",
}
