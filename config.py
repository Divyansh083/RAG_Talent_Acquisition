import os

# === CONFIGURATION ===
BASE_DIR = os.getcwd()
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRANSCRIPT_DIR = os.path.join(BASE_DIR, "transcripts")
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)

WHISPER_MODEL = "medium"
GROQ_MODEL = "llama3-70b-8192"

# === API Keys ===
os.environ["GROQ_API_KEY"] = "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"
