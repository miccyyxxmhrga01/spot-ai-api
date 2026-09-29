from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VALIDATION_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODEL_PATH = MODELS_DIR / "spot_talent_classifier.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"
TRAINING_HISTORY_PATH = REPORTS_DIR / "training_history.json"
EVALUATION_JSON_PATH = REPORTS_DIR / "evaluation.json"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"

# V1 deliberately contains only visual classes that can be reasonably inferred
# from a still image. Photography means a person using a camera, not a scenic
# photo, because every upload is already a photo. Communication is left out
# because a speaker holding a mic looks the same as a singer in a still image.
CLASS_NAMES = (
    "dancing",
    "singing",
    "drawing",
    "drums",
    "guitar",
    "other",
    "painting",
    "keyboard",
    "cooking",
    "basketball",
    "futsal",
    "volleyball",
    "photography",
    "coding",
)

SPOT_CATEGORY_MAP = {
    "singing":"Music",
    "guitar": "Music",
    "keyboard": "Music",
    "drums": "Music",
    "painting": "Art",
    "drawing": "Art",
    "dancing": "Dance",
    "cooking": "Culinary",
    "basketball": "Sports",
    "futsal": "Sports",
    "volleyball": "Sports",
    "photography": "Media & Content Creation",
    "coding": "Technology",
    "other": "Other",
}

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
INITIAL_EPOCHS = 12
FINE_TUNE_EPOCHS = 8
INITIAL_LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5
FINE_TUNE_LAST_N_LAYERS = 30

# A prediction below this threshold should be treated as uncertain and confirmed
# manually by the user in SPOT.
CONFIDENCE_THRESHOLD = 0.70
TOP_K = 3

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_RECOMMENDED_IMAGES_PER_CLASS = 100
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
