from pathlib import Path
import os
import torch

MAIN_PATH = Path(__file__).resolve().parents[1]
DATA_PATH = MAIN_PATH / "data" / "captcha_images_v2"
MODEL_PATH = MAIN_PATH / "artifacts" / "model_ckpt"

BATCH_SIZE = 32
IMG_WIDTH = 300
IMG_HEIGHT = 75
NUM_WORKERS = os.cpu_count()
EPOCHS = 10
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
