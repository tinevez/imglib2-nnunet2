import os
import shutil
import tempfile
import time
import warnings
import zipfile
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message=r"`torch\.jit\.interface` is deprecated.*",
    category=FutureWarning,
    module=r"torch\.jit\._script",
)


import torch
from appose.python_worker import Task
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
