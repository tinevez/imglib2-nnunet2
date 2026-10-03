# These functions are shared by the nnU-Net Appose run scripts.
# This file is executed as the Appose service init script; the functions and
# imports defined here are exported to the worker and available in all run
# scripts. If you don't put the imports here, notably the numpy ones, the
# Appose worker will fail on Windows platforms.

import os
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from appose.python_worker import message
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# The properties dict that NaturalImage2DIO returns for 2D images. nnU-Net
# uses the 999 spacing value to detect 2D natural images.
NATURAL_IMAGE_2D_PROPERTIES = {"spacing": (999, 1, 1)}


def resolve_device(device_name: str) -> torch.device:
    """Resolve a torch device by name, checking availability.

    Called by the run scripts before inference. On CPU, it also sets the
    number of torch threads to the CPU count for faster preprocessing.
    """
    device = torch.device(device_name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; use device='cpu' instead")
    torch.set_num_threads((os.cpu_count() or 1) if device.type == "cpu" else 1)
    return device


def parse_folds(value):
    """Normalize Appose fold input for nnU-Net predictor initialization.

    Called by the run scripts while reading task inputs; omitted, blank or
    ``all`` selects all numbered folds detected by nnU-Net.
    """
    if value is None or value == "all" or value == ["all"]:
        return None
    if isinstance(value, str):
        # Blank input (e.g. an empty string) selects all folds as well.
        # Accept comma or space separated fold indices.
        value = value.replace(",", " ").split()
        if not value:
            return None
    if isinstance(value, int):
        value = [value]
    return tuple(int(fold) for fold in value)


def find_model_folder(root: Path):
    """Find a valid nnU-Net model directory below a given root.

    Called by resolve_model_folder() to validate cached or newly extracted
    models.
    """
    for folder, _, files in os.walk(root):
        folder_path = Path(folder)
        if "dataset.json" not in files or "plans.json" not in files:
            continue
        if any(
            (fold / "checkpoint_final.pth").is_file()
            for fold in folder_path.glob("fold_*")
        ):
            return folder_path
    return None


def resolve_model_folder(task, model_path: Path) -> Path:
    """Resolve a model directory, extracting and caching ZIP models if needed.

    nnUNet models are folders that can be large. The API provides a method to
    compress them and ship them as a single ZIP file. A user can pass the ZIP,
    but the model still needs to be extracted before it can be used. This
    function extracts the ZIP to a cache folder and reuses it if the ZIP has not
    changed. Called by the run scripts before predictor initialization. Existing
    valid cache content is reused when the archive signature has not changed.
    """
    if model_path.is_dir():
        return model_path
    if model_path.suffix.lower() != ".zip":
        raise FileNotFoundError(f"Model folder or ZIP not found: {model_path}")

    cache_path = model_path.with_name(f"{model_path.stem}_extracted")
    marker_path = cache_path / ".archive_signature"
    archive_stat = model_path.stat()
    signature = f"{archive_stat.st_size}:{archive_stat.st_mtime_ns}"

    if (
        marker_path.is_file()
        and marker_path.read_text(encoding="ascii").strip() == signature
    ):
        cached_model = find_model_folder(cache_path)
        if cached_model is not None:
            task.update(message=f"nnU-Net: using cached model {cached_model}")
            return cached_model

    if cache_path.exists():
        shutil.rmtree(cache_path)

    task.update(message=f"nnU-Net: extracting model archive {model_path}")
    temporary_path = Path(
        tempfile.mkdtemp(prefix=f".{cache_path.name}.", dir=model_path.parent)
    )
    try:
        with zipfile.ZipFile(model_path) as archive:
            extraction_root = temporary_path.resolve()
            for member in archive.infolist():
                destination = (temporary_path / member.filename).resolve()
                if os.path.commonpath((extraction_root, destination)) != str(
                    extraction_root
                ):
                    raise RuntimeError(
                        f"Unsafe path in model archive: {member.filename}"
                    )
            archive.extractall(temporary_path)

        extracted_model = find_model_folder(temporary_path)
        if extracted_model is None:
            raise RuntimeError(
                f"ZIP does not contain a valid nnU-Net model: {model_path}"
            )
        (temporary_path / ".archive_signature").write_text(
            signature + "\n", encoding="ascii"
        )
        temporary_path.replace(cache_path)
    except Exception:
        shutil.rmtree(temporary_path, ignore_errors=True)
        raise

    resolved_model = find_model_folder(cache_path)
    if resolved_model is None:
        raise RuntimeError(f"Extracted model is invalid: {cache_path}")
    task.update(message=f"nnU-Net: extracted model to {resolved_model}")
    return resolved_model


def get_or_load_predictor(
    task, model_folder: Path, folds, device_name: str
) -> nnUNetPredictor:
    """Return a predictor for the given model, loading it only if needed.

    Called by the run scripts. The predictor is cached in the Appose worker
    exports, since deploying the model on the device is expensive. It is
    reused when the requested model folder, folds and device match the cached
    one.

    NB: the cache lives in the worker exports, not in this init namespace, so
    we reach it through the worker instance registered at startup.
    """
    worker = getattr(message, "_worker_instance", None)
    exports = worker.exports if worker is not None else {}
    cached = exports.get("predictor", None)

    if (
        cached is not None
        and exports.get("predictor_model", None) == str(model_folder)
        and exports.get("predictor_folds", None) == folds
        and exports.get("predictor_device", None) == device_name
    ):
        task.update(message=f"nnU-Net: reusing cached predictor for {model_folder}")
        return cached

    task.update(message=f"nnU-Net: loading model on {device_name}")
    start_time = time.time()
    predictor = nnUNetPredictor(device=torch.device(device_name), verbose=True)
    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=str(model_folder), use_folds=folds
    )
    duration = time.time() - start_time
    task.update(message=f"nnU-Net: model loaded in {duration:.2f} s")

    # Cache for subsequent tasks in this worker process.
    task.export(
        predictor=predictor,
        predictor_model=str(model_folder),
        predictor_folds=folds,
        predictor_device=device_name,
    )
    return predictor
