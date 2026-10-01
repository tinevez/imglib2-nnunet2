
SUFFIX_PRED = "_outline"
task: Task = globals()["task"]


def get_id(file_name: str) -> str:
    """Return the case identifier used in nnU-Net filenames.

    Called by copy_raw_files() for each accepted input image.
    """
    return file_name.split("_")[0]


def input_candidates(input_value) -> list[Path]:
    """Return the input paths to inspect without recursing into directories.

    Called by copy_raw_files() before filtering and copying input images.
    """
    if isinstance(input_value, (list, tuple)):
        return [Path(value) for value in input_value]

    input_path = Path(input_value)
    if input_path.is_dir():
        return sorted(input_path.iterdir())
    return [input_path]


def copy_raw_files(input_value, copy_folder: Path) -> list[Path]:
    """Copy accepted TIFF inputs into nnU-Net's channel-0 input folder.

    Called by main() after the predictor is initialized and before prediction.
    """
    task.update(message=f"nnU-Net: preparing input files in {copy_folder}")
    copy_folder.mkdir(parents=True, exist_ok=True)
    image_paths = []

    for source_path in input_candidates(input_value):
        if source_path.suffix.lower() not in (".tif", ".tiff"):
            continue
        if source_path.name.endswith(("contours.tif", "countours.tif")):
            continue
        if not source_path.is_file():
            continue

        case_id = get_id(source_path.name)
        destination = copy_folder / f"{case_id}_0000.tif"
        shutil.copy2(source_path, destination)
        image_paths.append(destination)
        task.update(message=f"nnU-Net: input image {source_path.resolve()}")

    if not image_paths:
        raise RuntimeError(f"No usable TIFF images found in {input_value}.")

    task.update(message=f"nnU-Net: prepared {len(image_paths)} input image(s)")
    return image_paths


def rename_outputs(output_folder: Path) -> None:
    """Append the project-specific outline suffix to prediction files.

    Called by main() after all input images have been predicted.
    """
    task.update(message="nnU-Net: renaming predicted files")
    for file_path in output_folder.glob("*.tif"):
        if not file_path.stem.endswith(SUFFIX_PRED):
            file_path.rename(output_folder / f"{file_path.stem}{SUFFIX_PRED}.tif")


def find_model_folder(root: Path):
    """Find a valid nnU-Net model directory below a given root.

    Called by resolve_model_folder() to validate cached or newly extracted models.
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


def resolve_model_folder(model_path: Path) -> Path:
    """Resolve a model directory, extracting and caching ZIP models if needed.

    Called by main() before predictor initialization. Existing valid cache
    content is reused when the archive signature has not changed.
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


def parse_folds(value):
    """Normalize Appose fold input for nnU-Net predictor initialization.

    Called by main() while reading task inputs; omitted or ``all`` selects all
    numbered folds detected by nnU-Net.
    """
    if value is None or value == "all" or value == ["all"]:
        return None
    if isinstance(value, str):
        # Blank input (e.g. an empty string) selects all folds as well.
        value = value.split()
        if not value:
            return None
    if isinstance(value, int):
        value = [value]
    return tuple(int(fold) for fold in value)


def main() -> None:
    """Run the complete Appose nnU-Net inference task.

    Called immediately at the end of this script by the Appose Python worker.
    It reads injected task inputs, initializes the model, predicts each image,
    renames outputs, and reports progress through task.update().
    """
    input_value = globals()["input"]
    output_folder = Path(globals()["output"])
    model_path = Path(globals()["model"])
    folds = parse_folds(globals().get("folds"))
    device_name = globals().get("device") or "mps"

    device = torch.device(device_name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; use device='cpu' instead")
    torch.set_num_threads(os.cpu_count() if device.type == "cpu" else 1)

    output_folder.mkdir(parents=True, exist_ok=True)
    model_folder = resolve_model_folder(model_path)

    task.update(message=f"nnU-Net: loading model on {device}")
    start_time = time.time()
    predictor = nnUNetPredictor(device=device, verbose=True)
    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=str(model_folder), use_folds=folds
    )
    task.update(message=f"nnU-Net: model loaded in {time.time() - start_time:.2f} s")

    if isinstance(input_value, (list, tuple)):
        copy_folder = output_folder / "raw_inference"
    else:
        copy_folder = Path(input_value).parent / "raw_inference"
    image_paths = copy_raw_files(input_value, copy_folder)

    task.update(message=f"nnU-Net: predicting {len(image_paths)} image(s)")
    for index, image_path in enumerate(image_paths, start=1):
        task.update(
            message=f"nnU-Net: predicting image {index}/{len(image_paths)}: "
            f"{image_path.name}"
        )
        predictor.predict_from_files_sequential([[str(image_path)]], str(output_folder))
        task.update(
            message=f"nnU-Net: completed image {index}/{len(image_paths)}: "
            f"{image_path.name}"
        )

    rename_outputs(output_folder)
    task.update(message=f"nnU-Net: processing completed; results in {output_folder}")


main()
