# This is the run script for the file-based variant of the nnUNet Appose
# task. The input is a folder (or a list) of TIFF images; predictions are
# written to an output folder. The model is loaded by nnunet2_utils.py
# helpers.


def input_candidates(input_value) -> list:
    """Return the input paths to inspect without recursing into directories.

    Called by copy_raw_files() before filtering and copying input images.
    """
    if isinstance(input_value, (list, tuple)):
        return [Path(value) for value in input_value]

    input_path = Path(input_value)
    if input_path.is_dir():
        return sorted(input_path.iterdir())
    return [input_path]


def copy_raw_files(input_value, copy_folder: Path) -> list:
    """Copy accepted TIFF inputs into nnU-Net's channel-0 input folder.

    Called by predict_folder() before prediction.
    """
    task.update(message=f"nnU-Net: preparing input files in {copy_folder}")
    copy_folder.mkdir(parents=True, exist_ok=True)
    image_paths = []
    destinations = {}

    for source_path in input_candidates(input_value):
        if source_path.suffix.lower() not in (".tif", ".tiff"):
            continue
        if not source_path.is_file():
            continue

        destination = copy_folder / f"{source_path.stem}_0000.tif"
        previous_source = destinations.get(destination)
        if previous_source is not None:
            raise ValueError(
                f"Input files {previous_source} and {source_path} map to the same "
                f"nnU-Net case: {destination.name}"
            )
        destinations[destination] = source_path
        image_paths.append((source_path, destination))

    if not image_paths:
        raise RuntimeError(f"No usable TIFF images found in {input_value}.")

    for source_path, destination in image_paths:
        shutil.copy2(source_path, destination)
        task.update(message=f"nnU-Net: input image {source_path.resolve()}")

    task.update(message=f"nnU-Net: prepared {len(image_paths)} input image(s)")
    return [destination for _, destination in image_paths]


def predict_folder(predictor, input_value, output_folder: Path) -> None:
    """Run nnU-Net on a folder or list of TIFF files, writing results to disk.

    Called by main() when the input is a file path or a list of file paths.
    """
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


def main() -> None:
    """Run the Appose nnU-Net inference task on files.

    Called immediately at the end of this script by the Appose Python worker.
    It reads injected task inputs, initializes the model, predicts each
    image, and reports progress through task.update().
    """
    input_value = globals()["input"]
    output_folder = Path(globals()["output"])
    model_path = Path(globals()["model"])
    folds = parse_folds(globals().get("folds"))
    device_name = globals().get("device") or "mps"

    resolve_device(device_name)
    output_folder.mkdir(parents=True, exist_ok=True)
    model_folder = resolve_model_folder(task, model_path)
    predictor = get_or_load_predictor(task, model_folder, folds, device_name)

    predict_folder(predictor, input_value, output_folder)
    task.update(message=f"nnU-Net: processing completed; results in {output_folder}")


main()
