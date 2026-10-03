# This is the run script for the shared-memory (image loaded) variant of the
# nnU-Net Appose task. The input image and the output labels are passed as
# shared-memory buffers, and the prediction is written in place in the output
# buffer. The model is loaded by nnunet2_utils.py helpers.


def predict_shared_memory(predictor, image, output_labels) -> None:
    """Run nnUNet on an in-memory image and write labels to shared memory.

    Called by main() when the input is a shared-memory image.

    The input arrives as a numpy array with the Appose C-order shape. For an
    ImgLib2 XYC image (X, Y, C) it is (C, Y, X); for an XY image it is
    (Y, X). This matches the channel-first layout of NaturalImage2DIO, the
    reader nnUNet used at training time for this kind of data, except that
    the reader adds a singleton Z axis: nnUNet 2D configurations expect
    (C, 1, Y, X) images, so we insert it here.

    What if your input image is a time-lapse or a volume? Well we don't
    support that yet.

    The prediction is a (1, Y, X) label array. Its trailing (Y, X) shape
    matches exactly the Appose C-order shape of the shared output labels
    buffer, whose ImgLib2 shape is (X, Y), so we can copy it in directly.
    """
    if image.ndim == 3:
        # (C, Y, X) -> (C, 1, Y, X)
        img = np.ascontiguousarray(image[:, None, :, :], dtype=np.float32)
    elif image.ndim == 2:
        # (Y, X) -> (1, 1, Y, X)
        img = np.ascontiguousarray(image[None, None, :, :], dtype=np.float32)
    else:
        raise RuntimeError(f"Unexpected shared-memory image shape: {image.shape}")

    task.update(message=f"nnU-Net: predicting image of shape {img.shape}")

    seg = predictor.predict_single_npy_array(
        input_image=img, image_properties=NATURAL_IMAGE_2D_PROPERTIES
    )
    # seg is (1, Y, X) with labels 0 = background, 1 = foreground. Its
    # trailing shape matches the shared output labels buffer directly.
    output_labels[:] = seg[0]


def main() -> None:
    """Run the Appose nnU-Net inference task on a shared-memory image.

    Called immediately at the end of this script by the Appose Python worker.
    It reads injected task inputs, initializes the model, predicts, and
    reports progress through task.update().
    """
    image = globals()["input"].ndarray()
    output_labels = globals()["output_labels"].ndarray()
    model_path = Path(globals()["model"])
    folds = parse_folds(globals().get("folds"))
    device_name = globals().get("device") or "mps"

    resolve_device(device_name)
    model_folder = resolve_model_folder(task, model_path)
    predictor = get_or_load_predictor(task, model_folder, folds, device_name)

    start_time = time.time()
    predict_shared_memory(predictor, image, output_labels)
    duration = time.time() - start_time
    task.update(message=f"nnU-Net: prediction completed in {duration:.2f} s")


main()
