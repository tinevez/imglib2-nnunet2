package net.imglib2.nnunet2;

import java.util.Map;
import java.util.Objects;

import org.apposed.appose.TaskException;

import net.imglib2.appose.runner.MambaApposeTaskRunner;
import net.imglib2.appose.util.ApposeTaskListener;

/**
 * Runs nnUNet v2 inference through Appose on files.
 * <p>
 * The input is a path to a folder of TIFF images (or a ZIP archive of a model),
 * and the predictions are written to an output folder as TIFF files. Image data
 * goes through the file system; for in-memory images, use {@link NNUnet2Runner}
 * instead, which passes them through shared memory.
 */
public class NNUnet2FolderRunner extends MambaApposeTaskRunner
{

	public static final String INPUT_FOLDER_PATH = "input";

	public static final String OUTPUT_FOLDER_PATH = "output";

	public static final String MODEL_PATH = "model";

	public static final String FOLDS = "folds";

	public static final String DEVICE = "device";

	private NNUnet2FolderRunner( final ApposeTaskListener listener )
	{
		super(
				NNUnet2FolderRunner.class.getResource( "environment.yaml" ),
				NNUnet2FolderRunner.class.getResource( "nnunet2_utils.py" ),
				NNUnet2FolderRunner.class.getResource( "nnunet2_folder.py" ),
				listener );
	}

	public static NNUnet2FolderRunner create( final ApposeTaskListener listener )
	{
		return new NNUnet2FolderRunner( listener );
	}

	/**
	 * Runs nnUNet inference on a folder or list of TIFF files.
	 *
	 * @param inputPath
	 *            path to the input folder of TIFF images.
	 * @param outputPath
	 *            path to the folder the predictions are written to.
	 * @param modelPath
	 *            path to the nnUNet model, either a model folder or a ZIP
	 *            archive of one. ZIP archives are extracted and cached next to
	 *            the archive.
	 * @param folds
	 *            the folds to use for ensembling, as a comma or space separated
	 *            string of fold indices, or "all" (also the default) to use all
	 *            folds found in the model.
	 * @param device
	 *            the torch device to run inference on, e.g. "cpu" or "mps".
	 * @throws InterruptedException
	 *             if the Python task is interrupted.
	 * @throws TaskException
	 *             if the Python task fails.
	 */
	public void run(
			final String inputPath,
			final String outputPath,
			final String modelPath,
			final String folds,
			final String device ) throws InterruptedException, TaskException
	{
		Objects.requireNonNull( inputPath, "inputPath" );
		Objects.requireNonNull( outputPath, "outputPath" );
		Objects.requireNonNull( modelPath, "modelPath" );

		final Map< String, Object > parameters = Map.of(
				INPUT_FOLDER_PATH, inputPath,
				OUTPUT_FOLDER_PATH, outputPath,
				MODEL_PATH, modelPath,
				FOLDS, ( folds == null || folds.isBlank() ) ? "all" : folds,
				DEVICE, ( device == null || device.isBlank() ) ? "mps" : device );
		run( parameters );
	}
}
