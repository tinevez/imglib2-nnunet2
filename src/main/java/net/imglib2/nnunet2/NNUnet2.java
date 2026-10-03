package net.imglib2.nnunet2;

import java.io.IOException;
import java.util.Map;

import org.apposed.appose.BuildException;
import org.apposed.appose.TaskException;

import net.imglib2.appose.util.ApposeTaskListener;

public class NNUnet2
{

	/**
	 * Runs nnUNet inference on a folder.
	 * 
	 * @param inputPath
	 *            the path to the input folder of TIFF images.
	 * @param outputPath
	 *            the path to the folder the predictions are written to.
	 * @param modelPath
	 *            the path to the nnUNet model, either a model folder or a ZIP
	 *            archive of one. ZIP archives are extracted and cached next to
	 *            the archive.
	 * @param folds
	 *            what folds to use for ensembling, as a comma or space
	 *            separated string of fold indices, or "all" (also the default)
	 *            to use all folds found in the model.
	 * @param device
	 *            the device to run inference on, e.g. "cpu" or "cuda:0". If not
	 *            specified, the default device is used.
	 * @param listener
	 *            a listener to receive progress updates and log messages.
	 * @throws TaskException
	 *             if the task fails to run.
	 * @throws InterruptedException
	 *             if the task is interrupted.
	 * @throws BuildException
	 *             if the nnUnet Python environment fails to build.
	 * @throws IOException
	 *             if the Python scripts cannot be accessed.
	 */
	public static void nnUnet2Folder(
			final String inputPath,
			final String outputPath,
			final String modelPath,
			final String folds,
			final String device,
			final ApposeTaskListener listener ) throws TaskException, InterruptedException, IOException, BuildException
	{
		try (final NNUnet2FolderRunner runner = NNUnet2FolderRunner.create( listener ))
		{
			runner.init();
			runner.run( Map.of(
					NNUnet2FolderRunner.INPUT_FOLDER_PATH, inputPath,
					NNUnet2FolderRunner.OUTPUT_FOLDER_PATH, outputPath,
					NNUnet2FolderRunner.MODEL_PATH, modelPath,
					NNUnet2FolderRunner.FOLDS, folds,
					NNUnet2FolderRunner.DEVICE, device ) );
		}
	}
}
