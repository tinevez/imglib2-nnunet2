package net.imglib2.nnunet2;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import org.apposed.appose.BuildException;
import org.apposed.appose.TaskException;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.appose.util.ApposeTaskListener;
import net.imglib2.appose.util.AxisInfo;
import net.imglib2.img.Img;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.UnsignedByteType;

/**
 * Static utility methods for running nnUNet inference, for third-party
 * developers.
 * <p>
 * Each method deploys the Python environment (if needed), runs the inference,
 * and closes the runner before returning. To process several images or volumes
 * efficiently, prefer the variants that take several images at once: they keep
 * a single runner alive for the whole batch, so the model is loaded only once.
 */
public class NNUnet2
{

	/**
	 * Runs nnUNet inference on a folder of TIFF images, writing predictions to
	 * an output folder.
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

	/**
	 * Runs nnUNet inference on a single image and returns the predicted labels.
	 * <p>
	 * Only 2D images (XY or XYC) are supported in this version, as this is what
	 * nnUNet 2D configurations process. The X and Y axes must be at positions 0
	 * and 1 respectively. For XYC images, the channel axis must be at position
	 * 2.
	 *
	 * @param <T>
	 *            the pixel type of the input image.
	 * @param image
	 *            the image to segment.
	 * @param axisInfo
	 *            the axis specification of the input image.
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
	 * @return a label image of {@link UnsignedByteType}, with the same X and Y
	 *         dimensions as the input (without the channel axis, if any).
	 * @throws TaskException
	 *             if the task fails to run.
	 * @throws InterruptedException
	 *             if the task is interrupted.
	 * @throws BuildException
	 *             if the nnUnet Python environment fails to build.
	 * @throws IOException
	 *             if the Python scripts cannot be accessed.
	 */
	public static < T extends RealType< T > & NativeType< T > > Img< UnsignedByteType > nnUnet2Image(
			final RandomAccessibleInterval< T > image,
			final AxisInfo axisInfo,
			final String modelPath,
			final String folds,
			final String device,
			final ApposeTaskListener listener ) throws TaskException, InterruptedException, IOException, BuildException
	{
		Objects.requireNonNull( image, "image" );
		final List< Img< UnsignedByteType > > list = nnUnet2Images( List.of( image ), axisInfo, modelPath, folds, device, listener );
		return list.get( 0 );
	}

	/**
	 * Runs nnUNet inference on a list of images and returns the predicted
	 * labels, in the same order as the input images.
	 * <p>
	 * A single runner is kept alive for the whole batch, so the model is loaded
	 * only once. Images may differ in size, but they must all match the
	 * specified axis specification.
	 * <p>
	 * Only 2D images (XY or XYC) are supported in this version, as this is what
	 * nnUNet 2D configurations process. The X and Y axes must be at positions 0
	 * and 1 respectively. For XYC images, the channel axis must be at position
	 * 2.
	 *
	 * @param <T>
	 *            the pixel type of the input images.
	 * @param images
	 *            the images to segment.
	 * @param axisInfo
	 *            the axis specification of the input images. All images must
	 *            match it.
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
	 * @return a list of label images of {@link UnsignedByteType}, in the same
	 *         order as the input images, each with the same X and Y dimensions
	 *         as its input (without the channel axis, if any).
	 * @throws TaskException
	 *             if the task fails to run.
	 * @throws InterruptedException
	 *             if the task is interrupted.
	 * @throws BuildException
	 *             if the nnUnet Python environment fails to build.
	 * @throws IOException
	 *             if the Python scripts cannot be accessed.
	 */
	public static < T extends RealType< T > & NativeType< T > > List< Img< UnsignedByteType > > nnUnet2Images(
			final List< RandomAccessibleInterval< T > > images,
			final AxisInfo axisInfo,
			final String modelPath,
			final String folds,
			final String device,
			final ApposeTaskListener listener ) throws TaskException, InterruptedException, IOException, BuildException
	{
		Objects.requireNonNull( images, "images" );
		Objects.requireNonNull( axisInfo, "axisInfo" );
		Objects.requireNonNull( modelPath, "modelPath" );
		Objects.requireNonNull( listener, "listener" );

		final List< Img< UnsignedByteType > > outputs = new ArrayList<>( images.size() );
		if ( images.isEmpty() )
			return outputs;

		final int nImages = images.size();
		try (final NNUnet2Runner runner = NNUnet2Runner.create( listener ))
		{
			runner.init();
			for ( int i = 0; i < nImages; i++ )
			{
				listener.message( String.format( "nnU-Net: image %d/%d", i + 1, nImages ) );
				runner.setInput( images.get( i ), axisInfo );
				runner.run( modelPath, folds, device );
				outputs.add( runner.getOutputLabels() );
			}
		}
		return outputs;
	}

	private NNUnet2()
	{}
}
