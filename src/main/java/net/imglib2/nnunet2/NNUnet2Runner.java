package net.imglib2.nnunet2;

import java.util.Map;
import java.util.Objects;

import org.apposed.appose.TaskException;

import net.imglib2.Dimensions;
import net.imglib2.FinalDimensions;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.appose.runner.AbstractShmApposeRunner;
import net.imglib2.appose.runner.MambaApposeTaskRunner;
import net.imglib2.appose.util.ApposeTaskListener;
import net.imglib2.appose.util.AxisInfo;
import net.imglib2.img.Img;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.util.Intervals;
import net.imglib2.util.Util;

/**
 * Runs nnUNet v2 inference through Appose using managed shared-memory image
 * buffers.
 * <p>
 * The input image is copied into a shared-memory buffer, and the prediction is
 * returned in another shared-memory buffer of {@link UnsignedByteType}, with
 * the same X and Y dimensions as the input. The Python side converts the image
 * to the layout expected by the nnUNet reader that was used at training time
 * (typically {@code NaturalImage2DIO}), which for an XYC image is a
 * channels-first {@code (C, Y, X)} array — exactly what Appose yields when
 * serializing an ImgLib2 {@code (X, Y, C)} buffer.
 * <p>
 * This class does not manage {@code ShmImg} instances directly. Image buffers
 * are owned by the composed shared-memory image store in
 * {@link AbstractShmApposeRunner}.
 */
public class NNUnet2Runner extends AbstractShmApposeRunner
{

	private static final String INPUT = "input";

	private static final String LABELS = "output_labels";

	private Dimensions labelDimensions;

	private NNUnet2Runner( final ApposeTaskListener listener )
	{
		super(
				new MambaApposeTaskRunner(
						NNUnet2Runner.class.getResource( "environment.yaml" ),
						NNUnet2Runner.class.getResource( "nnunet2_utils.py" ),
						NNUnet2Runner.class.getResource( "nnunet2.py" ),
						listener ) );
	}

	public static NNUnet2Runner create( final ApposeTaskListener listener )
	{
		return new NNUnet2Runner( listener );
	}

	/**
	 * Sets the input image to process.
	 * <p>
	 * Only 2D images (XY or XYC) are supported in this version, as this is what
	 * nnUNet 2D configurations process. The X and Y axes must be at positions 0
	 * and 1 respectively. For XYC images, the channel axis must be at position
	 * 2.
	 *
	 * @param input
	 *            the image to segment.
	 * @param axisInfo
	 *            the axis specification of the input image.
	 */
	public < T extends RealType< T > & NativeType< T > > void setInput(
			final RandomAccessibleInterval< T > input,
			final AxisInfo axisInfo )
	{
		Objects.requireNonNull( input, "input" );
		Objects.requireNonNull( axisInfo, "axisInfo" );

		if ( axisInfo.X() != 0 || axisInfo.Y() != 1 )
			throw new IllegalArgumentException( "X and Y axes must be at positions 0 and 1 respectively." );
		if ( axisInfo.Z() >= 0 || axisInfo.T() >= 0 )
			throw new IllegalArgumentException( "Only XY and XYC images are supported in this version. Found axes: " + axisInfo + "." );
		if ( axisInfo.C() >= 0 && axisInfo.C() != 2 )
			throw new IllegalArgumentException( "The channel axis must be at position 2 for XYC images." );

		writeImage( INPUT, input );
		this.labelDimensions = outputLabelDimensions( input, axisInfo );
		allocateImage( LABELS, new UnsignedByteType(), labelDimensions );
	}

	/**
	 * Runs nnUNet inference on the input image set by
	 * {@link #setInput(RandomAccessibleInterval, AxisInfo)}.
	 *
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
	public void run( final String modelPath, final String folds, final String device )
			throws InterruptedException, TaskException
	{
		Objects.requireNonNull( modelPath, "modelPath" );
		if ( !hasImage( INPUT ) )
			throw new IllegalStateException( "The input image has not been set. Please execute setInput() first." );

		final Map< String, Object > parameters = Map.of(
				"model", modelPath,
				"folds", ( folds == null || folds.isBlank() ) ? "all" : folds,
				"device", ( device == null || device.isBlank() ) ? "mps" : device );
		runTask( apposeMap( parameters ) );
	}

	/**
	 * Copies the prediction into the specified output image.
	 *
	 * @param outputLabels
	 *            the image to copy the prediction into. It must have the same
	 *            X and Y dimensions as the input image (without the channel
	 *            axis, if any), and be of {@link UnsignedByteType}.
	 */
	public void getOutputLabels( final RandomAccessibleInterval< UnsignedByteType > outputLabels )
	{
		Objects.requireNonNull( outputLabels, "outputLabels" );
		requireProcessed();
		if ( !Intervals.equalDimensions( labelDimensions, outputLabels ) )
			throw new IllegalArgumentException(
					"Output labels have dimensions " + Intervals.toString( outputLabels )
							+ " but expected " + Intervals.toString( labelDimensions ) + "." );
		readImage( LABELS, outputLabels );
	}

	/**
	 * Returns the prediction as a new {@link Img} of {@link UnsignedByteType},
	 * with the same X and Y dimensions as the input image (without the channel
	 * axis, if any).
	 *
	 * @return a new image containing the prediction.
	 */
	public Img< UnsignedByteType > getOutputLabels()
	{
		requireProcessed();
		final Img< UnsignedByteType > outputLabels = Util
				.getArrayOrCellImgFactory( labelDimensions, new UnsignedByteType() )
				.create( labelDimensions );
		readImage( LABELS, outputLabels );
		return outputLabels;
	}

	private static Dimensions outputLabelDimensions( final Dimensions input, final AxisInfo axisInfo )
	{
		final long[] dims = input.dimensionsAsLongArray();

		if ( axisInfo.C() < 0 )
			return new FinalDimensions( dims );

		final long[] result = new long[ dims.length - 1 ];
		int j = 0;
		for ( int d = 0; d < dims.length; d++ )
			if ( d != axisInfo.C() )
				result[ j++ ] = dims[ d ];

		return new FinalDimensions( result );
	}
}
