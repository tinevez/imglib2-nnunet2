package net.imglib2.nnunet2;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.apposed.appose.BuildException;
import org.apposed.appose.TaskException;

import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.plugin.Scaler;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.appose.util.ApposeTaskListener;
import net.imglib2.appose.util.AxisInfo;
import net.imglib2.converter.Converters;
import net.imglib2.img.Img;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.numeric.ARGBType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.util.Intervals;

public class NNUNet2Demo
{

	public static void main( final String[] args ) throws BuildException, IOException, InterruptedException, TaskException
	{
		fileBasedDemo();
		imglib2ImageDemo();
	}

	@SuppressWarnings( "unused" )
	private static void fileBasedDemo()
	{
		System.out.println( "File-based demo." );

		// Delete output folder if it exists, to avoid appose errors.

		final String rootPath = "/Users/tinevez/Projects/pdarnat_10098/";
		final String modelPath = rootPath + "ipa/model/Dataset000_2d_resencL.zip";
		final String inputFolderPath = rootPath + "samples/pdarnat_img/raw";
		final String outputFolderPath = rootPath + "samples/pdarnat_img/raw/output";
		final String folds = "all";
		final String device = "mps";

		try
		{
			FileUtils.deleteDirectory( new File( outputFolderPath ) );
		}
		catch ( final IOException e )
		{
			e.printStackTrace();
		}

		try
		{
			NNUnet2.nnUnet2Folder(
					inputFolderPath,
					outputFolderPath,
					modelPath,
					folds,
					device,
					ApposeTaskListener.STD );
		}
		catch ( TaskException | InterruptedException | IOException | BuildException e )
		{
			e.printStackTrace();
		}
	}

	@SuppressWarnings( "unused" )
	private static void imglib2ImageDemo()
	{
		System.out.println( "Shared-memory demo." );

		final String rootPath = "/Users/tinevez/Projects/pdarnat_10098/";
		final String modelPath = rootPath + "ipa/model/Dataset000_2d_resencL.zip";
		final String folds = "all";
		final String device = "mps";
		final String imagePath = rootPath + "samples/pdarnat_img/raw/2026041015_down.tif";

		// Load the sample RGB image.
		final ImagePlus imp2 = IJ.openImage( imagePath );
		final ImagePlus imp = Scaler.resize( imp2, 
				imp2.getWidth() / 2,
				imp2.getHeight() / 2,
				imp2.getNSlices(), "bilinear" );

		System.out.println( "Input image: " + imp );

		// Wrap it as an ImgLib2 image. RGB images wrap as an (X, Y) image of
		// ARGBType; we need to expose the channels as a third dimension.
		final Img< ARGBType > imgARGB = ImageJFunctions.wrap( imp );
		final RandomAccessibleInterval< UnsignedByteType > imgXYC = Converters.argbChannels( imgARGB, 1, 2, 3 );

		// Input axes.
		final AxisInfo axes = AxisInfo.XYC;

		// Run inference through the static utility method.
		final List< Img< UnsignedByteType > > results;
		try
		{
			results = NNUnet2.nnUnet2Images( List.of( imgXYC ), axes, modelPath, folds, device, ApposeTaskListener.STD );
		}
		catch ( TaskException | InterruptedException | IOException | BuildException e )
		{
			e.printStackTrace();
			return;
		}

		final Img< UnsignedByteType > labels = results.get( 0 );
		// Sanity check: count foreground pixels (value != 0).
		final long[] hist = new long[ 256 ];
		labels.forEach( p -> hist[ p.get() ]++ );
		System.out.println( "Label histogram (non-zero bins): " );
		for ( int v = 0; v < 256; v++ )
			if ( hist[ v ] > 0 )
				System.out.println( String.format( "  value %3d: %d pixels", v, hist[ v ] ) );

		ImageJ.main( null );
		ImageJFunctions.show( labels ).setTitle( "nnU-Net prediction" );
		System.out.println( "Done. Labels image: " + Intervals.toString( labels ) );
	}

	@SuppressWarnings( "unused" )
	private static void imglib2ImageDemoRunner()
	{
		System.out.println( "Shared-memory demo (runner API)." );

		final String rootPath = "/Users/tinevez/Projects/pdarnat_10098/";
		final String modelPath = rootPath + "ipa/model/Dataset000_2d_resencL.zip";
		final String folds = "all";
		final String device = "mps";
		final String imagePath = rootPath + "samples/pdarnat_img/raw/2026041015_down.tif";

		// Load the sample RGB image.
		final ImagePlus imp2 = IJ.openImage( imagePath );
		final ImagePlus imp = Scaler.resize( imp2,
				imp2.getWidth() / 2,
				imp2.getHeight() / 2,
				imp2.getNSlices(), "bilinear" );

		System.out.println( "Input image: " + imp );

		// Wrap it as an ImgLib2 image. RGB images wrap as an (X, Y) image of
		// ARGBType; we need to expose the channels as a third dimension.
		final Img< ARGBType > imgARGB = ImageJFunctions.wrap( imp );
		final RandomAccessibleInterval< UnsignedByteType > imgXYC = Converters.argbChannels( imgARGB, 1, 2, 3 );

		// Input axes.
		final AxisInfo axes = AxisInfo.XYC;

		final NNUnet2Runner runner = NNUnet2Runner.create( ApposeTaskListener.STD );
		try (runner)
		{
			System.out.println( "Initializing..." );
			runner.init();
			System.out.println( "Running..." );

			runner.setInput( imgXYC, axes );
			runner.run( modelPath, folds, device );

			final Img< UnsignedByteType > labels = runner.getOutputLabels();
			// Sanity check: count foreground pixels (value != 0).
			final long[] hist = new long[ 256 ];
			labels.forEach( p -> hist[ p.get() ]++ );
			System.out.println( "Label histogram (non-zero bins): " );
			for ( int v = 0; v < 256; v++ )
				if ( hist[ v ] > 0 )
					System.out.println( String.format( "  value %3d: %d pixels", v, hist[ v ] ) );

			ImageJ.main( null );
			ImageJFunctions.show( labels ).setTitle( "nnU-Net prediction" );
			System.out.println( "Done. Labels image: " + Intervals.toString( labels ) );
		}
		catch ( IOException | BuildException | InterruptedException | TaskException e )
		{
			e.printStackTrace();
		}
	}

	private NNUNet2Demo()
	{}
}
