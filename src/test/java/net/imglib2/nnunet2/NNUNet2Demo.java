package net.imglib2.nnunet2;

import java.io.IOException;
import java.util.Map;

import org.apposed.appose.BuildException;
import org.apposed.appose.TaskException;

import net.imglib2.appose.util.ApposeTaskListener;

public class NNUNet2Demo
{

	public static void main( final String[] args )
	{
		final String rootPath = "/Users/tinevez/Projects/pdarnat_10098/";
		final String modelPath = rootPath + "ipa/model/Dataset000_2d_resencL.zip";
		final String inputFolderPath = rootPath + "samples/pdarnat_img/raw";
		final String outputFolderPath = rootPath + "samples/pdarnat_img/raw/output";
		final String folds = "all";

		final Map< String, Object > params = Map.of(
				NNUnet2Runner.INPUT_FOLDER_PATH, inputFolderPath,
				NNUnet2Runner.OUTPUT_FOLDER_PATH, outputFolderPath,
				NNUnet2Runner.MODEL_PATH, modelPath,
				NNUnet2Runner.FOLDS, folds,
				NNUnet2Runner.DEVICE, "mps" );


		final NNUnet2Runner runner = NNUnet2Runner.create( ApposeTaskListener.STD );
		try (runner)
		{
			System.out.println( "Initializing..." );
			runner.init();
			System.out.println( "Running..." );
			runner.run( params );
			System.out.println( "Done." );
		}
		catch ( final IOException e )
		{
			e.printStackTrace();
		}
		catch ( final BuildException e )
		{
			e.printStackTrace();
		}
		catch ( final InterruptedException e )
		{
			e.printStackTrace();
		}
		catch ( final TaskException e )
		{
			e.printStackTrace();
		}
	}
}
