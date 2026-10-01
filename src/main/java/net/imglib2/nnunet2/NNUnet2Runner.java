package net.imglib2.nnunet2;

import net.imglib2.appose.runner.MambaApposeTaskRunner;
import net.imglib2.appose.util.ApposeTaskListener;

public class NNUnet2Runner extends MambaApposeTaskRunner
{

	public static final String INPUT_FOLDER_PATH = "input";

	public static final String OUTPUT_FOLDER_PATH = "output";

	public static final String MODEL_PATH = "model";

	public static final String FOLDS = "folds";

	public static final String DEVICE = "device";

	private NNUnet2Runner( final ApposeTaskListener listener )
	{
		super(
				NNUnet2Runner.class.getResource( "environment.yaml" ),
				NNUnet2Runner.class.getResource( "nnunet2_utils.py" ),
				NNUnet2Runner.class.getResource( "nnunet2.py" ),
				listener );
	}

	public static NNUnet2Runner create( final ApposeTaskListener listener )
	{
		return new NNUnet2Runner( listener );
	}
}
