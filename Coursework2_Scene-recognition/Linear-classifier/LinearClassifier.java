package run2;

import Helper.Classifier;
import Helper.Paths;
import de.bwaldvogel.liblinear.SolverType;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.local.LocalFeatureImpl;
import org.openimaj.feature.local.SpatialLocation;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.feature.local.list.MemoryLocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.processing.algorithm.MeanCenter;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.pair.IntFloatPair;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class LinearClassifier extends Classifier
{
	/** Height and width of the patch */
	private static final int patchSize = 8;
	/** Step in x and y direction */
	private static final int sampleRate = 4;

	private LiblinearAnnotator<FImage, String> ann;

	/** Arg options: "-v" invokes validation,
	 *               No option, labels the testing dataset.
	 */
	public static void main(String[] args)
	{
		try
		{
			LinearClassifier linearClassifier = new LinearClassifier();

//			if (args[0].equals("-v"))
//				linearClassifier.validate(Paths.trainingDataset);
//			else
			{
				linearClassifier.setOutputFile(Paths.run2Output);
				linearClassifier.train(Paths.trainingDataset);
				linearClassifier.test(Paths.testingDataset);
			}
		}
		catch (Exception e) {System.err.println(e);}
	}

	/**
	 * Train the LiblinearAnnotator on the training dataset
	 * @param trainingDataset Section of the dataset to train on
	 */
	@Override
	protected void processTrainingImages(GroupedDataset<String, ? extends ListDataset<FImage>, FImage> trainingDataset)
	{
		ann = new LiblinearAnnotator<FImage, String>(new PatchExtractor(trainQuantiser(GroupedUniformRandomisedSampler.sample(trainingDataset, 30))),
				LiblinearAnnotator.Mode.MULTICLASS,
				SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);

		ann.train(trainingDataset);
	}

	/**
	 * Test the LiblinearAnnotator on an image
	 * @param image Image to label
	 * @return The guess
	 */
	@Override
	protected String processTestingImage(FImage image)
	{
		return (String) ann.classify(image).getPredictedClasses().toArray()[0];
	}

	/**
	 * Preform validation on a dataset
	 * @param dataset Testing section of the dataset
	 */
	@Override
	protected void processValidation(GroupedDataset<String, ListDataset<FImage>, FImage> dataset)
	{
		ClassificationEvaluator<CMResult<String>, String, FImage> eval =
				new ClassificationEvaluator<CMResult<String>, String, FImage>(ann, dataset,
						new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));

		Map<FImage, ClassificationResult<String>> guesses = eval.evaluate();
		CMResult<String> result = eval.analyse(guesses);

		System.out.println(result.getDetailReport());
	}

	/**
	 * Trains the vector quantisation for the hard assigner on the patches.
	 * @param sample A sample of the training dataset
	 * @return A trained hard assigner
	 */
	private static HardAssigner<float[], float[], IntFloatPair> trainQuantiser(Dataset<FImage> sample)
	{
		List<LocalFeatureList<LocalFeatureImpl<SpatialLocation, FloatFV>>> allKeys = new ArrayList<>();

		//Get all of the patches from the image dataset
		for (FImage image : sample)
			allKeys.add(getPatches(image));

		//Preform K means clustering on the image patches
		FloatKMeans km = FloatKMeans.createKDTreeEnsemble(500);
		DataSource<float[]> dataSource = new LocalFeatureListDataSource<>(allKeys);
		FloatCentroidsResult result = km.cluster(dataSource);

		return result.defaultHardAssigner();
	}

	/**
	 * Get patches generates a list of feature vectors from an image using fixed size patches
	 * @param image Image to get the patches from
	 * @return A feature vector list
	 */
	public static LocalFeatureList<LocalFeatureImpl<SpatialLocation, FloatFV>> getPatches(FImage image)
	{
		LocalFeatureList<LocalFeatureImpl<SpatialLocation, FloatFV>> features = new MemoryLocalFeatureList<>();

		for (int y = 0; y < image.getHeight() - patchSize; y += sampleRate)
			for (int x = 0; x < image.getWidth() - patchSize; x += sampleRate)
			{
				//This is a feature vector for the patch, using a 2D array so that it can be easily be converted into an
				//FImage so that mean centering and normalisation can be applied.
				float[][] vector = new float[1][patchSize * patchSize];

				//Get all of the pixels from the patch section
				for (int i = 0; i < patchSize; i++)
					for (int j = 0; j < patchSize; j++)
						vector[0][i * patchSize + j] = image.getPixel(x + j, y + i);

				//Add a new feature using the spatial location and preforming mean centering and normalisation on the
				//vector then turn it into a Float feature vector
				features.add(new LocalFeatureImpl<>(new SpatialLocation(x, y),
						new FloatFV(new FImage(vector).process(new MeanCenter()).normalise().pixels[0])));
			}

		return features;
	}
}