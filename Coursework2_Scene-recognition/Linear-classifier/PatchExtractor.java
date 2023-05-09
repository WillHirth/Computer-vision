package run2;

import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.util.pair.IntFloatPair;

class PatchExtractor implements FeatureExtractor<SparseIntFV, FImage>
{
	HardAssigner<float[], float[], IntFloatPair> assigner;
	
	public PatchExtractor(HardAssigner<float[], float[], IntFloatPair> assigner)
	{
		this.assigner = assigner;
	}
	
	/**
	 * Extract features from each image
	 * @param image Image to extract feature from
	 * @return Feature vector
	 */
	public SparseIntFV extractFeature(FImage image)
	{
		//Create a bag of visual words using the hard assigner
		BagOfVisualWords<float[]> bovw = new BagOfVisualWords<float[]>(assigner);
		
		//Aggregate the patches into a feature vector
		return bovw.aggregate(LinearClassifier.getPatches(image));
	}
}