package run1;

import Helper.CountingHashMap;
import Helper.Classifier;
import Helper.Paths;
import org.openimaj.data.dataset.*;
import org.openimaj.image.FImage;
import org.openimaj.image.processing.algorithm.MeanCenter;
import org.openimaj.image.processing.resize.ResizeProcessor;

import java.util.*;

public class NearestNeighbourClassifier extends Classifier
{
	/** The number of k nearest neighbours */
	private int k;
	/** List of feature vectors */
	private ArrayList<Map.Entry<double[], String>> features = new ArrayList<Map.Entry<double[], String>>();
	
    public static void main(String[] args)
	{
        try
		{
			NearestNeighbourClassifier nearestNeighbourClassifier = new NearestNeighbourClassifier(35);
			
			nearestNeighbourClassifier.setOutputFile(Paths.run1Output);
			
			nearestNeighbourClassifier.validate(Paths.trainingDataset);
			nearestNeighbourClassifier.clearFeatures();
			
			nearestNeighbourClassifier.train(Paths.trainingDataset);
			nearestNeighbourClassifier.test(Paths.testingDataset);
		} catch (Exception e) {System.err.println(e);}
    }
	
	/**
	 * Assignment constructor for the value of K
	 * @param k K nearest neighbour
	 */
	public NearestNeighbourClassifier(int k) { this.k = k; }
	
	/**
	 * This method is called on the dataset of all the images inside of the training dataset, crops
	 * the image and preforms zero mean
	 */
    @Override
	protected void processTrainingImages(GroupedDataset<String, ? extends ListDataset<FImage>, FImage> trainingDataset)
	{
		for (Map.Entry<String, ? extends ListDataset<FImage>> images : trainingDataset.entrySet())
			for (FImage image : images.getValue())
			{
				FImage procImage = cropImage(image).processInplace(new MeanCenter()).normalise();
				features.add(new AbstractMap.SimpleEntry<>(procImage.getDoublePixelVector(), images.getKey()));
			}
	}
	
	/**
	 * This method is called on every image inside of the testing dataset to classify
	 * the image and write the classification to the output file
	 * @param image Image to be classified
	 */
	@Override
	protected String processTestingImage(FImage image)
	{
		try
		{
			//Crop the image and preform zero mean averaging
			double[] testingFeatures = cropImage(image).processInplace(new MeanCenter()).normalise().getDoublePixelVector();
			//TreeMap that sorts the neighbour on insertion
			SortedMap<Double, String> kNear = new TreeMap<Double, String>();
			
			for (Map.Entry<double[], String> feature : features)
			{
				double dist = 0;
				
				//Calculate the Euclidean distance between the testing features and training features
				for (int j = 0; j < feature.getKey().length; j++)
					dist += Math.abs(feature.getKey()[j] - testingFeatures[j]);
				
				kNear.put(dist, feature.getValue());
			}
			
			//Add the counts of the k nearest neighbours
			CountingHashMap counts = new CountingHashMap();
			Iterator<String> iterator = kNear.values().iterator();
			for (int j = 0; j < k; j++)
				counts.addCount(iterator.next());
			
			return counts.getHighestCount();
		}
		catch (Exception e) { e.printStackTrace(); return null; }
	}
	
	/**
	 * This method clears the feature list
	 */
	private void clearFeatures() { features = new ArrayList<Map.Entry<double[], String>>(); }
	
	/**
	 * This method crops an image into 16x16 pixels
	 * @param image Image to be cropped
	 * @return Cropped image
	 */
	private static FImage cropImage(FImage image)
	{
		int smallSide = Math.max(image.getHeight(), image.getWidth());
		FImage cropped = image.extractCenter(smallSide, smallSide);
		cropped.processInplace(new ResizeProcessor(16, 16));
		
		return cropped;
	}
	
	@Override
	protected void processValidation(GroupedDataset<String, ListDataset<FImage>, FImage> dataset)
	{
		//Number of images
		int total = 0;
		//Number of correct guesses
		int correct = 0;
		
		for (Map.Entry<String, ListDataset<FImage>> images : dataset.entrySet())
			for (FImage image : images.getValue())
			{
				String guess = processTestingImage(image);
				
				total++;
				if (guess.equals(images.getKey())) correct++;
			}
		
		float accuracy = ((float)correct/(float)total)*100;
		System.out.println("Validation Accuracy: " + accuracy + "%");
	}
}
