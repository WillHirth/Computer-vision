package Helper;

import org.openimaj.data.dataset.*;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.FImage;

import java.io.BufferedWriter;
import java.io.File;
import java.io.PrintWriter;

/**
 * This is an abstract class that implements image classification though training, testing and validation
 */
public abstract class Classifier
{
	/** File to output the classifications to */
	protected File output;
	
	/**
	 * This method uses the training dataset to train the classifier
	 * @param path Path to the training dataset
	 */
	public void train(String path)
	{
		try
		{
			VFSGroupDataset<FImage> trainingDataset = new VFSGroupDataset<FImage>(new File(path).getAbsolutePath(), ImageUtilities.FIMAGE_READER);
			trainingDataset.remove("training");
			
			processTrainingImages(trainingDataset);
		}
		catch (Exception e) { e.printStackTrace(); }
	}
	
	/**
	 * This method uses the testing dataset to classify the images
	 * @param path Path to the testing dataset
	 */
	public void test(String path)
	{
		try
		{
			VFSListDataset<FImage> testingDataset = new VFSListDataset<FImage>(new File(path).getAbsolutePath(), ImageUtilities.FIMAGE_READER);
			BufferedWriter writer = new BufferedWriter(new PrintWriter(output));
			
			for (int i = 0; i < testingDataset.size(); i++)
			{
				String guess = processTestingImage(testingDataset.get(i));
				
				//Write the highest number of neighbours to output
				writer.write(testingDataset.getID(i) + " " + guess);
				writer.newLine();
			}
			
			writer.close();
		}
		catch (Exception e) { e.printStackTrace(); }
	}
	
	/**
	 * Calculates accuracy on a section of the training data
	 * @param path Path to the training data
	 */
	public void validate(String path)
	{
		try
		{
			VFSGroupDataset<FImage> trainingDataset = new VFSGroupDataset<FImage>(new File(path).getAbsolutePath(), ImageUtilities.FIMAGE_READER);
			trainingDataset.remove("training");
			GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<String, FImage>(trainingDataset, 80, 0, 20);
			
			processTrainingImages(splits.getTrainingDataset());
			processValidation(splits.getTestDataset());
		}
		catch (Exception e) { e.printStackTrace(); }
	}
	
	protected abstract void processTrainingImages(GroupedDataset<String, ? extends ListDataset<FImage>, FImage> trainingDataset);
	protected abstract String processTestingImage(FImage image);
	protected abstract void processValidation(GroupedDataset<String, ListDataset<FImage>, FImage> splits);
	
	public void setOutputFile(String path) { output = new File(path); }
}
