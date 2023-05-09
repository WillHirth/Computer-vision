package Helper;

import java.util.HashMap;
import java.util.Map;

/**
 * This class counts the number of times a string is seen
 */
public class CountingHashMap
{
	private HashMap<String, Integer> counts = new HashMap<>();
	
	/**
	 * Increases the count associated with the label
	 * @param label String to be counted
	 */
	public void addCount(String label)
	{
		if (counts.containsKey(label)) counts.put(label, counts.get(label) + 1);
		else counts.put(label, 1);
	}
	
	/**
	 * Finds the string with the highest count
	 * @return The string that has been seen the most / has the highest count
	 */
	public String getHighestCount()
	{
		Map.Entry<String, Integer> highestCount = null;
		
		//Finds the highest count
		for (Map.Entry<String, Integer> entry : counts.entrySet())
			if (highestCount == null || entry.getValue() > highestCount.getValue()) highestCount = entry;
			
		assert highestCount != null;
		return highestCount.getKey();
	}
}
