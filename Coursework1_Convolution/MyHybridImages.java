import org.openimaj.image.processing.convolution.Gaussian2D;
import org.openimaj.image.MBFImage;

public class MyHybridImages {	
	
	public static MBFImage makeHybrid(MBFImage lowImage, float lowSigma, MBFImage highImage, float highSigma) {		
		MBFImage highPass = highImage.subtract(highImage.process(new MyConvolution(Gaussian2D.createKernelImage(getSize(highSigma), highSigma).pixels)));	
		return highPass.add(lowImage.process(new MyConvolution(Gaussian2D.createKernelImage(getSize(lowSigma), lowSigma).pixels)));
	}
	
	private static int getSize(float sigma) {
		int size = (int) (8.0f * sigma + 1.0f);
		if (size % 2 == 0) size++;
		return size;
	}
}