import org.openimaj.image.FImage;
import org.openimaj.image.processor.SinglebandImageProcessor;

public class MyConvolution implements SinglebandImageProcessor<Float, FImage> {
	private float[][] kernel;
	private int paddingx;
	private int paddingy;

	public MyConvolution(float[][] kernel) {
		this.kernel = kernel;    
		this.paddingx = Math.floorDiv(kernel.length, 2);
		this.paddingy = Math.floorDiv(kernel[0].length, 2);
	}

	@Override
	public void processImage(FImage image) {
		FImage buffer = new FImage(image.getWidth(), image.getHeight());
		
		for (int x = 0; x < buffer.getHeight(); x++)
			for (int y = 0; y < buffer.getWidth(); y++) {
				float sum = 0;
				
				for (int i = x - this.paddingx; i <= x + this.paddingx; i++)
					for (int j = y - this.paddingy; j <= y + this.paddingy; j++)
						if (!(i < 0 || j < 0 || i >= buffer.getHeight() || j >= buffer.getWidth()))
							sum += image.pixels[i][j]*kernel[x - i + this.paddingx][y - j + this.paddingy];		
				
				//floating point addition issues
//				for (int i = 0; i < this.kernel.length; i++)
//				    for(int j = 0; j < this.kernel[0].length; j++)
//				    	if (!(y - j + this.paddingy < 0 || x - i + this.paddingx < 0 || y - j + this.paddingy >= buffer.getWidth() || x - i + this.paddingx >= buffer.getHeight())) 
//				    		sum += image.getPixel(y - j + this.paddingy, x - i + this.paddingx) * this.kernel[i][j];
				
		    	buffer.setPixel(y, x, sum);			   
			}				
		image.internalAssign(buffer);			
	}
}