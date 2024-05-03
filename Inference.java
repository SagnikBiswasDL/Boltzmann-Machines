import java.util.Arrays;

@SuppressWarnings("unused")
public class Inference {
	
	static int[][] J = {{0, -1, 2}, {-1, 0, 2}, {2, 2, 0}};
	static int[] H = {1, 1, -2};
	
	public static void main (String[] args) {
		
		for (int i = 0; i < 10; i++) {
			simulate();
		}
		
	}
	
	static void simulate() {
		int[] state = {-1, -1, -1};
		int[] freq = new int[8];
		
		int trials = 1000000;
		
		for (int i = 0; i < trials; i++) {
			// update the current state
			int[] I = new int[3];
			for (int j = 0; j < 3; j++) {
				for (int k = 0; k < 3; k++) {
					I[j] += state[k] * J[j][k];
				}
				
				I[j] += H[j];
				
				state[j] = sign(Math.tanh(I[j]) + 2*Math.random() - 1);
			}
			
			
			freq[value(state)]++;
		}
		
		
		
		double[] probability = new double[8];
		double totalSum = 0;
		for (int v : freq) totalSum += v;
		
		
		
		for (int i = 0; i < 8; i++) {
			probability[i] = ((double)freq[i])/totalSum;
			System.out.printf("%.5f ", probability[i]);
		}
		
		
		System.out.println();
	}
	
	static int sign(double x) {
	    return x < 0 ? -1 : 1;
	}

	
	static int value(int[] A) {
		return (A[0] + 1) * 2 + (A[1] + 1) + (A[2] + 1)/2;
	}
}
