import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;
import java.util.StringTokenizer;

public class IsingModel {
	

    static double[][] J;
    static double[] h;
    static int NT;
    static int N;
    static int[][] truthTable;
    
    public static void main(String[] args) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader("fulladder.txt"));
        StringTokenizer st = new StringTokenizer(br.readLine());
        
        NT = Integer.parseInt(st.nextToken());
        N = Integer.parseInt(st.nextToken());
        J = new double[N][N];
        h = new double[N];
        truthTable = new int[NT][N];
        
        for (int i = 0; i < NT; i++) {
            st = new StringTokenizer(br.readLine());
            for (int j = 0; j < N; j++) {
                truthTable[i][j] = Integer.parseInt(st.nextToken());
            }
        }
        
        br.close();
        
        int num_iterations = 10000;
        double learning_rate = 0.1;
        
        // Initialize weights and biases
        Random rand = new Random();
        for (int i = 0; i < N; i++) {
            h[i] = rand.nextDouble() * 0.01;
            for (int j = 0; j < N; j++) {
                J[i][j] = rand.nextDouble() * 0.01;
            }
        }
        
        
        double[][] dataCorrelation = new double[N][N];
        double[] dataAverage = new double[N];
        for (int i = 0; i < NT; i++) {
        	for (int row = 0; row < N; row++) {
        		for (int col = 0; col < N; col++) {
        			double add = truthTable[i][row] * truthTable[i][col];
        			add /= NT;
        			dataCorrelation[row][col] += add;
        		}
        	}
        	
        	for (int index = 0; index < N; index++) {
        		double add = truthTable[i][index];
        		add /= NT;
        		dataAverage[index] += add;
        	}
        }
        
        // need to compute dJ and need to compute dh
        
        // Main training loop
        for (int epoch = 0; epoch < num_iterations; epoch++) {
        	double[][] modelCorrelation = new double[N][N];
        	double[] modelAverage = new double[N];
        	// start at some state and explore
        	
        	int[] state = new int[N];
        	
        	// clamp to an input example
    		Arrays.fill(state, -1);
    		
    		
    		int trials = 1000;
    		
    		// generate states - typically would be done with a probabilistic computer
    		for (int i = 0; i < trials; i++) {
    			// update the current state
    			double[] I = new double[N];
    			for (int j = 0; j < N; j++) {
    				for (int k = 0; k < N; k++) {
    					I[j] += state[k] * J[j][k];
    				}
    				I[j] += h[j];
    				state[j] = sign(Math.tanh(I[j]) + 2*Math.random() - 1);
    			}
    			
    			
    			for (int r = 0; r < N; r++) {
    				for (int c = 0; c < N; c++) {
    					double add = state[r] * state[c];
    					add /= trials;
    					modelCorrelation[r][c] += add;
    				}
    			}
    			
    			for (int index = 0; index < N; index++) {
    				double add = state[index];
    				add /= trials;
    				modelAverage[index] += add;
    			}
    			
    		}
    		
        	
        	updateJ(dataCorrelation, modelCorrelation, learning_rate);
        	updateH(dataAverage, modelAverage, learning_rate);
        }
        
        // Optionally, display some results or status here
        simulate();
        
        System.out.println("Training complete.");
    }
	
    static void updateJ(double[][] dataCorrelation, double[][] modelCorrelation, double epsilon) {
    	for (int i = 0; i < N; i++) {
    		for (int j = 0; j < N; j++) {
    			J[i][j] += epsilon * (dataCorrelation[i][j] - modelCorrelation[i][j]);
    		}
    	}
    }
    
    static void updateH(double[] data, double[] model, double epsilon) {
    	for (int i = 0; i < N; i++) {
    		h[i] += epsilon * (data[i] - model[i]);
    	}
    }
    
	static void simulate() {
		int[] state = new int[N];
		Arrays.fill(state, -1);
		int[] freq = new int[(1<<N)];
		
		int trials = 1000000;
		
		for (int i = 0; i < trials; i++) {
			// update the current state
			double[] I = new double[N];
			for (int j = 0; j < N; j++) {
				for (int k = 0; k < N; k++) {
					I[j] += state[k] * J[j][k];
				}
				I[j] += h[j];
				state[j] = sign(Math.tanh(I[j]) + 2*Math.random() - 1);
			}
			
			
			freq[value(state)]++;
		}
		
		
		
		double[] probability = new double[(1<<N)];
		double totalSum = 0;
		for (int v : freq) totalSum += v;
		
		
		
		for (int i = 0; i < (1<<N); i++) {
			probability[i] = ((double)freq[i])/totalSum;
			if (probability[i] > 0.01) {
				System.out.printf("[Prob(%d): %.2f]\n",i,  probability[i]);
			}
		}
		
		System.out.println();
	}
	
	static int sign(double x) {
	    return x < 0 ? -1 : 1;
	}
	
	static int value(int[] A) {
		int power = 1;
		int sum = 0;
		for (int i = A.length - 1; i >= 0; i--) {
			sum += power * ((A[i]+1)/2);
			power *= 2;
		}
		return sum;
	}
}
