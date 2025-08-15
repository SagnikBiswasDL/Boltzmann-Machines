import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;
import java.util.StringTokenizer;

public class EquilibriumPropagation {
	static double[][] J; // symmetric couplings (zero diagonal)
	static double[] b;   // biases
	static int NT;       // number of training patterns
	static int N;        // number of units
	static int[][] truthTable; // NT x N, entries in {-1, +1}

	public static void main(String[] args) throws IOException {
		if (args.length < 2) {
			System.out.println("Usage: java EquilibriumPropagation <data_file> <num_outputs> [epochs] [beta] [lr] [freeSteps] [nudgedSteps] [seed]");
			System.out.println("Examples:");
			System.out.println("  java EquilibriumPropagation andgate.txt 1");
			System.out.println("  java EquilibriumPropagation fulladder.txt 2 2000 0.5 0.05 50 30 42");
			return;
		}

		String dataPath = args[0];
		int numOutputs = Integer.parseInt(args[1]);
		int epochs = args.length > 2 ? Integer.parseInt(args[2]) : 2000;
		double beta = args.length > 3 ? Double.parseDouble(args[3]) : 0.5;
		double learningRate = args.length > 4 ? Double.parseDouble(args[4]) : 0.05;
		int freeSteps = args.length > 5 ? Integer.parseInt(args[5]) : 50;
		int nudgedSteps = args.length > 6 ? Integer.parseInt(args[6]) : 30;
		long seed = args.length > 7 ? Long.parseLong(args[7]) : 42L;

		loadDataset(dataPath);
		int numInputs = N - numOutputs;
		if (numInputs <= 0) {
			throw new IllegalArgumentException("numOutputs must be < N");
		}

		Random rng = new Random(seed);
		J = new double[N][N];
		b = new double[N];
		for (int i = 0; i < N; i++) {
			b[i] = (rng.nextDouble() - 0.5) * 0.02;
			for (int j = i + 1; j < N; j++) {
				double w = (rng.nextDouble() - 0.5) * 0.02;
				J[i][j] = w;
				J[j][i] = w;
			}
		}

		trainEP(numInputs, numOutputs, epochs, beta, learningRate, freeSteps, nudgedSteps, rng);

		System.out.println("Training complete. Evaluating...");
		evaluate(numInputs, numOutputs);
	}

	static void loadDataset(String path) throws IOException {
		try (BufferedReader br = new BufferedReader(new FileReader(path))) {
			StringTokenizer st = new StringTokenizer(br.readLine());
			NT = Integer.parseInt(st.nextToken());
			N = Integer.parseInt(st.nextToken());
			truthTable = new int[NT][N];
			for (int i = 0; i < NT; i++) {
				st = new StringTokenizer(br.readLine());
				for (int j = 0; j < N; j++) {
					truthTable[i][j] = Integer.parseInt(st.nextToken());
				}
			}
		}
	}

	static void trainEP(int numInputs, int numOutputs, int epochs, double beta, double learningRate, int freeSteps, int nudgedSteps, Random rng) {
		// Batch gradients
		for (int epoch = 1; epoch <= epochs; epoch++) {
			double[][] gradJ = new double[N][N];
			double[] gradB = new double[N];

			for (int p = 0; p < NT; p++) {
				// Prepare input and target
				double[] inputs = new double[numInputs];
				double[] targets = new double[numOutputs];
				for (int i = 0; i < numInputs; i++) inputs[i] = truthTable[p][i];
				for (int o = 0; o < numOutputs; o++) targets[o] = truthTable[p][numInputs + o];

				// Initialize state near zero, clamp inputs
				double[] s = new double[N];
				for (int i = 0; i < N; i++) s[i] = (rng.nextDouble() - 0.5) * 0.1;
				for (int i = 0; i < numInputs; i++) s[i] = inputs[i];

				// Free phase
				double[] sFree = runPhase(s, /*beta*/0.0, inputs, targets, numInputs, freeSteps);

				// Nudged phase (start from free fixed point)
				double[] sNudged = runPhase(Arrays.copyOf(sFree, N), beta, inputs, targets, numInputs, nudgedSteps);

				// One-sample estimates for correlations and means
				for (int i = 0; i < N; i++) {
					gradB[i] += (sNudged[i] - sFree[i]) / beta;
					for (int j = 0; j < N; j++) {
						gradJ[i][j] += (sNudged[i] * sNudged[j] - sFree[i] * sFree[j]) / beta;
					}
				}
			}

			// Average over patterns
			double invNT = 1.0 / NT;
			for (int i = 0; i < N; i++) {
				b[i] += learningRate * invNT * gradB[i];
				for (int j = 0; j < N; j++) {
					J[i][j] += learningRate * invNT * gradJ[i][j];
				}
			}

			// Enforce symmetry and zero diagonal for stability
			for (int i = 0; i < N; i++) {
				J[i][i] = 0.0;
				for (int j = i + 1; j < N; j++) {
					double w = 0.5 * (J[i][j] + J[j][i]);
					J[i][j] = w;
					J[j][i] = w;
				}
			}

			if (epoch % Math.max(1, epochs / 10) == 0) {
				System.out.printf("Epoch %d/%d\n", epoch, epochs);
			}
		}
	}

	static double[] runPhase(double[] sInit, double beta, double[] inputs, double[] targets, int numInputs, int steps) {
		double[] s = Arrays.copyOf(sInit, sInit.length);
		for (int t = 0; t < steps; t++) {
			// Update non-input units synchronously using tanh nonlinearity
			double[] newS = Arrays.copyOf(s, s.length);
			for (int i = 0; i < s.length; i++) {
				if (i < numInputs) {
					newS[i] = inputs[i]; // clamp inputs
					continue;
				}
				double field = b[i];
				for (int j = 0; j < s.length; j++) {
					field += J[i][j] * s[j];
				}
				// Nudging on outputs only: field -= beta * (s_i - target_i)
				if (i >= numInputs) {
					int outIndex = i - numInputs; // 0..numOutputs-1
					if (outIndex < targets.length) {
						field -= beta * (s[i] - targets[outIndex]);
					}
				}
				newS[i] = Math.tanh(field);
			}
			s = newS;
		}
		return s;
	}

	static void evaluate(int numInputs, int numOutputs) {
		int correct = 0;
		for (int p = 0; p < NT; p++) {
			double[] inputs = new double[numInputs];
			double[] targets = new double[numOutputs];
			for (int i = 0; i < numInputs; i++) inputs[i] = truthTable[p][i];
			for (int o = 0; o < numOutputs; o++) targets[o] = truthTable[p][numInputs + o];

			double[] s0 = new double[N];
			for (int i = 0; i < numInputs; i++) s0[i] = inputs[i];
			// small relaxation from inputs only
			double[] s = runPhase(s0, 0.0, inputs, targets, numInputs, 60);

			int ok = 1;
			for (int o = 0; o < numOutputs; o++) {
				int idx = numInputs + o;
				int pred = s[idx] >= 0 ? 1 : -1;
				if (pred != (int)targets[o]) ok = 0;
			}
			correct += ok;
		}
		System.out.printf("Accuracy: %d/%d (%.1f%%)\n", correct, NT, 100.0 * correct / NT);
	}
}