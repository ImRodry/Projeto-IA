package algoritmo;

import java.util.Random;

import breakout.BreakoutBoard;
import utils.Commons;
import utils.GameController;

public class FeedforwardNeuralNetwork implements GameController {
	private int inputDim;
	private int hiddenDim;
	private int outputDim;
	private double[][] hiddenWeights;
	private double[][] outputWeights;
	private double[] hiddenBiases;
	private double[] outputBiases;
	private BreakoutBoard board = new BreakoutBoard(this, false, new Random().nextInt(1000000));

	public FeedforwardNeuralNetwork(int inputDim, int hiddenDim, int outputDim) {
		this.inputDim = inputDim;
		this.hiddenDim = hiddenDim;
		this.outputDim = outputDim;
		initializeParameters();
	}

	// [w1,1; w1,2; w2,1; w2,2; B1; B2; w1,o; w2,o; Bo]
	public FeedforwardNeuralNetwork(int inputDim, int hiddenDim, int outputDim, double[] values) {
		this.inputDim = inputDim;
		this.hiddenDim = hiddenDim;
		this.outputDim = outputDim;
		applyWeightsAndBiases(values);
	}

	public void applyWeightsAndBiases(double[] values) {
		hiddenWeights = new double[inputDim][hiddenDim];
		outputWeights = new double[hiddenDim][outputDim];
		hiddenBiases = new double[hiddenDim];
		outputBiases = new double[outputDim];

		if (values.length != inputDim * hiddenDim + hiddenDim + hiddenDim * outputDim + outputDim) {
			throw new IllegalArgumentException("Invalid number of parameters");
		}

		int index = 0;
		for (int i = 0; i < inputDim; i++) {
			for (int j = 0; j < hiddenDim; j++) {
				hiddenWeights[i][j] = values[index++];
			}
		}
		for (int i = 0; i < hiddenDim; i++) {
			hiddenBiases[i] = values[index++];
		}
		for (int i = 0; i < hiddenDim; i++) {
			for (int j = 0; j < outputDim; j++) {
				outputWeights[i][j] = values[index++];
			}
		}
		for (int i = 0; i < outputDim; i++) {
			outputBiases[i] = values[index++];
		}
	}

	public double[] forward(int[] inputValues) {
		if (inputValues.length != inputDim) {
			throw new IllegalArgumentException("Invalid number of input values");
		}
		double[] hiddenLayer = new double[hiddenDim];
		for (int i = 0; i < hiddenDim; i++) {
			double sum = 0;
			for (int j = 0; j < inputDim; j++) {
				sum += inputValues[j] * hiddenWeights[j][i];
			}
			hiddenLayer[i] = sigmoid(sum + hiddenBiases[i]);
		}
		double[] outputLayer = new double[outputDim];
		for (int i = 0; i < outputDim; i++) {
			double sum = 0;
			for (int j = 0; j < hiddenDim; j++)
				sum += hiddenLayer[j] * outputWeights[j][i];

			outputLayer[i] = sigmoid(sum + outputBiases[i]);
		}
		return outputLayer;
	}

	private double sigmoid(double z) {
		return 1 / (1 + Math.exp(-z));
	}

	public double[] getNeuralNetwork() {
		double[] result = new double[inputDim * hiddenDim + hiddenDim + hiddenDim * outputDim + outputDim];
		int index = 0;
		for (int i = 0; i < inputDim; i++) {
			for (int j = 0; j < hiddenDim; j++) {
				result[index++] = hiddenWeights[i][j];
			}
		}
		for (int i = 0; i < hiddenDim; i++) {
			result[index++] = hiddenBiases[i];
		}

		for (int i = 0; i < hiddenDim; i++) {
			for (int j = 0; j < outputDim; j++) {
				result[index++] = outputWeights[i][j];
			}
		}
		for (int i = 0; i < outputDim; i++) {
			result[index++] = outputBiases[i];
		}
		return result;
	}

	private void initializeParameters() {
		hiddenWeights = new double[inputDim][hiddenDim];
		outputWeights = new double[hiddenDim][outputDim];
		hiddenBiases = new double[hiddenDim];
		outputBiases = new double[outputDim];

		// The for loop is intentionally swapped here so that the biases can also be
		// generated without requiring an extra loop
		for (int i = 0; i < hiddenDim; i++) {
			hiddenBiases[i] = Math.random() / 2;
			for (int j = 0; j < inputDim; j++) {
				hiddenWeights[j][i] = Math.random() / 2;
			}
		}
		for (int i = 0; i < outputDim; i++) {
			outputBiases[i] = Math.random() / 2;
			for (int j = 0; j < hiddenDim; j++) {
				outputWeights[j][i] = Math.random() / 2;
			}
		}
	}

	@Override
	public String toString() {
		String result = "Neural Network: \nNumber of inputs: "
				+ inputDim + "\n"
				+ "Weights between input and hidden layer with " + hiddenDim + " neurons: \n";
		String hidden = "";
		for (int input = 0; input < inputDim; input++) {
			for (int i = 0; i < hiddenDim; i++) {
				hidden += " w" + (input + 1) + (i + 1) + ": "
						+ hiddenWeights[input][i] + "\n";
			}
		}
		result += hidden;
		String biasHidden = "Hidden biases: \n";
		for (int i = 0; i < hiddenDim; i++) {
			biasHidden += " b " + (i + 1) + ": " + hiddenBiases[i] + "\n";
		}
		result += biasHidden;
		String output = "Weights between hidden and output layer with "
				+ outputDim + " neurons: \n";
		for (int hiddenw = 0; hiddenw < hiddenDim; hiddenw++) {
			for (int i = 0; i < outputDim; i++) {
				output += " w" + (hiddenw + 1) + "o" + (i + 1) + ": "
						+ outputWeights[hiddenw][i] + "\n";
			}
		}
		result += output;
		String biasOutput = "Ouput biases: \n";
		for (int i = 0; i < outputDim; i++) {
			biasOutput += " bo" + (i + 1) + ": " + outputBiases[i] + "\n";
		}
		result += biasOutput;
		return result;
	}

	public static void main(String[] args) {
		double[] values = { 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.2, 0.1, 0.3, 0.3, 0.5 };
		FeedforwardNeuralNetwork nn_values = new FeedforwardNeuralNetwork(3, 2, 1, values);
		System.out.println(nn_values);
		System.out.println("Forward result:");
		double[] input = { 0.0, 0.0, 1.0 };
		double[] result = nn_values.forward(input);
		for (int i = 0; i < result.length; i++) {
			System.out.println(" Result for neuron ouput" + (i + 1) +
					"= " + result[i]);
		}
	}

	@Override
	public int nextMove(int[] currentState) {
		return (int)(Math.random() * Commons.BREAKOUT_NUM_ACTIONS + 1);
	}

	public int getFitness() {
		return board.getFitness();
	}

	public void runSimulation() {
		board.runSimulation();
	}
}
