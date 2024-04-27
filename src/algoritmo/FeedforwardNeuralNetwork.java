package algoritmo;

import breakout.BreakoutBoard;
import pacman.PacmanBoard;
import utils.*;

public class FeedforwardNeuralNetwork implements GameController {
	private int inputDim;
	private int hiddenDim;
	private int outputDim;
	private double[][] hiddenWeights;
	private double[][] outputWeights;
	private double[] hiddenBiases;
	private double[] outputBiases;
	private Board board;

	public FeedforwardNeuralNetwork(int inputDim, int hiddenDim, int outputDim) {
		this.inputDim = inputDim;
		this.hiddenDim = hiddenDim;
		this.outputDim = outputDim;
		this.board = inputDim == Commons.BREAKOUT_STATE_SIZE ? new BreakoutBoard(this, false, 0)
				: new PacmanBoard(this, false, 0);
		initializeParameters();
	}

	// [w1,1; w1,2; w2,1; w2,2; B1; B2; w1,o; w2,o; Bo]
	public FeedforwardNeuralNetwork(int inputDim, int hiddenDim, int outputDim, double[] values) {
		this.inputDim = inputDim;
		this.hiddenDim = hiddenDim;
		this.outputDim = outputDim;
		this.board = inputDim == Commons.BREAKOUT_STATE_SIZE ? new BreakoutBoard(this, false, 0)
				: new PacmanBoard(this, false, 0);
		applyWeightsAndBiases(values);
	}

	public void applyWeightsAndBiases(double[] values) {
		hiddenWeights = new double[inputDim][hiddenDim];
		outputWeights = new double[hiddenDim][outputDim];
		hiddenBiases = new double[hiddenDim];
		outputBiases = new double[outputDim];

		if (values.length != (inputDim == Commons.BREAKOUT_STATE_SIZE ? Commons.BREAKOUT_NETWORK_SIZE
				: Commons.PACMAN_NETWORK_SIZE)) {
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
		double[] result = new double[inputDim == Commons.BREAKOUT_STATE_SIZE ? Commons.BREAKOUT_NETWORK_SIZE
				: Commons.PACMAN_NETWORK_SIZE];
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
			for (int h = 0; h < hiddenDim; h++) {
				hidden += " w" + (input + 1) + "_" + (h + 1) + ": "
						+ hiddenWeights[input][h] + "\n";
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
		for (int hiddenW = 0; hiddenW < hiddenDim; hiddenW++) {
			for (int i = 0; i < outputDim; i++) {
				output += " w" + (hiddenW + 1) + "o" + (i + 1) + ": "
						+ outputWeights[hiddenW][i] + "\n";
			}
		}
		result += output;
		String biasOutput = "Ouput biases: \n";
		for (int i = 0; i < outputDim; i++) {
			biasOutput += " bo" + (i + 1) + ": " + outputBiases[i] + "\n";
		}
		result += biasOutput + "\nFitness: " + getFitness();
		return result;
	}

	@Override
	public int nextMove(int[] currentState) {
		double[] outputLayer = forward(currentState);
		if (outputLayer[0] > outputLayer[1])
			return BreakoutBoard.LEFT;
		return BreakoutBoard.RIGHT;
	}

	public double getFitness() {
		return board.getFitness();
	}

	public void runSimulation() {
		board.runSimulation();
	}
}
