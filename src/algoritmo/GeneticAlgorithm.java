package algoritmo;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Random;
import java.util.Scanner;

import breakout.Breakout;
import pacman.Pacman;
import utils.BoardTypes;
import utils.Commons;

public class GeneticAlgorithm {
	private static final BoardTypes BOARD_TYPE = BoardTypes.BREAKOUT;
	private static final int POPULATION_SIZE = BOARD_TYPE == BoardTypes.BREAKOUT  ? 100  : 10; // The file size for 100 Pacman individuals is too large 
	private static final int NUM_GENERATIONS = 10000;
	private static final double MUTATION_RATE = 0.01;
	private static final int TOURNAMENT_SIZE = 10;
	private static final String FILENAME = BOARD_TYPE == BoardTypes.BREAKOUT ? "breakout.txt" : "pacman.txt";
	private static Random random = new Random();
	private static FeedforwardNeuralNetwork bestSolution;

	public GeneticAlgorithm() {
		// Initialize the population
		FeedforwardNeuralNetwork[] population = new FeedforwardNeuralNetwork[POPULATION_SIZE];
		try {
			population = readFile(FILENAME);
			System.out.println("File read successfully");
		} catch (Exception e) {
			System.out.println("Failed to read file, creating new population");
			for (int i = 0; i < POPULATION_SIZE; i++) {
				population[i] = generateNetwork();
			}
		}
		// Evolve the population for a fixed number of generations
		for (int i = 0; i < NUM_GENERATIONS; i++) {
			// Sort the population by fitness
			Arrays.sort(population, (a, b) -> (int) (b.getFitness() - a.getFitness()));
			// Print the best solution of this generation
			System.out.println("Generation " + i + ": " + population[0].getFitness());
			// Create the next generation
			for (int j = 0; j < POPULATION_SIZE; j++) {
				// Select two parents from the population
				int parent1Index = selectParent();
				int parent2Index = selectParent();
				while (parent2Index == parent1Index)
					parent2Index = selectParent();
				// Crossover the parents to create a new child
				double[] childNetwork = crossover(population[parent1Index].getNeuralNetwork(),
						population[parent2Index].getNeuralNetwork());
				// Mutate the child
				mutate(childNetwork);
				FeedforwardNeuralNetwork child = generateNetwork(childNetwork);
				child.runSimulation();
				// Add the child to the population
				if (population[parent1Index].getFitness() >= population[parent2Index].getFitness()) {
					population[parent2Index] = child;
				} else {
					population[parent1Index] = child;
				}
			}
		}
		// Print the best solution we found
		Arrays.sort(population, (a, b) -> (int) (b.getFitness() - a.getFitness()));
		System.out.println("Best solution found: " + population[0]);
		bestSolution = population[0];
		writePopulation(population, FILENAME);
	}

	private FeedforwardNeuralNetwork generateNetwork() {
		return BOARD_TYPE == BoardTypes.BREAKOUT
				? new FeedforwardNeuralNetwork(Commons.BREAKOUT_STATE_SIZE, Commons.BREAKOUT_HIDDEN_DIM,
						Commons.BREAKOUT_NUM_ACTIONS)
				: new FeedforwardNeuralNetwork(Commons.PACMAN_STATE_SIZE, Commons.PACMAN_HIDDEN_DIM,
						Commons.PACMAN_NUM_ACTIONS);
	}

	private FeedforwardNeuralNetwork generateNetwork(double[] values) {
		return BOARD_TYPE == BoardTypes.BREAKOUT
				? new FeedforwardNeuralNetwork(Commons.BREAKOUT_STATE_SIZE, Commons.BREAKOUT_HIDDEN_DIM,
						Commons.BREAKOUT_NUM_ACTIONS, values)
				: new FeedforwardNeuralNetwork(Commons.PACMAN_STATE_SIZE, Commons.PACMAN_HIDDEN_DIM,
						Commons.PACMAN_NUM_ACTIONS, values);
	}

	public static void main(String[] args) {
		new GeneticAlgorithm();
		if (BOARD_TYPE == BoardTypes.PACMAN)
			new Pacman(new FeedforwardNeuralNetwork(Commons.PACMAN_STATE_SIZE, Commons.PACMAN_HIDDEN_DIM,
					Commons.PACMAN_NUM_ACTIONS, bestSolution.getNeuralNetwork()), 1);
		else
			new Breakout(new FeedforwardNeuralNetwork(Commons.BREAKOUT_STATE_SIZE, Commons.BREAKOUT_HIDDEN_DIM,
					Commons.BREAKOUT_NUM_ACTIONS, bestSolution.getNeuralNetwork()), 1);
	}

	// Select the index of a parent from the population using tournament selection
	private int selectParent() {
		return random.nextInt(TOURNAMENT_SIZE);
	}

	// Crossover two parents to create a new child
	private double[] crossover(double[] parent1, double[] parent2) {
		double[] childValues = new double[parent1.length];
		for (int i = 0; i < childValues.length; i++) {
			if (random.nextDouble() < 0.5) { // TODO k-point crossover
				childValues[i] = parent1[i];
			} else {
				childValues[i] = parent2[i];
			}
		}
		return childValues;
	}

	// Mutate a FeedforwardNeuralNetwork by randomly changing one of its positions
	private void mutate(double[] feedforwardNeuralNetwork) {
		for (int i = 0; i < feedforwardNeuralNetwork.length; i++) {
			// Random value between -MUTATION_RATE / 2 and MUTATION_RATE / 2
			double mutationValue = (random.nextDouble() - 0.5) * MUTATION_RATE;
			feedforwardNeuralNetwork[i] += mutationValue;
		}
	}

	/**
	 * Writes the contents of the population's neural network array to a file
	 */
	private void writePopulation(FeedforwardNeuralNetwork[] population, String filename) {
		try {
			PrintWriter writer = new PrintWriter(new File(filename));

			for (FeedforwardNeuralNetwork individual : population) {
				double[] network = individual.getNeuralNetwork();
				writer.println(Arrays.toString(network));
			}

			writer.close();
		} catch (FileNotFoundException e) {
			System.out.println("An error occurred while trying to write to the file.");
			e.printStackTrace();
		}
	}

	/**
	 * Reads the contents of the file and returns them as an array of FFNNs
	 * The networks' simulations are ran in this function
	 * Throws an exception if the file doesn't have POPULATION_SIZE lines
	 */
	private FeedforwardNeuralNetwork[] readFile(String filename) throws FileNotFoundException {
		try {
			Scanner scanner = new Scanner(new File(filename));
			FeedforwardNeuralNetwork[] population = new FeedforwardNeuralNetwork[POPULATION_SIZE];
			int index = 0;
			while (scanner.hasNextLine()) {
				String line = scanner.nextLine();
				// Remove brackets
				line = line.substring(1, line.length() - 1);
				String[] values = line.split(", ");
				double[] network = new double[values.length];
				for (int i = 0; i < values.length; i++) {
					network[i] = Double.parseDouble(values[i]);
				}
				// Create the new network and run its simulation
				FeedforwardNeuralNetwork newNetwork = generateNetwork(network);
				newNetwork.runSimulation();
				population[index++] = newNetwork;
			}
			scanner.close();
			if (population.length != POPULATION_SIZE) {
				throw new IllegalArgumentException("Invalid number of individuals in the file");
			}
			return population;
		} catch (FileNotFoundException e) {
			throw e;
		}
	}
}
