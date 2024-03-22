package algoritmo;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;

import utils.Commons;

public class GeneticAlgorithm {
	/*
	 * This implementation uses a population of 100 solutions,
	 * evolves the population for 1000 generations, and uses tournament selection to
	 * select parents for crossover. Each solution is represented by an array of
	 * integers
	 * representing the column index of each FeedforwardNeuralNetwork in its row.
	 * The fitness of a
	 * solution
	 * is the number of pairs of FeedforwardNeuralNetworks that are attacking each
	 * other.
	 * Crossover is performed by randomly selecting a row from each parent, and the
	 * child is mutated by randomly changing one of its positions with a probability
	 * of 0.01.
	 * The best solution found is printed after each generation, and the best
	 * solution overall
	 * is printed at the end.
	 */
	private static final int POPULATION_SIZE = 100;
	private static final int NUM_GENERATIONS = 1000;
	private static final double MUTATION_RATE = 0.01;
	private static final int BOARD_SIZE = 8;
	private static Random random = new Random();

	public GeneticAlgorithm() {
		// Initialize the population
		FeedforwardNeuralNetwork[] population = new FeedforwardNeuralNetwork[POPULATION_SIZE];
		for (int i = 0; i < POPULATION_SIZE; i++) {
			population[i] = new FeedforwardNeuralNetwork(Commons.BREAKOUT_STATE_SIZE, 4, 1);
		}
		// Evolve the population for a fixed number of generations
		for (int i = 0; i < NUM_GENERATIONS; i++) {
			// Sort the population by fitness
			Arrays.sort(population);
			// Print the best solution of this generation
			System.out.println("Generation " + i + ": " + population[0]);
			// Check if we have found a solution
			if (population[0].getFitness() == 0) {
				break;
			}
			// Create the next generation
			FeedforwardNeuralNetwork[] newPopulation = new FeedforwardNeuralNetwork[POPULATION_SIZE];
			for (int j = 0; j < POPULATION_SIZE; j++) {
				// Select two parents from the population
				FeedforwardNeuralNetwork parent1 = selectParent(population);
				FeedforwardNeuralNetwork parent2 = selectParent(population);
				// Crossover the parents to create a new child
				FeedforwardNeuralNetwork child = crossover(parent1, parent2);
				// Mutate the child
				mutate(child);
				// Add the child to the new population
				newPopulation[j] = child;
			}
			// Replace the old population with the new population
			population = newPopulation;
		}
		// Print the best solution we found
		Arrays.sort(population);
		System.out.println("Best solution found: " + population[0]);
	}

	public static void main(String[] args) {
		new GeneticAlgorithm();
	}

	// Select a parent from the population using tournament selection
	private FeedforwardNeuralNetwork selectParent(FeedforwardNeuralNetwork[] population) {
		ArrayList<FeedforwardNeuralNetwork> tournament = new ArrayList<>();
		for (int i = 0; i < 10; i++) {
			tournament.add(population[random.nextInt(population.length)]);
		}
		Collections.sort(tournament);
		return tournament.get(0);
	}

	// Crossover two parents to create a new child
	private FeedforwardNeuralNetwork crossover(FeedforwardNeuralNetwork parent1, FeedforwardNeuralNetwork parent2) {
		FeedforwardNeuralNetwork child = new FeedforwardNeuralNetwork();
		for (int i = 0; i < BOARD_SIZE; i++) {
			if (random.nextDouble() < 0.5) {
				child.setRow(i, parent1.getRows()[i]);
			} else {
				child.setRow(i, parent2.getRows()[i]);
			}
		}
		return child;
	}

	// Mutate a FeedforwardNeuralNetwork by randomly changing one of its positions
	private void mutate(FeedforwardNeuralNetwork FeedforwardNeuralNetwork) {
		if (random.nextDouble() < MUTATION_RATE) {
			int row = random.nextInt(BOARD_SIZE);
			int col = random.nextInt(BOARD_SIZE);
			FeedforwardNeuralNetwork.setRow(row, col);
		}
	}
}
