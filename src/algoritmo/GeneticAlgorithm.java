package algoritmo;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;

import breakout.Breakout;

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
	private static final int TOURNAMENT_SIZE = 10;
	private static Random random = new Random();
	private static FeedforwardNeuralNetwork bestSolution;

	public GeneticAlgorithm() {
		// Initialize the population
		FeedforwardNeuralNetwork[] population = new FeedforwardNeuralNetwork[POPULATION_SIZE];
		for (int i = 0; i < POPULATION_SIZE; i++) {
			population[i] = new FeedforwardNeuralNetwork(Commons.BREAKOUT_STATE_SIZE, Commons.BREAKOUT_HIDDEN_DIM,
					Commons.BREAKOUT_NUM_ACTIONS);
		}
		// Evolve the population for a fixed number of generations
		for (int i = 0; i < NUM_GENERATIONS; i++) {
			// Sort the population by fitness
			Arrays.sort(population, (a, b) -> b.getFitness() - a.getFitness());
			// Print the best solution of this generation
			System.out.println("Generation " + i + ": " + population[0]);
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
				child.runSimulation();
			}
			// Replace the old population with the new population
			population = newPopulation;
		}
		// Print the best solution we found
		Arrays.sort(population, (a, b) -> b.getFitness() - a.getFitness());
		bestSolution = population[0];
		System.out.println("Best solution found: " + population[0]);
	}

	public static void main(String[] args) {
		new GeneticAlgorithm();
		new Breakout(new FeedforwardNeuralNetwork(Commons.BREAKOUT_STATE_SIZE, Commons.BREAKOUT_HIDDEN_DIM,
				Commons.BREAKOUT_NUM_ACTIONS, bestSolution.getNeuralNetwork(), true), 1);
	}

	// Select a parent from the population using tournament selection
	private FeedforwardNeuralNetwork selectParent(FeedforwardNeuralNetwork[] population) {
		ArrayList<FeedforwardNeuralNetwork> tournament = new ArrayList<>();
		for (int i = 0; i < TOURNAMENT_SIZE; i++) {
			tournament.add(population[random.nextInt(population.length)]);
		}
		Collections.sort(tournament, (a, b) -> b.getFitness() - a.getFitness());
		return tournament.get(0);
	}

	// Crossover two parents to create a new child
	private FeedforwardNeuralNetwork crossover(FeedforwardNeuralNetwork parent1, FeedforwardNeuralNetwork parent2) {
		double[] parent1Values = parent1.getNeuralNetwork();
		double[] parent2Values = parent2.getNeuralNetwork();
		double[] childValues = new double[parent1Values.length];
		for (int i = 0; i < childValues.length; i++) {
			if (random.nextDouble() < 0.5) { // TODO k-point crossover
				childValues[i] = parent1Values[i];
			} else {
				childValues[i] = parent2Values[i];
			}
		}
		return new FeedforwardNeuralNetwork(Commons.BREAKOUT_STATE_SIZE, Commons.BREAKOUT_HIDDEN_DIM,
				Commons.BREAKOUT_NUM_ACTIONS, childValues);
	}

	// Mutate a FeedforwardNeuralNetwork by randomly changing one of its positions
	private void mutate(FeedforwardNeuralNetwork feedforwardNeuralNetwork) {
		double[] values = feedforwardNeuralNetwork.getNeuralNetwork();
		if (random.nextDouble() < MUTATION_RATE) {
			values[random.nextInt(values.length)] = random.nextDouble();
		}
	}
}
