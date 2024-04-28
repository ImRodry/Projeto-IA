package pacman;

import java.awt.EventQueue;
import java.io.FileNotFoundException;

import javax.swing.JFrame;

import algoritmo.FeedforwardNeuralNetwork;
import algoritmo.GeneticAlgorithm;
import utils.GameController;

public class Pacman extends JFrame {

	public Pacman(GameController c, int seed) {
		EventQueue.invokeLater(() -> {
			add(new PacmanBoard(c, true, seed));

			setTitle("Pacman");
			setDefaultCloseOperation(EXIT_ON_CLOSE);
			setSize(380, 420);
			setLocationRelativeTo(null);
			setVisible(true);
		});
	}

	public static void main(String[] args) throws FileNotFoundException {
		FeedforwardNeuralNetwork bestNetwork = GeneticAlgorithm.readFile("pacman.txt", 1)[0];
		new Pacman(bestNetwork, 1);
	}
}
