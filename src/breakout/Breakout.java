package breakout;

import java.io.FileNotFoundException;

import javax.swing.JFrame;

import algoritmo.FeedforwardNeuralNetwork;
import algoritmo.GeneticAlgorithm;
import utils.Commons;
import utils.GameController;

public class Breakout extends JFrame {

	private static final long serialVersionUID = 1L;

	public Breakout(GameController network, int seed) {
		add(new BreakoutBoard(network, true, seed));
		setTitle("Breakout");

		setDefaultCloseOperation(EXIT_ON_CLOSE);
		setLocationRelativeTo(null);
		setResizable(false);
		pack();
		setVisible(true);
	}

	public static void main(String[] args) throws FileNotFoundException {
		FeedforwardNeuralNetwork bestNetwork = GeneticAlgorithm.readFile("breakout.txt", 1)[0];
		new Breakout(bestNetwork, Commons.SEED);
	}
}
