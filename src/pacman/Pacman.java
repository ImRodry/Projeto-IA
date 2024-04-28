package pacman;

import java.awt.EventQueue;

import javax.swing.JFrame;

import algoritmo.FeedforwardNeuralNetwork;
import utils.Commons;
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

	public static void main(String[] args) {
		new Pacman(new FeedforwardNeuralNetwork(Commons.PACMAN_STATE_SIZE, Commons.PACMAN_HIDDEN_DIM,
				Commons.PACMAN_NUM_ACTIONS), 1);
	}
}
