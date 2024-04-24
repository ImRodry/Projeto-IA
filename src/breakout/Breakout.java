package breakout;

import javax.swing.JFrame;

import algoritmo.FeedforwardNeuralNetwork;
import utils.Commons;
import utils.GameController;

public class Breakout extends JFrame {

	private static final long serialVersionUID = 1L;

	public Breakout(GameController network, int i) {
		add(new BreakoutBoard(network, true, i));
		setTitle("Breakout");

		setDefaultCloseOperation(EXIT_ON_CLOSE);
		setLocationRelativeTo(null);
		setResizable(false);
		pack();
		setVisible(true);
	}

	public static void main(String[] args) {
		new Breakout(new FeedforwardNeuralNetwork(Commons.BREAKOUT_STATE_SIZE, 4, 1), 1);
	}

}
