package breakout;

import javax.swing.JFrame;

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
}
