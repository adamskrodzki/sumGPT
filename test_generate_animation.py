import unittest
import matplotlib.pyplot as plt
from sumGPT.generate_animation import logit_to_color

class TestLogitToColor(unittest.TestCase):

    def test_logit_to_color_positive_logit(self):
        logit = 1.0
        expected_color = plt.cm.Greens(255 * (1 - 1 / (1 + logit)))
        self.assertEqual(logit_to_color(logit), expected_color)

    def test_logit_to_color_negative_logit(self):
        logit = -1.0
        expected_color = plt.cm.Greens(255 * (1 - 1 / (1 + abs(logit))))
        self.assertEqual(logit_to_color(logit), expected_color)

    def test_logit_to_color_zero_logit(self):
        logit = 0.0
        expected_color = plt.cm.Greens(255 * (1 - 1 / (1 + abs(logit))))
        self.assertEqual(logit_to_color(logit), expected_color)

if __name__ == "__main__":
    unittest.main()
