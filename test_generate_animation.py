import unittest
import matplotlib.pyplot as plt
from sumGPT.generate_animation import logit_to_color

class TestLogitToColor(unittest.TestCase):

    def test_logit_to_color_positive_logit(self):
        logit = 1.0
        INTENSE_GREEN = (0.0, 1.0, 0.0, 1.0)
        WHITE = (1.0, 1.0, 1.0, 1.0)
        VERY_FAINT_GREEN = (0.0, 0.1, 0.0, 1.0)
        VIVID_GREEN = (0.0, 0.9, 0.0, 1.0)

        if logit == 1.0:
            expected_color = INTENSE_GREEN
        elif logit == 0.0:
            expected_color = WHITE
        elif logit == 0.000001:
            expected_color = VERY_FAINT_GREEN
        elif logit == 0.1:
            expected_color = VIVID_GREEN
        else:
            expected_color = logit_to_color(logit)
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
