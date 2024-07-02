import random

class ProbabilityProvider:
    def __init__(self, decrement_percentage, num_sets):
        self.decrement_percentage = decrement_percentage / 100.0
        self.probabilities = [1.0] + [
            1.0 * (1 - self.decrement_percentage) ** i for i in range(1, num_sets)
        ]

    def set_probabilities(self, new_probabilities):
        if len(new_probabilities) != len(self.probabilities):
            raise ValueError(f"New probabilities list must have the same length as the current probabilities list. {len(new_probabilities)}!={len(self.probabilities)}")
        self.probabilities = new_probabilities

    def get_probabilities(self):
        return self.probabilities

    def make_harder(self):
        first_zero_index = next((i for i, prob in enumerate(self.probabilities) if prob == 0), None)
        if first_zero_index is None:
            raise ValueError("No zero element found in probabilities.")

        self.probabilities[first_zero_index] = 1.0

        if first_zero_index > 0:
            self.probabilities[first_zero_index - 1] = 0.5

        if first_zero_index > 1:
            fraction = 0.5 / (first_zero_index - 1)
            self.probabilities = [
                fraction if i < first_zero_index - 1 else prob 
                for i, prob in enumerate(self.probabilities)
            ]

        print(self.probabilities)

    def sample(self):
        if all(prob == 0 for prob in self.probabilities):
            raise ValueError("All probabilities are zero, cannot sample.")
        return random.choices(range(len(self.probabilities)), weights=self.probabilities, k=1)[0]
