import random
class ProbabilityProvider:
    def __init__(self, decrement_percentage, num_sets):
        self.decrement_percentage = decrement_percentage / 100.0  # Convert percentage to a decimal
        self.probabilities = [1.0]  # Start with a probability of 1 for the first set
        for i in range(1, num_sets):
            self.probabilities.append(self.probabilities[-1] * (1 - self.decrement_percentage))

    def set_probabilities(self, new_probabilities):
        if len(new_probabilities) != len(self.probabilities):
            raise ValueError("New probabilities list must have the same length as the current probabilities list.")
        self.probabilities = new_probabilities

    def get_probabilities(self):
        return self.probabilities

    def make_harder(self):
        non_zero_index = next((i for i, prob in enumerate(reversed(self.probabilities)) if prob != 0), None)
        if non_zero_index is not None:
            if non_zero_index > 0:
                self.probabilities[non_zero_index-1] = 0.5
            
        if non_zero_index > 1:
            fraction = 0.5/(non_zero_index - 1)
            self.probabilities[:non_zero_index-1]=fraction

    def sample(self):
        if all(prob == 0 for prob in self.probabilities):
            raise ValueError("All probabilities are zero, cannot sample.")
        return random.choices(range(len(self.probabilities)), weights=self.probabilities, k=1)[0]