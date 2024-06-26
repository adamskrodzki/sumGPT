import random;
class FixedSums:
    def __init__(self, left, right):
        self.min_right = 10**(right-1)
        self.min_left = 10**(left-1)
        self.max_left = 10**(left)-1
        self.max_right = 10**(right)-1
        if self.min_right == 1:
            self.min_right = 0
        if self.min_right == 1:
            self.min_right = 0

    def sample(self):
        left_arg = random.randint(self.min_left, self.max_left)
        right_arg = random.randint(self.min_right, self.max_right)
        return f"{left_arg}+{right_arg}={left_arg+right_arg}\n"
