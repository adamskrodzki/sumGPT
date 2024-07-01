import random

class FixedSums10:
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
        
        # Determine which argument is shorter
        if len(str(left_arg)) < len(str(right_arg)):
            shorter, longer = left_arg, right_arg
        else:
            shorter, longer = right_arg, left_arg
        
        # If shorter is 0, skip the loop and multiplying
        if shorter == 0:
            return f"{left_arg}+{right_arg}={left_arg+right_arg}\n"
        
        # Determine the range of legal powers of 10
        min_power = 1
        max_power = 1
        while (shorter * max_power) <= longer:
            max_power *= 10
        
        # Select a random power of 10 within the legal range
        power = 10 ** random.randint(0, len(str(max_power)) - 1)
        
        # Multiply the shorter argument by the chosen power of 10
        if len(str(left_arg)) < len(str(right_arg)):
            left_arg *= power
        else:
            right_arg *= power
        
        return f"{left_arg}+{right_arg}={left_arg+right_arg}\n"
