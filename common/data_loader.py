import torch
import numpy as np
import random

class Set:
    samples = []

    def __init__(self, bufor_size : int, generator):
        self.bufor_size = bufor_size
        self.samples = [None]*bufor_size
        self.generator = generator
        for i in range(bufor_size):
            self.samples[i] = generator.sample()

    def sample(self):
        pos = random.randint(0,self.bufor_size-1)
        return self.samples[pos]

    def reset(self):
        for i in range(self.bufor_size):
            self.samples[i] = self.generator.sample()

class DataLoader:
    def __init__(self, B : int, T : int, probability_provider, formatter, sets : list):
        self.B = int(round(B))
        self.T = T
        self.formatter = formatter
        self.sets = sets
        self.probs_provider = probability_provider

    def reset(self):
        for i in range(len(self.sets)):
            self.sets[i].reset();

    def mask_targets(self, source, target, mask_index=-100):
        mask = [mask_index]*len(target)
        eq_index = source.index(11)#TODO: do not hardcode here
        mask[eq_index:] = target[eq_index:]
        #mask = [12 if element == 13 else element for element in mask]
        return mask
        
    def next_batch(self):
        B, T = self.B, self.T
        x_batch = []
        y_batch = []
        source_examples = []

        while len(x_batch) < B:
            idx = self.probs_provider.sample()
            example = self.sets[idx].sample()
            source_examples.append(example)

            x_str, y_str = self.formatter.format(self.T, example)
            #print(x_str)
            #print(y_str)
            x = self.formatter.tokenize(x_str)
            y = self.formatter.tokenize(y_str)
            y = self.mask_targets(x,y)
            x_batch.append(x)
            y_batch.append(y)

        x_batch = torch.tensor(np.array(x_batch[:B]), dtype=torch.long)
        y_batch = torch.tensor(np.array(y_batch[:B]), dtype=torch.long)
        return x_batch, y_batch, source_examples