import os, sys
import numpy as np
import torch
from torch.nn import functional as F
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'common')))

from character_tokenizer import CharacterTokenizer
from formatter import Formatter 
from generators.random_sums import RandomSums
from generators.fixed_sums import FixedSums
from utils import GenerationTools, VisualisationTools

from model import GPT, GPTConfig

def is_equal(first, second):
    # Replace all underscores with spaces
    first = first.replace('_', ' ')
    second = second.replace('_', ' ')
    
    # Remove all whitespace
    first = ''.join(first.split())
    second = ''.join(second.split())
    
    # Compare the two strings
    return first == second

device = 'cpu'

if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
    
seed = 4355

CHAR_VOCAB = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '=', '\n', '_', 'X', 'Y']
tokenizer = CharacterTokenizer(CHAR_VOCAB)
formatter = Formatter(tokenizer)

model = GPT(GPTConfig(vocab_size=len(CHAR_VOCAB), n_embd=512, n_layer=3, n_head=8))
step = model.load(41, torch.device(device))
print(step)
sample_rng = torch.Generator(device=device)
sample_rng.manual_seed(seed)

B = 128
T = 32

gen = RandomSums(12)
gen2 = FixedSums(8,7)

visualizer = VisualisationTools()

t = GenerationTools(device, visualizer)

visualizer.set_status(False)

total_count = 0
total_examples = 0
correctness_records = []

for i in range(1,10):
    for j in range(1,10):
        correct_count = 0
        gen2 = FixedSums(i,j)
        if i==8 and j>=5 and visualizer.get_count()<50:
            visualizer.set_status(True)
        else:
            visualizer.set_status(False)
        examples = t.generate_examples(gen2, B)
        queries = [example.split('=')[0] + '=' for example in examples]
        answers, probabilities = t.generate_answers(model, formatter, B, T, queries)
        total_examples+=len(answers)
        for k in range(0,len(answers)):
            if is_equal(examples[k], answers[k])==True:
                correct_count+=1
        
        correctness = float(100 * correct_count) / B
        correctness_records.append(correctness)
        print(f"FixedSums({i},{j}) correctness {correctness} %")
        total_count+=correct_count

average_correctness = float(100 * total_count) / total_examples
print(f"Total correctness = {average_correctness} %")

# Calculate weights based on correctness
weights = []
for correctness in correctness_records:
    weight = 5 + 4 * (average_correctness - correctness) / (100 - average_correctness)
    weight = weight ** 0.2
    weight = 1 + (weight - 1) * 20
    weights.append(weight)

print(f"Average correctness: {average_correctness} %")
print(f"Weights: {weights}")

visualizer.save_data("visualisation2.txt")

examples = t.generate_examples(gen2, B)

queries = [example.split('=')[0] + '=' for example in examples]

answers, probabilities = t.generate_answers(model, formatter, B, T, queries)
