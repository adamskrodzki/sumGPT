import os,sys
import numpy as np
import torch
from torch.nn import functional as F
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'common')))

from character_tokenizer import CharacterTokenizer
from formatter import Formatter 
from generators.fixed_sums import FixedSums
from utils import GenerationTools

from model import GPT, GPTConfig

device = 'cpu'


if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
    
seed = 233




CHAR_VOCAB = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '=', '\n', '_', 'X', 'Y']
tokenizer = CharacterTokenizer(CHAR_VOCAB)
formatter = Formatter(tokenizer)


model = GPT(GPTConfig(vocab_size=len(CHAR_VOCAB), n_embd=64))
model.load(66, torch.device(device))
sample_rng = torch.Generator(device=device)
sample_rng.manual_seed(seed)

B = 16
T = 16

gen = FixedSums(3,5)

t = GenerationTools(device)

examples = t.generate_examples(gen, B)

queries = [example.split('=')[0] + '=' for example in examples]

answers, probabilities = t.generate_answers(model, formatter, B, T, queries)

print(answers)
print(probabilities)