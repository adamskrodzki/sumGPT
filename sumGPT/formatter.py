import random
import numpy as np
import math


class Formatter:
    def __init__(self, tokenizer, zero='_'):
        self.tokenizer = tokenizer
        self.zero = zero

    def format(self, length : int, source : str, generate_targets = True):

        if(generate_targets == False):
            end_of_data = source.index("=")
            x=source[:end_of_data+1]
            return self.pad_right(x, length), []

        last_cleared_idx = source.find('=')
        if last_cleared_idx >= 0:
            y=''.join([self.zero]*(last_cleared_idx)+list(source[last_cleared_idx+1:]))
        else:
            y=source[1:]+"_"
        x=source[:-1]
        pads_needed = length - len(x)
        return self.pad_right(x, length), self.pad_right(y, length)

    def tokenize(self, data : str):
        return self.tokenizer.encode(data)

    def pad_left(self, sequence, total_length, padding_char='_'):
        if len(sequence) > total_length:
            print(f"Suence length {len(sequence)} is greater than total length {total_length}")
            print(sequence)
            assert False, f"Sequence length {len(sequence)} is greater than total length {total_length}"
        return sequence.rjust(total_length, padding_char)
        
    def pad_right(self, sequence, total_length, padding_char='_'):
        if len(sequence) > total_length:
            print(f"Suence length {len(sequence)} is greater than total length {total_length}")
            print(sequence)
            assert False, f"Sequence length {len(sequence)} is greater than total length {total_length}"
        return sequence.ljust(total_length, padding_char)

    def pad_random(self, sequence, total_length, pads_right, padding_char='_'):
        if len(sequence) > total_length:
            print(f"Suence length {len(sequence)} is greater than total length {total_length}")
            print(sequence)
            assert False, f"Sequence length {len(sequence)} is greater than total length {total_length}"
        
        return self.pad_right(self.pad_left(sequence, total_length-pads_right), total_length)

    def add_right(self, sequence, counter, character):
        return sequence+character*counter
