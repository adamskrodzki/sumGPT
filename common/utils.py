import os,sys
import numpy as np
import torch
import re
from torch.nn import functional as F
from collections import Counter

class GenerationTools:
    def __init__(self, device):
        self.device = device

    @staticmethod
    def print_probs(probs):
        probs = probs.to(dtype=torch.float32)
        # Convert to numpy array
        probs_np = probs.numpy()
        # Apply the rounding rule
        probs_np[probs_np < 0.001] = 0
        # Round to three decimal places for readability
        probs_rounded = np.round(probs_np, 3)
        print(probs_rounded)

    @staticmethod
    def generate_examples(generator, count):
        examples = []
        for i in range(0,count):
            examples.append(generator.sample())
        return examples

    @staticmethod
    def tokenize(formatter, samples, context_length):
        batch = [formatter.tokenize(formatter.format(context_length, sample, False)[0]) for sample in samples]
        return torch.tensor(batch, dtype=torch.long)
        
    @staticmethod
    def count_score(examples, answers):
        def remove_whitespaces(s):
            return re.sub(r'\s_+', '', s)
        
        # Clean and count occurrences in examples and answers
        cleaned_examples = [remove_whitespaces(example) for example in examples]
        cleaned_answers = [remove_whitespaces(answer) for answer in answers]

        print(cleaned_answers[:5])
        print(cleaned_examples[:5])

        example_counter = Counter(cleaned_examples)
        answer_counter = Counter(cleaned_answers)
        
        match_count = 0
        miss_count = 0
        
        # Calculate matches and misses
        for example in example_counter:
            if example in answer_counter:
                match_count += min(example_counter[example], answer_counter[example])
            else:
                miss_count += example_counter[example]
        
        # Any remaining answers that didn't match examples are counted as misses
        for answer in answer_counter:
            if answer not in example_counter:
                miss_count += answer_counter[answer]
        
        return match_count, miss_count

    def average(probs):
        geometric_average = np.exp(np.mean(np.log(probabilities)))
        return geometric_average

    def generate_answers(self, model, formatter, B, T, queries):

        def get_logits_for_position(logits, queries, generation_count):
            # Create an empty list to store the selected logits for each batch element
            selected_logits = []
            for i, query in enumerate(queries):
                query_len = len(query)-1
                pos = query_len + generation_count
                selected_logits.append(logits[i, pos, :])
            return torch.stack(selected_logits)

        def update_at_index(xgen, queries, generation_count, topk_indices):
            new_xgen = []
            new_queries = []
            new_results = []
            new_compound_certainties = []
            zero = formatter.zero  # padding character
            zero_token_id = formatter.tokenizer.encode(zero)[0]  # Encode zero character to its token ID

            for i in range(xgen.size(0)):
                query_len = len(queries[i])
                pos = query_len + generation_count

                new_sequence = xgen[i].clone()  # Clone the current sequence to avoid in-place modification
                new_sequence[pos] = topk_indices[i].item()

                if topk_indices[i].item() > 10:
                    new_results.append((new_sequence.tolist(), compound_certainties[i]))
                else:
                    new_xgen.append(new_sequence.unsqueeze(0))
                    new_queries.append(queries[i])
                    new_compound_certainties.append(compound_certainties[i])

            if len(new_xgen) > 0:
                xgen = torch.cat(new_xgen, dim=0).to(self.device)
            else:
                xgen = torch.tensor([], device=self.device)
            
            return xgen, new_queries, new_results, new_compound_certainties

        results = []
        finished_queries = []
        compound_certainties = [1.0] * B  # Initialize compound certainties to 1
        data = GenerationTools.tokenize(formatter, queries, T)
        xgen = data.to(self.device)
        model.eval()
        generation_count=0
        with torch.no_grad():
            while xgen.size(0) > 0:  # Continue until there are no active sequences
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                    #print(xgen)
                    logits, loss = model(xgen)
                    logits = get_logits_for_position(logits, queries, generation_count)
                    probs = F.softmax(logits, dim=-1)
                    topk_probs, topk_indices = torch.topk(probs, 1, dim=-1)
                    #print(topk_indices)
                    # Update compound certainties
                    for i in range(len(compound_certainties)):
                        compound_certainties[i] *= topk_probs[i].item()
                    xgen, queries, new_results,compound_certainties  = update_at_index(xgen, queries, generation_count, topk_indices)
                    results = results + new_results
                    generation_count+=1

        formatted_results = [[formatter.tokenizer.decode(result[0]),result[1]] for result in results]
        answers = [item[0] for item in formatted_results]
        probabilities = [item[1] for item in formatted_results]
        return answers, probabilities