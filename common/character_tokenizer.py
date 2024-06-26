class CharacterTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.char_to_idx = {ch: idx for idx, ch in enumerate(vocab)}
        self.idx_to_char = {idx: ch for idx, ch in enumerate(vocab)}

    def encode(self, text):
        return [self.char_to_idx[ch] for ch in text if ch in self.char_to_idx]

    def decode(self, tokens):
        return ''.join([self.idx_to_char[idx] for idx in tokens])

    def get_vocab_size(self):
        return self.vocab_size
