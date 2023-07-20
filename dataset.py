import torch
import random
from torch.utils.data import Dataset

class PerceiverARDataset(Dataset):
    def __init__(self, data, block_size, latent_size):
        self.MASK_CHAR = u"\u2047"  # the doublequestionmark character, for mask
        self.PAD_CHAR = u"\u25A1"  # the empty square character, for pad
        self.EOS_CHAR = u"\u25C9"  # the fisheye character, for end of sentence
        self.SOS_CHAR = u"\u25CE"  # the bullseye character, for start of sentence

        chars = list(sorted(list(set(data))))
        assert self.MASK_CHAR not in chars
        assert self.PAD_CHAR not in chars
        assert self.EOS_CHAR not in chars
        assert self.SOS_CHAR not in chars
        chars.insert(0, self.MASK_CHAR)
        chars.insert(0, self.EOS_CHAR)
        chars.insert(0, self.SOS_CHAR)
        chars.insert(0, self.PAD_CHAR)

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

        data_size, vocab_size = len(data), len(chars)
        print("data has %d characters, %d unique." % (data_size, vocab_size))

        self.block_size = block_size
        self.latent_size = latent_size
        self.vocab_size = vocab_size
        self.data = data.split("\n")

    def __len__(self):
        # returns the length of the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # TODO [part e]: see spec above
        s = self.data[idx]
        s = self.SOS_CHAR + s + self.EOS_CHAR
        s = s[:self.block_size]
        length = len(s)

        # Randomly select a start index
        start_index = random.randint(1, max(1, length - self.latent_size))

        # Keep all text before start index, insert mask and then keep LATENT_SIZE chars afterwards
        s = s[:start_index+self.latent_size]

        # Extract x and y from s
        x = s[:-1]
        y = s[1:]

        # Calculate padding and trimming lengths
        x_padding_len = max(0, self.block_size - len(x))
        y_padding_len = max(0, self.latent_size - len(y))
        y_trimming_len = max(0, len(y) - self.latent_size)

        # Perform padding and trimming
        x = self.PAD_CHAR * x_padding_len + x
        y = y[y_trimming_len:] + self.PAD_CHAR * y_padding_len

        x = torch.tensor([self.stoi[c] for c in x], dtype=torch.long)
        y = torch.tensor([self.stoi[c] for c in y], dtype=torch.long)
        return x, y