import torch
import torch.nn as nn
from torch.nn import functional as F

# Let's set our hyper parameters
batch_size = 32 # How many sequences do we want to process in parallel
block_size = 8 # How long in characters should these characters be?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

torch.manual_seed(1337)
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# let's open the text file 
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#region encoding and decoding
chars = sorted(list(set(text)))
vocab_size = len(chars)
# We are now goign to create a mapping from integers to characters
chars = sorted(list(set(text))) # the list of every unique character in the text
stoi = { character:index for index,character in enumerate(chars) } # Converts a string to integer
itos = { index:character for index,character in enumerate(chars) } # Converts an integer to string
encode = lambda string: [stoi[character] for character in string]  # Convert a string to a list of integers
decode = lambda list: ''.join([itos[index] for index in list])  # Convert a list of integers to a string

# Using characters as our tokens isn't a bad idea, but it might be smarter to consider 3 characters at a time etc
# This is because we could represent our sentence in a much smaller list, just with integers up to 5000 instead of 65 for example
# There's a obviously an optimal combination, but to keep things simple we're going to use a character tokenizer
#endregion

