import torch
import torch.nn as nn
from torch.nn import functional as F

# Let's set our hyper parameters
batch_size = 64 # How many sequences do we want to process in parallel
block_size = 256 # How long in characters should these characters be?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu' # If you have a gpu, this code makes it run on the gpu.
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

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

# We are going to now encode the entire shakespeare text dataset, and turn it into a torch.Tensor
# A torch.Tensor is a multi-dimensional matrix containing elements of a single data type.
data = torch.tensor(encode(text), dtype = torch.long)
# Now let's split this data into a training and validation set
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# as we sample the chunks of text, we want to have multiple batches of chunks processing in parallel (as we're using GPUs)
# batch_size = how many independent sequences will we process in parallel
# block_size = what is the maximum context length for predictions
def get_batch(split):
    # generate a small batch of data of inputs and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad() 
def estimate_loss(): # This function averages the loss over multiple batches
    out = {} # for both the training and validation sets
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        weights = q @ k.transpose(-2, -1) * C # (B,T,C) @ (B,C,T) -> (B, T, T)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        weights = F.softmax(weights, dim=-1) # (B,T,T)
        weights = self.dropout(weights)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = weights @ v # (B,T,T) @ (B,T,C) -> (B,T,C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList( [Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non linearity """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, n_embd, n_head):
        # n_embd is the embedding dimension
        # n_head is the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Each input number will go to a specific row and will read off the logits (scores) for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        # we take the index and pass them into the embedding table
        # we arrange this into a batch (4) by time (8) by channel (vocab_size, 65) tensor
        token_embeddings = self.token_embedding_table(idx) # (B, T, C)
        position_embeddings = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = token_embeddings + position_embeddings # (B, T, C)
        x = self.blocks(x) # (B,T,C)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # reshape our logits so we can pass in the logits in the way the cross entropy function EXPECTS
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            # get the negative log likelihood loss (cross entropy loss) from our predicted next characters (logits) and our targets
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # given an array of indices idx of size B (batch_size) by T
        # we want to generate the next n tokens defined by max_new_tokens
        for _ in range(max_new_tokens):
            # crop the contex to the last block size tokens
            idx_cond = idx[:, -block_size:]
            # Get the predicted tokens
            logits, loss = self(idx_cond)
            # Get the last element in the time dimension
            logits = logits[:, -1, :]
            # Use softmax to get our probabilities (B, C)
            probs = F.softmax(logits, dim=-1)
            # get 1 sample from the distribution for each batch
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append our next sampled index into the current sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # evaluate the loss on train and validation sets every once in a while
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    #evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))