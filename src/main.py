import torch
from nanotokens import NanoTokens
from gptmodel import GPTModel
from hyperparameters import HyperParameters

# Let's set our hyper parameters
hp = HyperParameters()
torch.manual_seed = hp.torch_seed

# open the text file and read it
with open('../datasets/starwars.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# using the NanoTokens class, let's get our tokens, and encode our text
nanotokens = NanoTokens(token_method='2-character', text=text)
hp.vocab_size = nanotokens.vocab_size

# We are going to now encode the entire text dataset, and turn it into a torch.Tensor
# A torch.Tensor is a multi-dimensional matrix containing elements of a single data type.
data = torch.tensor(nanotokens.encode(text), dtype = torch.long)

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
    ix = torch.randint(len(data) - hp.block_size, (hp.batch_size,))
    x = torch.stack([data[i:i+hp.block_size] for i in ix])
    y = torch.stack([data[i+1:i+hp.block_size+1] for i in ix])
    x, y = x.to(hp.device), y.to(hp.device)
    return x, y

@torch.no_grad() 
def estimate_loss(): # This function averages the loss over multiple batches
    out = {} # for both the training and validation sets
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(hp.eval_iters)
        for k in range(hp.eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



model = GPTModel(hyperparameters=hp)
m = model.to(hp.device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=hp.learning_rate)

for iter in range(hp.max_iters):
    # evaluate the loss on train and validation sets every once in a while
    if iter % hp.eval_interval == 0 or iter == hp.max_iters - 1:
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
context = torch.zeros((1, 1), dtype=torch.long, device=hp.device)
print(nanotokens.decode(m.generate(context, max_new_tokens=2000)[0].tolist()))