import torch

class HyperParameters():
    def __init__(self):
        self.batch_size = 32 # How many sequences do we want to process in parallel
        self.block_size = 64 # How long in characters should these characters be?
        self.max_iters = 500 # How many iterations should we train for?
        self.eval_interval = 500
        self.learning_rate = 3e-4
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' # If you have a gpu, this code makes it run on the gpu.
        self.eval_iters = 200
        self.n_embd = 384
        self.n_head = 6
        self.n_layer = 8
        self.dropout = 0.2
        self.torch_seed = 1337
        self.vocab_size = 0