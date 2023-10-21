import torch
import torch.nn as nn
from torch.nn import functional as F
from block import Block
from hyperparameters import HyperParameters

class GPTModel(nn.Module):
    def __init__(self, hyperparameters: HyperParameters):
        super().__init__()
        # Each hp number will go to a specific row and will read off the logits (scores) for the next token from a lookup table
        self.hp = hyperparameters
        self.token_embedding_table = nn.Embedding(self.hp.vocab_size, self.hp.n_embd)
        self.position_embedding_table = nn.Embedding(self.hp.block_size, self.hp.n_embd)
        self.blocks = nn.Sequential(*[
            Block(
                n_embd=self.hp.n_embd, 
                n_head=self.hp.n_head,
                block_size=self.hp.block_size,
                dropout=self.hp.dropout,
            ) for _ in range(self.hp.n_layer)])
        self.ln_f = nn.LayerNorm(self.hp.n_embd)
        self.lm_head = nn.Linear(self.hp.n_embd, self.hp.vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        # we take the index and pass them into the embedding table
        # we arrange this into a batch (4) by time (8) by channel (vocab_size, 65) tensor
        token_embeddings = self.token_embedding_table(idx) # (B, T, C)
        position_embeddings = self.position_embedding_table(torch.arange(T, device=self.hp.device)) # (T, C)
        x = token_embeddings + position_embeddings # (B, T, C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)

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
            idx_cond = idx[:, -self.hp.block_size:]
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

    