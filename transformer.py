import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
dim_embedding = 32

torch.manual_seed(42)

# tiny shakespeare dataset
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# read data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    
    
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s] # encoder: from string, output a list of integers
decode = lambda l: "".join([itos[i] for i in l]) # decoder from list of integers, output a string

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]



class Head(nn.Module):
    """one self attention head"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(dim_embedding, head_size, bias=False) # false means matrix multiply with out bias
        self.query = nn.Linear(dim_embedding, head_size, bias=False)
        self.value = nn.Linear(dim_embedding, head_size, bias = False)
        self.head_size = head_size
        
        # for later use
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size))) # not a model parameter

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, C of key, query dimension)
        q = self.query(x) # (B, T, C of key, query dimension)

        # computing attention score
        # # keep batch dimension in mind, only transpose last two dim for each batch dimension
        # then normalize by key, query dimension to not have too large dot products
        weights = q @ k.transpose(-2, -1) * self.head_size**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)

        # we dont want future tokens influence current ones -> set upper triangle -inf
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)
        
        # perform weighted aggregation of key, query dot product with value map
        v = self.value(x) # (B, T, C)
        out = weights @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of attention working in parallel"""
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(dim_embedding, dim_embedding) # projection layer for residual conenction
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    """a simple linear layer with a ReLU activation"""
    
    def __init__(self, dim_embedding):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_embedding, 4 * dim_embedding), # 4x channel size for innder dimension as AAYN paper suggests
            nn.ReLU(),
            nn.Linear(4 * dim_embedding, dim_embedding),  
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block"""
    
    def __init__(self, dim_embedding, n_head):
        # dim_embedding: embedding dimension, n_head: number of attention heads
        super().__init__()
        head_size = dim_embedding // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(dim_embedding)
        self.ln1 = nn.LayerNorm(dim_embedding) # unit mean and gaussian at initliazation for features
        self.ln2 = nn.LayerNorm(dim_embedding) # layer norm for self attention

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # x + is residual connection forking off
        x = x + self.ffwd(self.ln2(x)) # added layer norm 
        return x

# data loading
def get_batch(split):
    # generate small batch of data of inputs x and targets y from 
    # either train or val data

    data = train_data if split == "train" else val_data

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    # if we have cuda, move data to it
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad() # tell pytorch that backward is not called on this
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# simple bigram model
class BigramLanguageModel(nn.Module):

  def __init__(self):
    super().__init__()

    # each token reads off the logits for the next token from lookup table
    self.token_embedding_table = nn.Embedding(vocab_size, dim_embedding)
    self.position_embedding_table = nn.Embedding(block_size, dim_embedding)
    
    self.blocks = nn.Sequential(
        Block(dim_embedding, n_head=4),
        Block(dim_embedding, n_head=4),
        Block(dim_embedding, n_head=4),
        nn.LayerNorm(dim_embedding)
    )
    #self.sa_heads = MultiHeadAttention(4, dim_embedding//4) # 4 heads of self attention, each with embedding dimension of 8 (32//4)
    #self.ffwd = FeedForward(dim_embedding) # feed forward layer
    self.lm_head = nn.Linear(dim_embedding, vocab_size) # languange modeling head

  def forward(self, idx, targets=None):

    
    B, T = idx.shape
    
    # idx and targets are both (B,T) tesnor of integers
    token_embedding = self.token_embedding_table(idx) # (B,T,C) batch x time x channel (4x8x65)
    position_embedding = self.position_embedding_table(torch.arange(T, device = device))
    x = token_embedding + position_embedding # (B, T, C)
    x = self.blocks(x) # (B, T, C) after transformer blocks
    
    #x = self.sa_heads(x) # apply one head of attnetion (B, T, C)
    #x = self.ffwd(x) # apply feed forward layer (B, T, C)
    
    logits = self.lm_head(x) # (B, T, vocab_size)

    if targets is None:
      loss = None
    
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets) # negative log likelihood

    return logits, loss

  def generate(self, idx, max_new_tokens):

    # idx is (B, T) array of indices in the current context

    for _ in range(max_new_tokens):


        # crop idx to the last block_size tokens
        # otherwise position table will run out of scope
        idx_cond = idx[:, -block_size:]
        
        # get prediction 
        logits, loss = self(idx_cond)

        # only focus on last time step
        logits = logits[:, -1, :] # becomes (B, C)
        # apply softmax
        probs = F.softmax(logits, dim=-1) # (B, 1)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples = 1) # (B, 1) each batch dimension has one prediciont
        # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    return idx

model = BigramLanguageModel()

# move model parameters to gpu
m = model.to(device)

# create pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train losss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
    xb, yb = get_batch("train")

    # evaluate loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))