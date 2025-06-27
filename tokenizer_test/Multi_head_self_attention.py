# Here we are preparing our Inputs


# now one thing that i always was confused about were that 
# how these variables(X) that aint defined earlier are popping now
# and have a function attached to them, but this Machine Learning bro
# and since we'd like to compress data, so such variables exist
# that are readily defined on-the-go. with some libraries and their pre-defined 
# structure of maintaining such variables. also I like this new python 3.11.x


import torch
import torch.nn as nn

# Example Input: batch_size = 2, sequence_length = 4, embed_dimension - 8

batch_size = 2
sequence_length = 4
embed_dimension = 8

X = torch.randn(batch_size, sequence_length, embed_dimension)
print(X.shape)  # (2, 4, 8)

# i had this question first that why introduce three variables
# (batch_size, sequence_length, embed_dimension) if we are anyway
#  going to randomize them with torch.randn, but as i explored 
# turns out that torch knows how to handle 
# tensors(guys with dimensions) and we need to randomize data 
# but than randomization must have limitation, hence 2, 4, 8.  

# Step 2 : Linear Projection (Q, K, V) i.e. Query, Key, Value
# We project X into K, Q, V spaces using different(those utterly randomized weights)
# We define linear layers for Q, K, V

W_Q = nn.Linear(embed_dimension, embed_dimension, bias=False)
W_K = nn.Linear(embed_dimension, embed_dimension, bias=False)
W_V = nn.Linear(embed_dimension, embed_dimension, bias=False)
# here above we just simplified our operations, we first defined 
# what inputs W_X would take and then below fed X to it.
Q = W_Q(X)
K = W_K(X)
V = W_V(X)


# Step3: Scaled Dot-Product Attention
# i.e. Attention(Q, K, V) = 
# softmax((Q*(K ** T) / sqrt(D_k(dimension of keys)) #T is Transpose
# dimension of keys is usually embed_dim / num_heads


import torch.nn.functional as F 

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.size(-1)
    # matmul mean matrix multiplication, now this -2, -1 has an explanation
    # -2 second last dim, -1 the last dimension
    # so from (batch_size, num_heads, seqlen, head_dim) this is an e.g.
    # we go to (batch_size, num_heads, head_dim, seqlen)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    # dim=-1 mean we apply softmax across the key dimension for each query 
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output, attn_weights

# Multi_head Split, we split the embedding dimension into multiple heads

# Let head_dim = embed_dim / num_heads
# we reshape Q, K, V into Q, now Q is a matrix with 
# dimensions(batchSize, numHeads, SeqLen, HeadDim) btw 
# all real numbers

num_heads = 2
head_dim = embed_dimension // num_heads

def split_heads(x, num_heads):
    batch_size, sequence_length, embed_dimension = x.size()
    x = x.view(batch_size, sequence_length, num_heads, head_dim)
    return x.permute(0, 2, 1, 3)

Q_heads = split_heads(Q, num_heads)
K_heads = split_heads(K, num_heads)
V_heads = split_heads(V, num_heads)


# Step5: Apply attention to each head

head_outputs = []
attn_maps = []

for i in range (num_heads):
    out, attn = scaled_dot_product_attention(Q_heads[:, i], K_heads[:, i], V_heads[:, i])
    head_outputs.append(out)
    attn_maps.append(attn)

# Stack and concatenate heads
head_outputs = torch.cat(head_outputs, dim=-1) # (batch_size, seqlen, embedDim)

# Step6: Final Linear Projection
# Output = Concat(heads)W ** O , where W is a set of Real 

# numbers of matrix with dimension embedDim x embedDim

W_O = nn.Linear(embed_dimension, embed_dimension, bias=False)
output = W_O(head_outputs)

print(f"Final Multi-Head Attention Output Shape: ", output.shape)

