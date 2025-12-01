# 🎯 Transformer Concepts - Hands-On Exercises

**Goal:** Implement key transformer components from scratch to deeply understand how they work.

---

## Exercise 1: Implement Scaled Dot-Product Attention

### 📝 Theory Recap
```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
```

### 💻 Your Task

```python
import numpy as np

def scaled_dot_product_attention(Q, K, V):
    """
    Implement scaled dot-product attention from scratch.
    
    Args:
        Q: Query matrix (seq_len, d_k)
        K: Key matrix (seq_len, d_k)
        V: Value matrix (seq_len, d_v)
    
    Returns:
        output: Attended values (seq_len, d_v)
        attention_weights: Attention weights (seq_len, seq_len)
    """
    # TODO: Implement this function
    # Step 1: Compute Q·K^T
    # Step 2: Scale by √d_k
    # Step 3: Apply softmax
    # Step 4: Multiply by V
    
    pass

# Test your implementation
if __name__ == "__main__":
    np.random.seed(42)
    
    seq_len, d_k, d_v = 4, 8, 8
    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    V = np.random.randn(seq_len, d_v)
    
    output, weights = scaled_dot_product_attention(Q, K, V)
    
    print("Attention weights shape:", weights.shape)  # Should be (4, 4)
    print("Output shape:", output.shape)  # Should be (4, 8)
    print("\nAttention weights (each row sums to 1):")
    print(weights)
    print("Row sums:", weights.sum(axis=1))
```

### ✅ Solution

<details>
<summary>Click to reveal solution</summary>

```python
def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]
    
    # Step 1: Compute Q·K^T
    scores = np.matmul(Q, K.T)  # (seq_len, seq_len)
    
    # Step 2: Scale by √d_k
    scores = scores / np.sqrt(d_k)
    
    # Step 3: Apply softmax
    attention_weights = softmax(scores, axis=-1)
    
    # Step 4: Multiply by V
    output = np.matmul(attention_weights, V)
    
    return output, attention_weights

def softmax(x, axis=-1):
    # Numerical stability: subtract max
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
```

**Key Insights:**
- **Scaling prevents saturation**: Without division by √d_k, softmax gets very sharp
- **Attention weights sum to 1**: Each row is a probability distribution
- **Output combines all values**: Weighted average based on relevance

</details>

---

## Exercise 2: Add Causal Masking (GPT-style)

### 📝 Theory
In decoder-only models (GPT), each position can only attend to previous positions.

```
Mask matrix (lower triangular):
     0   1   2   3
0 [ 1   0   0   0 ]  Position 0 sees only itself
1 [ 1   1   0   0 ]  Position 1 sees 0, 1
2 [ 1   1   1   0 ]  Position 2 sees 0, 1, 2
3 [ 1   1   1   1 ]  Position 3 sees all
```

### 💻 Your Task

```python
def masked_attention(Q, K, V, mask=None):
    """
    Implement attention with optional masking.
    
    Args:
        Q, K, V: Same as before
        mask: Boolean mask (seq_len, seq_len) where True = keep, False = mask
    
    Returns:
        output, attention_weights
    """
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.T) / np.sqrt(d_k)
    
    # TODO: Apply mask
    # Hint: Set masked positions to -inf before softmax
    # This makes them become 0 after softmax
    
    attention_weights = softmax(scores, axis=-1)
    output = np.matmul(attention_weights, V)
    
    return output, attention_weights

def create_causal_mask(seq_len):
    """
    Create lower triangular mask for causal attention.
    """
    # TODO: Implement this
    pass

# Test
mask = create_causal_mask(4)
output, weights = masked_attention(Q, K, V, mask)
print("Masked attention weights:")
print(weights)
```

### ✅ Solution

<details>
<summary>Click to reveal solution</summary>

```python
def masked_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.T) / np.sqrt(d_k)
    
    if mask is not None:
        # Set masked positions to -inf
        scores = np.where(mask, scores, -1e9)
    
    attention_weights = softmax(scores, axis=-1)
    output = np.matmul(attention_weights, V)
    
    return output, attention_weights

def create_causal_mask(seq_len):
    # Lower triangular matrix (including diagonal)
    mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))
    return mask
```

**Example output:**
```
Causal mask:
[[ True False False False]
 [ True  True False False]
 [ True  True  True False]
 [ True  True  True  True]]

Attention weights (upper triangle is zero):
[[1.00 0.00 0.00 0.00]
 [0.45 0.55 0.00 0.00]
 [0.28 0.35 0.37 0.00]
 [0.22 0.29 0.27 0.22]]
```

</details>

---

## Exercise 3: Implement Multi-Head Attention

### 📝 Theory
Instead of one attention, run h parallel attention "heads" with different learned projections.

```
MultiHead(Q,K,V) = Concat(head1, head2, ..., headh) · W_O

where head_i = Attention(Q·W_Q^i, K·W_K^i, V·W_V^i)
```

### 💻 Your Task

```python
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        """
        d_model: Model dimension (e.g., 512)
        num_heads: Number of attention heads (e.g., 8)
        """
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 64 if d_model=512, heads=8
        
        # TODO: Initialize weight matrices
        # W_Q, W_K, W_V: (d_model, d_model)
        # W_O: (d_model, d_model)
        
    def split_heads(self, x):
        """
        Split last dimension into (num_heads, d_k).
        
        Input:  (batch, seq_len, d_model)
        Output: (batch, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, _ = x.shape
        # TODO: Implement reshape and transpose
        pass
    
    def forward(self, Q, K, V, mask=None):
        """
        Multi-head attention forward pass.
        """
        # TODO: 
        # 1. Linear projections: Q, K, V = X @ W_Q, X @ W_K, X @ W_V
        # 2. Split heads
        # 3. Scaled dot-product attention for each head
        # 4. Concatenate heads
        # 5. Final linear projection
        pass

# Test
mha = MultiHeadAttention(d_model=512, num_heads=8)
X = np.random.randn(1, 10, 512)  # (batch, seq_len, d_model)
output = mha.forward(X, X, X)
print("Output shape:", output.shape)  # Should be (1, 10, 512)
```

### ✅ Solution

<details>
<summary>Click to reveal solution</summary>

```python
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Initialize weights (in practice, these are learned)
        self.W_Q = np.random.randn(d_model, d_model) * 0.01
        self.W_K = np.random.randn(d_model, d_model) * 0.01
        self.W_V = np.random.randn(d_model, d_model) * 0.01
        self.W_O = np.random.randn(d_model, d_model) * 0.01
    
    def split_heads(self, x):
        batch_size, seq_len, _ = x.shape
        # Reshape: (batch, seq_len, d_model) -> (batch, seq_len, num_heads, d_k)
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        # Transpose: (batch, num_heads, seq_len, d_k)
        return x.transpose(0, 2, 1, 3)
    
    def forward(self, Q, K, V, mask=None):
        batch_size, seq_len, _ = Q.shape
        
        # 1. Linear projections
        Q_proj = np.matmul(Q, self.W_Q)  # (batch, seq_len, d_model)
        K_proj = np.matmul(K, self.W_K)
        V_proj = np.matmul(V, self.W_V)
        
        # 2. Split heads
        Q_heads = self.split_heads(Q_proj)  # (batch, heads, seq_len, d_k)
        K_heads = self.split_heads(K_proj)
        V_heads = self.split_heads(V_proj)
        
        # 3. Attention for each head
        d_k = self.d_k
        scores = np.matmul(Q_heads, K_heads.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
        
        if mask is not None:
            scores = np.where(mask, scores, -1e9)
        
        attn_weights = softmax(scores, axis=-1)
        attn_output = np.matmul(attn_weights, V_heads)
        
        # 4. Concatenate heads
        attn_output = attn_output.transpose(0, 2, 1, 3)  # (batch, seq_len, heads, d_k)
        concat = attn_output.reshape(batch_size, seq_len, self.d_model)
        
        # 5. Final projection
        output = np.matmul(concat, self.W_O)
        
        return output
```

**Why multi-head?**
- Each head can learn different patterns (syntax, semantics, etc.)
- More expressive than single attention
- Parallel computation efficiency

</details>

---

## Exercise 4: Positional Encoding

### 📝 Theory
Transformers have no inherent position information. We add positional encodings:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### 💻 Your Task

```python
def positional_encoding(seq_len, d_model):
    """
    Generate sinusoidal positional encoding.
    
    Args:
        seq_len: Sequence length
        d_model: Model dimension
    
    Returns:
        PE matrix of shape (seq_len, d_model)
    """
    PE = np.zeros((seq_len, d_model))
    
    # TODO: Implement the formula above
    # Hint: Use np.sin and np.cos
    
    return PE

# Test
pe = positional_encoding(seq_len=100, d_model=512)
print("PE shape:", pe.shape)

# Visualize (if you have matplotlib)
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))
plt.imshow(pe[:50, :], cmap='RdBu', aspect='auto')
plt.xlabel('Dimension')
plt.ylabel('Position')
plt.title('Positional Encoding Heatmap')
plt.colorbar()
plt.savefig('positional_encoding.png')
print("Saved visualization to positional_encoding.png")
```

### ✅ Solution

<details>
<summary>Click to reveal solution</summary>

```python
def positional_encoding(seq_len, d_model):
    PE = np.zeros((seq_len, d_model))
    
    position = np.arange(seq_len).reshape(-1, 1)  # (seq_len, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    # Even dimensions: sine
    PE[:, 0::2] = np.sin(position * div_term)
    
    # Odd dimensions: cosine
    PE[:, 1::2] = np.cos(position * div_term)
    
    return PE
```

**Key properties:**
- **Different frequencies**: Early dimensions change faster than later ones
- **Relative positions**: PE(pos+k) can be expressed as linear function of PE(pos)
- **Unbounded**: Works for any sequence length (unlike learned positions)

</details>

---

## Exercise 5: Complete Transformer Block

### 💻 Your Task

Now combine everything into a full transformer encoder block:

```python
class TransformerEncoderBlock:
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension (usually 4 * d_model)
        """
        self.attention = MultiHeadAttention(d_model, num_heads)
        
        # TODO: Initialize FFN weights
        # W1: (d_model, d_ff)
        # W2: (d_ff, d_model)
        
        # TODO: Initialize layer norm parameters
        
    def feed_forward(self, x):
        """
        FFN(x) = ReLU(x·W1 + b1)·W2 + b2
        """
        # TODO: Implement
        pass
    
    def layer_norm(self, x):
        """
        LayerNorm(x) = γ * (x - μ) / σ + β
        """
        # TODO: Implement
        pass
    
    def forward(self, x, mask=None):
        """
        Full encoder block:
        1. Multi-head attention + residual + norm
        2. Feed-forward + residual + norm
        """
        # TODO: Implement
        # x = LayerNorm(x + Attention(x))
        # x = LayerNorm(x + FFN(x))
        pass
```

---

## 🎓 Challenge Exercises

### Challenge 1: RoPE (Rotary Position Embedding)
Implement RoPE used in LLaMA, which is more effective than sinusoidal:

```python
def apply_rotary_emb(q, k, position):
    """
    Apply rotary position embedding to q and k.
    
    RoPE rotates q and k by position-dependent angles.
    """
    # TODO: Research and implement RoPE
    pass
```

### Challenge 2: Grouped Query Attention (GQA)
Implement GQA from LLaMA 2/3:

```python
class GroupedQueryAttention:
    def __init__(self, d_model, num_q_heads, num_kv_heads):
        """
        num_q_heads: Number of query heads (e.g., 32)
        num_kv_heads: Number of key/value heads (e.g., 8)
        
        Each KV head is shared by num_q_heads / num_kv_heads query heads
        """
        # TODO: Implement GQA
        pass
```

### Challenge 3: Flash Attention
Research Flash Attention optimization and explain:
- Why is standard attention memory-intensive?
- How does Flash Attention reduce memory usage?
- What is "tiling" in this context?

---

## 📊 Self-Assessment

After completing these exercises, you should be able to:

- [ ] Explain the formula Attention(Q,K,V) = softmax(QK^T/√d_k)V in your own words
- [ ] Implement scaled dot-product attention from scratch
- [ ] Understand why we scale by √d_k
- [ ] Explain the difference between causal and bidirectional masking
- [ ] Implement multi-head attention
- [ ] Explain why we use multiple attention heads
- [ ] Generate and visualize positional encodings
- [ ] Combine all components into a transformer block

---

## 🔗 Next Steps

1. **Implement in PyTorch**: Redo all exercises using PyTorch tensors
2. **Train a model**: Build a small transformer and train on simple task
3. **Read papers**: 
   - "Attention is All You Need" (Vaswani et al., 2017)
   - "RoFormer: Enhanced Transformer with RoPE" (Su et al., 2021)
   - "Flash Attention" (Dao et al., 2022)
4. **Explore architectures**: Study BERT, GPT, and T5 implementations

**Remember:** Implementation is understanding! 🚀
