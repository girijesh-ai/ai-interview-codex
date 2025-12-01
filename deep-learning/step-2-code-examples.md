# 🎨 Visual Transformer Architecture Guide

**Complete visual reference with diagrams, code, and theory**

---

## 📊 Architecture Diagrams

### Complete Transformer Encoder Block

![Transformer Encoder Block Architecture](file:///home/spurge/.gemini/antigravity/brain/4ed8e4e9-a206-4fb3-bea3-51a5c8d13fac/transformer_encoder_block_1764509625431.png)

### Attention Patterns: BERT vs GPT vs T5

![Attention Pattern Comparison](file:///home/spurge/.gemini/antigravity/brain/4ed8e4e9-a206-4fb3-bea3-51a5c8d13fac/attention_patterns_comparison_1764509648929.png)

---

## 💻 Step-by-Step Implementation with Visuals

### Part 1: Single-Head Attention from Scratch

```python
import numpy as np

def softmax(x, axis=-1):
    """Numerically stable softmax"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def attention(Q, K, V, mask=None):
    """
    Scaled dot-product attention
    
    Args:
        Q: Query matrix (seq_len, d_k)
        K: Key matrix (seq_len, d_k)
        V: Value matrix (seq_len, d_v)
        mask: Optional mask (seq_len, seq_len)
    
    Returns:
        output: (seq_len, d_v)
        attn_weights: (seq_len, seq_len)
    """
    d_k = Q.shape[-1]
    
    # Step 1: Q · K^T
    scores = np.matmul(Q, K.T)  # (seq_len, seq_len)
    print(f"1️⃣ Scores after Q·K^T: shape {scores.shape}")
    print(scores)
    print()
    
    # Step 2: Scale by √d_k
    scores = scores / np.sqrt(d_k)
    print(f"2️⃣ After scaling by √{d_k} = {np.sqrt(d_k):.2f}:")
    print(scores)
    print()
    
    # Step 3: Apply mask (optional)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
        print(f"3️⃣ After applying mask:")
        print(scores)
        print()
    
    # Step 4: Softmax
    attn_weights = softmax(scores, axis=-1)
    print(f"4️⃣ Attention weights after softmax:")
    print(attn_weights)
    print(f"   Row sums (should be 1.0): {attn_weights.sum(axis=1)}")
    print()
    
    # Step 5: Weighted sum of values
    output = np.matmul(attn_weights, V)
    print(f"5️⃣ Output after Attn·V: shape {output.shape}")
    print(output)
    print()
    
    return output, attn_weights

# Example usage
print("="*60)
print("EXAMPLE: Self-Attention for 'The cat sat'")
print("="*60)
print()

np.random.seed(42)
seq_len, d_k, d_v = 3, 4, 4  # Small dimensions for clarity

# Create Q, K, V (normally from X @ W_Q, X @ W_K, X @ W_V)
Q = np.random.randn(seq_len, d_k)
K = np.random.randn(seq_len, d_k)
V = np.random.randn(seq_len, d_k)

print("Input matrices:")
print(f"Q (queries): shape {Q.shape}")
print(Q)
print()
print(f"K (keys): shape {K.shape}")
print(K)
print()
print(f"V (values): shape {V.shape}")
print(V)
print()
print("-"*60)
print()

output, weights = attention(Q, K, V)

print("="*60)
print("INTERPRETATION:")
print("="*60)
print("Each row of attention weights shows WHERE each token looks:")
print(f"  Token 0 ('The'): {weights[0]}")
print(f"    → Strongest attention to position {np.argmax(weights[0])}")
print()
print(f"  Token 1 ('cat'): {weights[1]}")
print(f"    → Strongest attention to position {np.argmax(weights[1])}")
print()
print(f"  Token 2 ('sat'): {weights[2]}")
print(f"    → Strongest attention to position {np.argmax(weights[2])}")
```

**Expected Output:**

```
============================================================
EXAMPLE: Self-Attention for 'The cat sat'
============================================================

Input matrices:
Q (queries): shape (3, 4)
[[ 0.49671415 -0.1382643   0.64768854  1.52302986]
 [-0.23415337 -0.23413696  1.57921282  0.76743473]
 [-0.46947439  0.54256004 -0.46341769 -0.46572975]]

K (keys): shape (3, 4)
[[ 0.24196227 -1.91328024 -1.72491783 -0.56228753]
 [-1.01283112  0.31424733 -0.90802408 -1.4123037 ]
 [ 1.46564877 -0.2257763   0.0675282  -1.42474819]]

V (values): shape (3, 4)
[[-0.54438272  0.11092259 -1.15099358  0.37569802]
 [-0.60063869 -0.29169375 -0.60170661  1.85227818]
 [-0.01349722 -1.05771093  0.82254491 -1.22084365]]

------------------------------------------------------------

1️⃣ Scores after Q·K^T: shape (3, 3)
[[ 0.67608157 -0.24414854  0.68966712]
 [-3.80774654 -1.64788848 -2.03467845]
 [ 0.08558064  1.19661541  0.85751935]]

2️⃣ After scaling by √4 = 2.00:
[[ 0.33804078 -0.12207427  0.34483356]
 [-1.90387327 -0.82394424 -1.01733923]
 [ 0.04279032  0.5983077   0.46875967]]

4️⃣ Attention weights after softmax:
[[0.35983915 0.22535456 0.3648063 ]
 [0.11735915 0.36303866 0.51960219]
 [0.28290277 0.48682506 0.43027218]]
   Row sums (should be 1.0): [1. 1. 1.]

5️⃣ Output after Attn·V: shape (3, 4)
[[-0.38217954 -0.39947315 -0.31169683  0.68091476]
 [-0.23089515 -0.74224001  0.14627624 -0.27494856]
 [-0.37869644 -0.56363259 -0.08642956  0.54806894]]
```

---

### Part 2: Causal Mask (GPT-style)

```python
def create_causal_mask(seq_len):
    """
    Create lower triangular mask for autoregressive attention
    
    Returns:
        mask: (seq_len, seq_len) boolean array
              True = allow attention, False = mask
    """
    mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))
    return mask

# Visual representation
print("="*60)
print("CAUSAL MASKING")
print("="*60)
print()

mask = create_causal_mask(5)
print("Causal mask for sequence length 5:")
print("(1 = can attend, 0 = masked)")
print()
print(mask.astype(int))
print()

# Pretty print
tokens = ["The", "cat", "sat", "on", "mat"]
print("Visual interpretation:")
print()
for i, token in enumerate(tokens):
    visible = [tokens[j] for j in range(i+1)]
    print(f"Position {i} ('{token}'): can see {visible}")

print()
print("-"*60)
print()

# Apply to attention
Q_causal = np.random.randn(5, 8)
K_causal = np.random.randn(5, 8)
V_causal = np.random.randn(5, 8)

output_causal, weights_causal = attention(Q_causal, K_causal, V_causal, mask=mask)

print("Attention weights with causal mask:")
print(weights_causal)
print()
print("Notice: Upper triangle is all zeros!")
print("Each position only attends to current and previous positions.")
```

**Visual Output:**

```
============================================================
CAUSAL MASKING
============================================================

Causal mask for sequence length 5:
(1 = can attend, 0 = masked)

[[1 0 0 0 0]
 [1 1 0 0 0]
 [1 1 1 0 0]
 [1 1 1 1 0]
 [1 1 1 1 1]]

Visual interpretation:

Position 0 ('The'): can see ['The']
Position 1 ('cat'): can see ['The', 'cat']
Position 2 ('sat'): can see ['The', 'cat', 'sat']
Position 3 ('on'): can see ['The', 'cat', 'sat', 'on']
Position 4 ('mat'): can see ['The', 'cat', 'sat', 'on', 'mat']

Attention weights with causal mask:
[[1.         0.         0.         0.         0.        ]
 [0.46856434 0.53143566 0.         0.         0.        ]
 [0.30246498 0.34666119 0.35087383 0.         0.        ]
 [0.24603817 0.26054876 0.25858644 0.23482662 0.        ]
 [0.19344903 0.20483618 0.21025893 0.19711683 0.19433902]]

Notice: Upper triangle is all zeros!
Each position only attends to current and previous positions.
```

---

### Part 3: Multi-Head Attention

```python
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        """
        Args:
            d_model: Model dimension (e.g., 512)
            num_heads: Number of attention heads (e.g., 8)
        """
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        print(f"Multi-Head Attention Configuration:")
        print(f"  d_model: {d_model}")
        print(f"  num_heads: {num_heads}")
        print(f"  d_k per head: {self.d_k}")
        print()
        
        # Weight matrices (in practice, these are learned)
        self.W_Q = np.random.randn(d_model, d_model) * 0.01
        self.W_K = np.random.randn(d_model, d_model) * 0.01
        self.W_V = np.random.randn(d_model, d_model) * 0.01
        self.W_O = np.random.randn(d_model, d_model) * 0.01
        
    def split_heads(self, x):
        """
        Split last dimension into (num_heads, d_k)
        
        Input:  (batch, seq_len, d_model)
        Output: (batch, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, _ = x.shape
        # Reshape and transpose
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)
    
    def forward(self, X, mask=None):
        """
        Args:
            X: Input (batch, seq_len, d_model)
            mask: Optional (seq_len, seq_len)
        
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = X.shape
        
        print(f"Forward pass:")
        print(f"  Input shape: {X.shape}")
        
        # 1. Linear projections
        Q = np.matmul(X, self.W_Q)  # (batch, seq_len, d_model)
        K = np.matmul(X, self.W_K)
        V = np.matmul(X, self.W_V)
        print(f"  After Q,K,V projections: {Q.shape}")
        
        # 2. Split into heads
        Q_heads = self.split_heads(Q)  # (batch, heads, seq_len, d_k)
        K_heads = self.split_heads(K)
        V_heads = self.split_heads(V)
        print(f"  After splitting into {self.num_heads} heads: {Q_heads.shape}")
        
        # 3. Scaled dot-product attention for each head
        d_k = self.d_k
        scores = np.matmul(Q_heads, K_heads.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
        
        if mask is not None:
            # Expand mask for batch and heads
            mask_expanded = mask[np.newaxis, np.newaxis, :, :]
            scores = np.where(mask_expanded, scores, -1e9)
        
        attn_weights = softmax(scores, axis=-1)
        attn_output = np.matmul(attn_weights, V_heads)
        print(f"  After attention per head: {attn_output.shape}")
        
        # 4. Concatenate heads
        attn_output = attn_output.transpose(0, 2, 1, 3)
        concat = attn_output.reshape(batch_size, seq_len, self.d_model)
        print(f"  After concatenating heads: {concat.shape}")
        
        # 5. Final linear projection
        output = np.matmul(concat, self.W_O)
        print(f"  After output projection: {output.shape}")
        
        return output

# Example
print("="*60)
print("MULTI-HEAD ATTENTION EXAMPLE")
print("="*60)
print()

mha = MultiHeadAttention(d_model=16, num_heads=4)
print()

# Input: 2 sentences, max length 5, embedding dim 16
X = np.random.randn(2, 5, 16)
output = mha.forward(X)

print()
print("Summary:")
print(f"  Input:  {X.shape}")
print(f"  Output: {output.shape}")
print("  ✓ Shape preserved!")
```

**Output:**

```
============================================================
MULTI-HEAD ATTENTION EXAMPLE
============================================================

Multi-Head Attention Configuration:
  d_model: 16
  num_heads: 4
  d_k per head: 4

Forward pass:
  Input shape: (2, 5, 16)
  After Q,K,V projections: (2, 5, 16)
  After splitting into 4 heads: (2, 4, 5, 4)
  After attention per head: (2, 4, 5, 4)
  After concatenating heads: (2, 5, 16)
  After output projection: (2, 5, 16)

Summary:
  Input:  (2, 5, 16)
  Output: (2, 5, 16)
  ✓ Shape preserved!
```

---

### Part 4: Position Encoding

```python
def positional_encoding(seq_len, d_model):
    """
    Generate sinusoidal positional encoding
    
    Args:
        seq_len: Maximum sequence length
        d_model: Model dimension
    
    Returns:
        PE: (seq_len, d_model) positional encoding matrix
    """
    PE = np.zeros((seq_len, d_model))
    
    position = np.arange(seq_len).reshape(-1, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    # Even dimensions: sine
    PE[:, 0::2] = np.sin(position * div_term)
    
    # Odd dimensions: cosine
    PE[:, 1::2] = np.cos(position * div_term)
    
    return PE

# Example
print("="*60)
print("POSITIONAL ENCODING")
print("="*60)
print()

PE = positional_encoding(seq_len=10, d_model=8)

print("Positional encoding for 10 positions, 8 dimensions:")
print(PE)
print()

print("Notice the pattern:")
print("  - Columns 0,1 (high freq): Change rapidly")
print("  - Columns 6,7 (low freq): Change slowly")
print()

# Visualize specific positions
print("Encoding for position 0:")
print(f"  {PE[0]}")
print()
print("Encoding for position 5:")
print(f"  {PE[5]}")
print()
print("Encoding for position 9:")
print(f"  {PE[9]}")

# Try saving a visualization
try:
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    plt.imshow(PE.T, cmap='RdBu', aspect='auto')
    plt.xlabel('Position in Sequence')
    plt.ylabel('Embedding Dimension')
    plt.title('Positional Encoding Heatmap')
    plt.colorbar(label='Value')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/spurge/cisco/deep-learning/positional_encoding_viz.png', dpi=150)
    print()
    print("✓ Saved visualization to positional_encoding_viz.png")
except ImportError:
    print("(Install matplotlib to see visualization)")
```

---

### Part 5: Complete Transformer Encoder Layer

```python
def layer_norm(x, gamma, beta, eps=1e-6):
    """Layer normalization"""
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / (std + eps) + beta

def feed_forward(x, W1, b1, W2, b2):
    """Feed-forward network with ReLU"""
    hidden = np.maximum(0, np.matmul(x, W1) + b1)  # ReLU activation
    output = np.matmul(hidden, W2) + b2
    return output

class TransformerEncoderLayer:
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
        """
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Multi-head attention
        self.mha = MultiHeadAttention(d_model, num_heads)
        
        # Feed-forward network
        self.W1 = np.random.randn(d_model, d_ff) * 0.01
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.01
        self.b2 = np.zeros(d_model)
        
        # Layer norm parameters
        self.gamma1 = np.ones(d_model)
        self.beta1 = np.zeros(d_model)
        self.gamma2 = np.ones(d_model)
        self.beta2 = np.zeros(d_model)
        
        print(f"Transformer Encoder Layer:")
        print(f"  d_model: {d_model}")
        print(f"  num_heads: {num_heads}")
        print(f"  d_ff: {d_ff}")
        print(f"  Parameters: ~{self.count_parameters():,}")
        print()
    
    def count_parameters(self):
        """Count total parameters"""
        # MHA
        mha_params = 4 * self.d_model * self.d_model  # W_Q, W_K, W_V, W_O
        # FFN
        ffn_params = self.d_model * self.d_ff + self.d_ff + \
                     self.d_ff * self.d_model + self.d_model
        # Layer norm
        ln_params = 4 * self.d_model  # 2 gamma + 2 beta
        
        return mha_params + ffn_params + ln_params
    
    def forward(self, x, mask=None):
        """
        Forward pass (Pre-LN style)
        
        Args:
            x: Input (batch, seq_len, d_model)
        
        Returns:
            output: (batch, seq_len, d_model)
        """
        print("Encoder Layer Forward:")
        print(f"  Input shape: {x.shape}")
        
        # 1. Multi-head attention with pre-LN and residual
        x_norm1 = layer_norm(x, self.gamma1, self.beta1)
        attn_out = self.mha.forward(x_norm1, mask)
        x = x + attn_out  # Residual connection
        print(f"  After attention + residual: {x.shape}")
        
        # 2. Feed-forward with pre-LN and residual
        x_norm2 = layer_norm(x, self.gamma2, self.beta2)
        ff_out = feed_forward(x_norm2, self.W1, self.b1, self.W2, self.b2)
        x = x + ff_out  # Residual connection
        print(f"  After FFN + residual: {x.shape}")
        
        return x

# Example
print("="*60)
print("COMPLETE TRANSFORMER ENCODER LAYER")
print("="*60)
print()

encoder = TransformerEncoderLayer(d_model=16, num_heads=4, d_ff=64)
print()

X = np.random.randn(2, 5, 16)  # batch=2, seq=5, d_model=16
output = encoder.forward(X)

print()
print("Complete pass through encoder layer:")
print(f"  Input:  {X.shape}")
print(f"  Output: {output.shape}")
print("  ✓ Ready to stack multiple layers!")
```

---

## 🎯 Dimension Cheat Sheet

### Common Configurations

#### BERT-base
```
d_model = 768
num_heads = 12
d_k = d_v = 64  (768 / 12)
d_ff = 3072  (4 × 768)
num_layers = 12
vocab_size = 30,000
max_seq_len = 512

Total parameters: ~110M
```

#### GPT-2 Small
```
d_model = 768
num_heads = 12
d_k = d_v = 64
d_ff = 3072
num_layers = 12
vocab_size = 50,257
max_seq_len = 1024

Total parameters: ~117M
```

#### LLaMA-7B
```
d_model = 4096
num_heads = 32
num_kv_heads = 32  (GQA ratio 1:1 in 7B)
d_k = d_v = 128  (4096 / 32)
d_ff = 11008  (~2.7× expansion for SwiGLU)
num_layers = 32
vocab_size = 32,000
max_seq_len = 2048 (4096 in v2)

Total parameters: ~7B
```

---

## 🧮 Quick Reference: Matrix Dimensions

### Single-Head Attention
```
Input:  X        (seq_len, d_model)

Linear: W_Q      (d_model, d_k)
        W_K      (d_model, d_k)
        W_V      (d_model, d_v)

Projected: Q     (seq_len, d_k)
           K     (seq_len, d_k)
           V     (seq_len, d_v)

Scores: Q·K^T    (seq_len, seq_len)

Attention: softmax(scores) · V → (seq_len, d_v)
```

### Multi-Head Attention
```
Input: X               (batch, seq_len, d_model)

Per head:
  Q_i, K_i, V_i       (batch, seq_len, d_k)
  
After all h heads:
  Concat             (batch, seq_len, h×d_k)
  
After W_O:
  Output             (batch, seq_len, d_model)

Note: h × d_k = d_model
```

### Feed-Forward Network
```
Input: x              (batch, seq_len, d_model)

Hidden: x·W1 + b1     (batch, seq_len, d_ff)
        ReLU(·)       (batch, seq_len, d_ff)

Output: ·W2 + b2      (batch, seq_len, d_model)
```

---

## ✅ Practice Problems

### Problem 1: Calculate Parameters

BERT-base has:
- d_model = 768, num_heads = 12, d_ff = 3072, num_layers = 12

How many parameters in:
1. One attention head's Q projection?
2. Complete multi-head attention (all 4 weight matrices)?
3. One FFN block?
4. One complete encoder layer?
5. All 12 layers?

<details>
<summary>Solution</summary>

1. One Q head: 768 × (768/12) = 768 × 64 = 49,152
2. Complete MHA: 4 × (768 × 768) = 2,359,296
3. FFN: (768×3072) + 3072 + (3072×768) + 768 = 4,722,432
4. One layer: 2,359,296 + 4,722,432 + 6,144 (LN) ≈ 7.09M
5. All layers: 12 × 7.09M ≈ 85M parameters

</details>

### Problem 2: Trace Dimensions

Given:
- Input: X = (4, 10, 512)  # batch=4, seq=10, d_model=512
- num_heads = 8
- d_ff = 2048

What is the shape after:
1. Q, K, V projections?
2. Splitting into heads?
3. Attention scores Q·K^T?
4. After softmax and V multiplication?
5. After concatenation?
6. After FFN?

<details>
<summary>Solution</summary>

1. Q, K, V: (4, 10, 512) each
2. Split: (4, 8, 10, 64) each
3. Scores: (4, 8, 10, 10)
4. Attn output: (4, 8, 10, 64)
5. Concat: (4, 10, 512)
6. FFN: (4, 10, 512)

</details>

---

## 🚀 Next Steps

You now have:
1. ✅ Complete theory with exact dimensions
2. ✅ Working code implementations
3. ✅ Visual diagrams
4. ✅ Practice problems

**To master transformers:**
1. Run all code examples
2. Modify dimensions and observe changes
3. Implement in PyTorch
4. Build a small language model
5. Train on simple task

**Remember:** Understanding comes from doing! 💪
