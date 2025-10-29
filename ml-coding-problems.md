# ML Coding Practice Problems

## Problem Set for Sindhuja's Coding Round

### General Guidelines
- Write clean, readable code
- Think aloud - explain your approach
- Consider edge cases
- Discuss time/space complexity
- Ask clarifying questions
- Test your solution

---

## Problem 1: Attention Mechanism Implementation

**Difficulty:** Medium
**Time:** 30-40 minutes
**Topics:** Transformers, Neural Networks

**Problem:**
Implement a simple scaled dot-product attention mechanism from scratch using NumPy or PyTorch.

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Args:
        Q: Query matrix (batch_size, seq_len, d_k)
        K: Key matrix (batch_size, seq_len, d_k)
        V: Value matrix (batch_size, seq_len, d_v)
        mask: Optional mask (batch_size, seq_len, seq_len)

    Returns:
        output: Attention output (batch_size, seq_len, d_v)
        attention_weights: Attention weights (batch_size, seq_len, seq_len)
    """
    # Your implementation here
    pass
```

**Solution:**

```python
import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled Dot-Product Attention implementation
    """
    # Get dimension of keys
    d_k = Q.shape[-1]

    # Compute attention scores: Q @ K^T / sqrt(d_k)
    # Shape: (batch_size, seq_len_q, seq_len_k)
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)

    # Apply mask if provided (e.g., for padding or causal attention)
    if mask is not None:
        scores = scores + (mask * -1e9)  # Large negative value

    # Apply softmax to get attention weights
    # Shape: (batch_size, seq_len_q, seq_len_k)
    attention_weights = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)

    # Apply attention to values
    # Shape: (batch_size, seq_len_q, d_v)
    output = np.matmul(attention_weights, V)

    return output, attention_weights

# Test
batch_size, seq_len, d_k, d_v = 2, 4, 8, 8
Q = np.random.randn(batch_size, seq_len, d_k)
K = np.random.randn(batch_size, seq_len, d_k)
V = np.random.randn(batch_size, seq_len, d_v)

output, weights = scaled_dot_product_attention(Q, K, V)
print(f"Output shape: {output.shape}")  # (2, 4, 8)
print(f"Attention weights shape: {weights.shape}")  # (2, 4, 4)
print(f"Attention weights sum: {weights.sum(axis=-1)}")  # Should be all 1s
```

**Key Points to Discuss:**
- Why divide by sqrt(d_k)? (Prevents softmax saturation)
- How does masking work? (Causal/padding masks)
- Time complexity: O(n²d) where n is sequence length
- Space complexity: O(n²) for attention weights

---

## Problem 2: Cosine Similarity for Sentence Embeddings

**Difficulty:** Easy-Medium
**Time:** 20-25 minutes
**Topics:** NLP, Embeddings, Similarity

**Problem:**
Given a list of sentence embeddings and a query embedding, find the top-k most similar sentences using cosine similarity.

```python
def find_similar_sentences(query_embedding, sentence_embeddings, k=5):
    """
    Args:
        query_embedding: numpy array of shape (d,)
        sentence_embeddings: numpy array of shape (n, d)
        k: number of top similar sentences to return

    Returns:
        indices: indices of top-k similar sentences
        similarities: cosine similarity scores
    """
    # Your implementation here
    pass
```

**Solution:**

```python
import numpy as np

def find_similar_sentences(query_embedding, sentence_embeddings, k=5):
    """
    Find top-k most similar sentences using cosine similarity
    """
    # Normalize query embedding
    query_norm = query_embedding / np.linalg.norm(query_embedding)

    # Normalize sentence embeddings
    sentence_norms = sentence_embeddings / np.linalg.norm(
        sentence_embeddings, axis=1, keepdims=True
    )

    # Compute cosine similarities (dot product of normalized vectors)
    similarities = np.dot(sentence_norms, query_norm)

    # Get top-k indices (in descending order)
    top_k_indices = np.argsort(similarities)[::-1][:k]

    # Get corresponding similarity scores
    top_k_similarities = similarities[top_k_indices]

    return top_k_indices, top_k_similarities

# Test
np.random.seed(42)
embedding_dim = 384
num_sentences = 100

query_embedding = np.random.randn(embedding_dim)
sentence_embeddings = np.random.randn(num_sentences, embedding_dim)

indices, similarities = find_similar_sentences(query_embedding, sentence_embeddings, k=5)

print(f"Top 5 similar sentence indices: {indices}")
print(f"Similarity scores: {similarities}")
```

**Optimization for Large Scale:**

```python
# For very large datasets, use approximate nearest neighbor search
# FAISS example:
import faiss

def find_similar_sentences_faiss(query_embedding, sentence_embeddings, k=5):
    """
    Fast similarity search using FAISS
    """
    d = sentence_embeddings.shape[1]

    # Build FAISS index
    index = faiss.IndexFlatIP(d)  # Inner Product (cosine with normalized vectors)

    # Normalize embeddings
    faiss.normalize_L2(sentence_embeddings)
    faiss.normalize_L2(query_embedding.reshape(1, -1))

    # Add to index
    index.add(sentence_embeddings.astype('float32'))

    # Search
    similarities, indices = index.search(query_embedding.reshape(1, -1).astype('float32'), k)

    return indices[0], similarities[0]
```

**Key Points:**
- Cosine similarity = dot product of normalized vectors
- Time complexity: O(nd) for brute force
- For large scale: use approximate methods (FAISS, Annoy)
- Normalization is critical for cosine similarity

---

## Problem 3: Custom Data Collator for Fine-tuning

**Difficulty:** Medium
**Time:** 25-30 minutes
**Topics:** Data Processing, Fine-tuning

**Problem:**
Implement a data collator for batching variable-length sequences in LLM fine-tuning. Handle padding and attention masks.

```python
def collate_batch(batch, tokenizer, max_length=512):
    """
    Args:
        batch: List of dictionaries with 'input_ids' and 'labels'
        tokenizer: Tokenizer with pad_token_id
        max_length: Maximum sequence length

    Returns:
        Dictionary with padded tensors
    """
    # Your implementation here
    pass
```

**Solution:**

```python
import torch

def collate_batch(batch, tokenizer, max_length=512):
    """
    Collate batch of variable-length sequences with padding
    """
    # Extract input_ids and labels
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]

    # Find max length in this batch (up to max_length)
    batch_max_length = min(max(len(ids) for ids in input_ids), max_length)

    # Initialize padded tensors
    batch_size = len(batch)
    padded_input_ids = torch.full(
        (batch_size, batch_max_length),
        tokenizer.pad_token_id,
        dtype=torch.long
    )
    padded_labels = torch.full(
        (batch_size, batch_max_length),
        -100,  # Ignore index for loss calculation
        dtype=torch.long
    )
    attention_mask = torch.zeros((batch_size, batch_max_length), dtype=torch.long)

    # Fill in the actual values
    for i, (ids, lbls) in enumerate(zip(input_ids, labels)):
        length = min(len(ids), batch_max_length)
        padded_input_ids[i, :length] = torch.tensor(ids[:length])
        padded_labels[i, :length] = torch.tensor(lbls[:length])
        attention_mask[i, :length] = 1

    return {
        'input_ids': padded_input_ids,
        'attention_mask': attention_mask,
        'labels': padded_labels
    }

# Test
class MockTokenizer:
    pad_token_id = 0

tokenizer = MockTokenizer()

batch = [
    {'input_ids': [1, 2, 3, 4, 5], 'labels': [1, 2, 3, 4, 5]},
    {'input_ids': [1, 2, 3], 'labels': [1, 2, 3]},
    {'input_ids': [1, 2, 3, 4, 5, 6, 7], 'labels': [1, 2, 3, 4, 5, 6, 7]},
]

result = collate_batch(batch, tokenizer)
print("Input IDs shape:", result['input_ids'].shape)
print("Attention mask:\n", result['attention_mask'])
print("Labels:\n", result['labels'])
```

**Key Points:**
- Padding to longest sequence in batch (not global max) for efficiency
- Attention mask marks real tokens (1) vs padding (0)
- Labels use -100 for padding (ignored in CrossEntropyLoss)
- Dynamic batching vs static batching trade-offs

---

## Problem 4: Simple Gradient Descent Implementation

**Difficulty:** Easy-Medium
**Time:** 20-25 minutes
**Topics:** Optimization, ML Fundamentals

**Problem:**
Implement gradient descent to minimize a simple function f(x) = (x-3)² + 5

```python
def gradient_descent(learning_rate=0.1, num_iterations=100, initial_x=0):
    """
    Minimize f(x) = (x-3)^2 + 5 using gradient descent

    Returns:
        x_history: list of x values at each iteration
        f_history: list of function values at each iteration
    """
    # Your implementation here
    pass
```

**Solution:**

```python
import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(learning_rate=0.1, num_iterations=100, initial_x=0):
    """
    Minimize f(x) = (x-3)^2 + 5 using gradient descent

    f(x) = (x-3)^2 + 5
    f'(x) = 2(x-3)
    """
    x = initial_x
    x_history = [x]
    f_history = []

    for i in range(num_iterations):
        # Compute function value
        f_x = (x - 3)**2 + 5
        f_history.append(f_x)

        # Compute gradient
        gradient = 2 * (x - 3)

        # Update x
        x = x - learning_rate * gradient
        x_history.append(x)

        # Early stopping if converged
        if abs(gradient) < 1e-6:
            print(f"Converged at iteration {i}")
            break

    return x_history, f_history

# Test
x_hist, f_hist = gradient_descent(learning_rate=0.1, num_iterations=50, initial_x=0)

print(f"Final x: {x_hist[-1]:.6f}")  # Should be close to 3
print(f"Final f(x): {f_hist[-1]:.6f}")  # Should be close to 5
print(f"Number of iterations: {len(f_hist)}")

# Visualization (optional but impressive)
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(x_hist, marker='o')
plt.xlabel('Iteration')
plt.ylabel('x')
plt.title('x value over iterations')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(f_hist, marker='o')
plt.xlabel('Iteration')
plt.ylabel('f(x)')
plt.title('Loss over iterations')
plt.grid(True)
plt.tight_layout()
```

**Extension - Adaptive Learning Rate:**

```python
def gradient_descent_adam(initial_x=0, num_iterations=100,
                          learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Implement Adam optimizer
    """
    x = initial_x
    m = 0  # First moment
    v = 0  # Second moment
    x_history = [x]

    for t in range(1, num_iterations + 1):
        # Compute gradient
        gradient = 2 * (x - 3)

        # Update biased first moment estimate
        m = beta1 * m + (1 - beta1) * gradient

        # Update biased second moment estimate
        v = beta2 * v + (1 - beta2) * (gradient ** 2)

        # Compute bias-corrected moment estimates
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        # Update parameters
        x = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        x_history.append(x)

    return x_history

x_hist_adam = gradient_descent_adam(initial_x=0, num_iterations=50)
print(f"Adam - Final x: {x_hist_adam[-1]:.6f}")
```

**Key Discussion Points:**
- Learning rate selection
- Convergence criteria
- Gradient descent variants (SGD, Adam, AdamW)
- Momentum and adaptive learning rates

---

## Problem 5: Confusion Matrix and Metrics

**Difficulty:** Easy
**Time:** 15-20 minutes
**Topics:** Evaluation, Classification

**Problem:**
Implement functions to compute precision, recall, and F1-score from scratch.

```python
def compute_metrics(y_true, y_pred):
    """
    Args:
        y_true: Ground truth labels (list or numpy array)
        y_pred: Predicted labels (list or numpy array)

    Returns:
        Dictionary with precision, recall, f1_score
    """
    # Your implementation here
    pass
```

**Solution:**

```python
import numpy as np

def compute_metrics(y_true, y_pred):
    """
    Compute classification metrics from scratch
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # True Positives, False Positives, False Negatives, True Negatives
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    # Precision = TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Recall = TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Accuracy = (TP + TN) / Total
    accuracy = (tp + tn) / len(y_true)

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'confusion_matrix': {
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
        }
    }

# Test
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]

metrics = compute_metrics(y_true, y_pred)
print("Metrics:")
for key, value in metrics.items():
    print(f"  {key}: {value}")

# Verify with sklearn
from sklearn.metrics import precision_score, recall_score, f1_score
print("\nSklearn verification:")
print(f"  Precision: {precision_score(y_true, y_pred)}")
print(f"  Recall: {recall_score(y_true, y_pred)}")
print(f"  F1: {f1_score(y_true, y_pred)}")
```

**Multi-class Extension:**

```python
def compute_metrics_multiclass(y_true, y_pred, num_classes):
    """
    Compute macro-averaged metrics for multi-class classification
    """
    precisions = []
    recalls = []
    f1_scores = []

    for class_id in range(num_classes):
        # One-vs-rest for each class
        y_true_binary = (np.array(y_true) == class_id).astype(int)
        y_pred_binary = (np.array(y_pred) == class_id).astype(int)

        metrics = compute_metrics(y_true_binary, y_pred_binary)
        precisions.append(metrics['precision'])
        recalls.append(metrics['recall'])
        f1_scores.append(metrics['f1_score'])

    return {
        'macro_precision': np.mean(precisions),
        'macro_recall': np.mean(recalls),
        'macro_f1': np.mean(f1_scores)
    }

# Test multi-class
y_true_mc = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
y_pred_mc = [0, 1, 1, 0, 1, 2, 2, 1, 2, 0]

metrics_mc = compute_metrics_multiclass(y_true_mc, y_pred_mc, num_classes=3)
print("\nMulti-class metrics:")
print(metrics_mc)
```

---

## Problem 6: K-Means Clustering Implementation

**Difficulty:** Medium
**Time:** 30-35 minutes
**Topics:** Unsupervised Learning, Clustering

**Problem:**
Implement K-means clustering from scratch.

**Solution:**

```python
import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iters=100, random_state=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None

    def fit(self, X):
        """
        Fit K-means clustering

        Args:
            X: numpy array of shape (n_samples, n_features)
        """
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape

        # Initialize centroids randomly from data points
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        for iteration in range(self.max_iters):
            # Assign each point to nearest centroid
            labels = self._assign_clusters(X)

            # Save old centroids for convergence check
            old_centroids = self.centroids.copy()

            # Update centroids
            for k in range(self.n_clusters):
                cluster_points = X[labels == k]
                if len(cluster_points) > 0:
                    self.centroids[k] = cluster_points.mean(axis=0)

            # Check convergence
            if np.allclose(old_centroids, self.centroids):
                print(f"Converged at iteration {iteration}")
                break

        return self

    def _assign_clusters(self, X):
        """
        Assign each point to nearest centroid
        """
        distances = np.zeros((X.shape[0], self.n_clusters))

        for k in range(self.n_clusters):
            # Euclidean distance to centroid k
            distances[:, k] = np.linalg.norm(X - self.centroids[k], axis=1)

        # Return index of nearest centroid for each point
        return np.argmin(distances, axis=1)

    def predict(self, X):
        """
        Predict cluster labels for new data
        """
        return self._assign_clusters(X)

    def inertia(self, X):
        """
        Compute within-cluster sum of squares
        """
        labels = self.predict(X)
        inertia = 0

        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - self.centroids[k]) ** 2)

        return inertia

# Test
from sklearn.datasets import make_blobs

X, y_true = make_blobs(n_samples=300, centers=4, n_features=2, random_state=42)

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)
labels = kmeans.predict(X)

print(f"Final centroids shape: {kmeans.centroids.shape}")
print(f"Inertia: {kmeans.inertia(X):.2f}")
print(f"Unique labels: {np.unique(labels)}")

# Visualization (optional)
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.6)
plt.title('True Labels')

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1],
           marker='X', s=200, c='red', edgecolors='black', linewidths=2)
plt.title('K-Means Clustering')

plt.tight_layout()
```

**Key Discussion Points:**
- Initialization methods (K-means++, random)
- Convergence criteria
- Time complexity: O(n * k * i * d) where n=samples, k=clusters, i=iterations, d=dimensions
- Limitations: assumes spherical clusters, sensitive to initialization

---

## Problem 7: Softmax with Numerical Stability

**Difficulty:** Easy-Medium
**Time:** 15-20 minutes
**Topics:** Neural Networks, Numerical Stability

**Problem:**
Implement numerically stable softmax function.

**Solution:**

```python
import numpy as np

def softmax_naive(x):
    """
    Naive softmax - can cause numerical overflow
    """
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

def softmax_stable(x):
    """
    Numerically stable softmax
    Subtract max value to prevent overflow
    """
    # Shift values by subtracting max
    x_shifted = x - np.max(x, axis=-1, keepdims=True)

    # Compute softmax
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Test with large values
x_large = np.array([[1000, 1001, 1002], [1, 2, 3]])

print("Testing with large values:")
print("Input:", x_large[0])

try:
    result_naive = softmax_naive(x_large[0])
    print("Naive softmax:", result_naive)
except:
    print("Naive softmax: OVERFLOW ERROR")

result_stable = softmax_stable(x_large[0])
print("Stable softmax:", result_stable)
print("Sum of probabilities:", result_stable.sum())

# Test with normal values
x_normal = np.array([[1, 2, 3], [4, 5, 6]])
print("\nTesting with normal values:")
print("Stable softmax:")
print(softmax_stable(x_normal))
```

**Cross-Entropy Loss with Softmax:**

```python
def cross_entropy_loss(logits, labels):
    """
    Compute cross-entropy loss with numerical stability

    Args:
        logits: (batch_size, num_classes)
        labels: (batch_size,) - class indices
    """
    batch_size = logits.shape[0]

    # Stable softmax
    probs = softmax_stable(logits)

    # Log probabilities
    log_probs = np.log(probs + 1e-9)  # Small epsilon to avoid log(0)

    # Gather log probabilities for true labels
    log_probs_for_labels = log_probs[np.arange(batch_size), labels]

    # Negative log likelihood
    loss = -np.mean(log_probs_for_labels)

    return loss

# Test
logits = np.array([[2.0, 1.0, 0.1],
                   [0.5, 2.1, 0.3],
                   [0.1, 0.2, 2.5]])
labels = np.array([0, 1, 2])

loss = cross_entropy_loss(logits, labels)
print(f"\nCross-entropy loss: {loss:.4f}")
```

**Key Points:**
- Why subtract max? Prevents exp() overflow while keeping result mathematically equivalent
- Log-sum-exp trick for numerical stability
- Combining softmax with cross-entropy for efficiency

---

## Quick Reference - Common Imports

```python
# Essential imports for ML coding interviews
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# For transformers
from transformers import AutoTokenizer, AutoModel

# Visualization (bonus points)
import matplotlib.pyplot as plt
import seaborn as sns

# Metrics and datasets
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_blobs
```

---

## Interview Tips for Coding Round

1. **Clarify Requirements**
   - Ask about input/output formats
   - Confirm edge cases
   - Understand performance requirements

2. **Think Aloud**
   - Explain your approach before coding
   - Discuss trade-offs
   - Mention alternative solutions

3. **Write Clean Code**
   - Use meaningful variable names
   - Add comments for complex logic
   - Follow Python conventions (PEP 8)

4. **Test Your Solution**
   - Walk through a simple example
   - Test edge cases (empty input, single element, etc.)
   - Verify output shapes and types

5. **Discuss Complexity**
   - Time complexity
   - Space complexity
   - How to optimize for large scale

6. **Be Ready to Extend**
   - How would you handle batching?
   - How to optimize for production?
   - What if dataset doesn't fit in memory?

## Topics to Review Before Interview

- [ ] NumPy operations (broadcasting, indexing, matrix operations)
- [ ] PyTorch basics (tensors, autograd, nn.Module)
- [ ] Attention mechanism implementation
- [ ] Data loading and batching
- [ ] Common ML metrics implementation
- [ ] Numerical stability tricks
- [ ] Vectorization techniques
- [ ] Basic optimization algorithms
