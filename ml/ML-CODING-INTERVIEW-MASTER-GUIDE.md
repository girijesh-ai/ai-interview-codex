# ML Coding Interview - Complete Master Guide

## Cisco ML Engineer Coding Round 2025

> **Complete preparation guide for ML coding interviews at Cisco and FAANG companies**

---

## Table of Contents

1. [Interview Format](#interview-format)
2. [What to Expect](#what-to-expect)
3. [Topics Covered](#topics-covered)
4. [Preparation Timeline](#preparation-timeline)
5. [Must-Know Algorithms](#must-know-algorithms)
6. [Common Interview Questions](#common-interview-questions)
7. [Coding Patterns](#coding-patterns)
8. [Interview Tips](#interview-tips)
9. [Resources Created](#resources-created)

---

## Interview Format

### Cisco ML Engineer Coding Round Structure

Based on 2025 interview data:

**Online Assessment (OA):**
- **Duration:** 90 minutes
- **Questions:** 30 total
  - 2 DSA questions (LeetCode medium-hard)
  - 3-5 ML coding problems
  - Rest: ML theory, system design concepts

**Live Coding Round:**
- **Duration:** 45-60 minutes
- **Format:** Implement ML algorithms from scratch
- **Allowed:** NumPy, SciPy (sometimes)
- **NOT Allowed:** sklearn, TensorFlow, PyTorch high-level APIs

**Common Topics:**
1. **Algorithm Implementation** (60%)
   - Linear/Logistic Regression
   - K-Means, k-NN
   - Neural network components
   - Gradient Descent variants

2. **Data Processing** (20%)
   - Feature scaling/normalization
   - Data collation/batching
   - Missing value handling

3. **Model Evaluation** (10%)
   - Metrics from scratch (precision, recall, F1, AUC)
   - Cross-validation
   - Confusion matrix

4. **Production Scenarios** (10%)
   - Memory-efficient implementations
   - Batch processing
   - Model deployment considerations

---

## What to Expect

### Difficulty Distribution

| Difficulty | Percentage | Example |
|-----------|-----------|---------|
| **Easy** | 20% | Train-test split, feature scaling, basic metrics |
| **Medium** | 60% | Linear regression, logistic regression, k-NN, activation functions |
| **Hard** | 20% | Complete neural network, K-means++, Adam optimizer |

### Example Interview Questions (Real from 2025)

**Google:** "Implement K-Means using NumPy only"

**Uber:** "Write AUC calculation from scratch using vanilla Python"

**Startup:** "Code Gradient Descent from scratch using NumPy and SciPy only"

**Meta:** "Implement softmax with numerical stability"

**Amazon:** "Code logistic regression with L2 regularization"

**Cisco-Specific:**
- "Implement momentum-based SGD"
- "Code attention mechanism for transformer"
- "Build data pipeline for fine-tuning"

---

## Topics Covered

### ✅ Core ML Algorithms (MUST KNOW)

These are the **ONLY 4 algorithms** commonly asked:

1. **Linear Regression**
   - Normal equation
   - Gradient descent
   - Time/space complexity

2. **Logistic Regression**
   - Sigmoid function
   - Cross-entropy loss
   - Gradient descent

3. **K-Nearest Neighbors (k-NN)**
   - Distance metrics
   - Majority voting
   - Optimization with vectorization

4. **K-Means Clustering**
   - Lloyd's algorithm
   - K-Means++
   - Elbow method

### ✅ Neural Network Components

1. **Layers**
   - Linear (Dense) layer
   - Forward/backward pass
   - Parameter initialization

2. **Activations**
   - ReLU, Sigmoid, Tanh
   - Softmax
   - Derivatives

3. **Loss Functions**
   - MSE (regression)
   - Binary cross-entropy
   - Cross-entropy (multi-class)

4. **Optimizers**
   - SGD
   - SGD with Momentum
   - Adam

### ✅ Supporting Functions

- Train-test split
- Feature scaling (StandardScaler, MinMaxScaler)
- K-fold cross-validation
- Metrics (accuracy, precision, recall, F1, AUC)
- One-hot encoding
- Batch processing

---

## Preparation Timeline

### Week 1-2: Core Algorithms (Priority 1)

**Day 1-2:** Linear Regression
- [ ] Implement normal equation
- [ ] Implement gradient descent
- [ ] Practice on toy dataset
- [ ] Understand when to use each method

**Day 3-4:** Logistic Regression
- [ ] Implement from scratch
- [ ] Binary cross-entropy loss
- [ ] Gradient descent with cross-entropy
- [ ] Test on classification dataset

**Day 5-6:** K-Nearest Neighbors
- [ ] Implement basic k-NN
- [ ] Optimize with vectorization
- [ ] Try different distance metrics
- [ ] Understand curse of dimensionality

**Day 7-8:** K-Means Clustering
- [ ] Implement Lloyd's algorithm
- [ ] Implement K-Means++
- [ ] Elbow method for choosing k
- [ ] Handle edge cases (empty clusters)

**Day 9-10:** Review & Practice
- [ ] Solve 5-10 problems using these algorithms
- [ ] Time yourself
- [ ] Practice explaining approach

### Week 3: Neural Network Components (Priority 2)

**Day 1-2:** Layers & Activations
- [ ] Linear layer forward/backward
- [ ] ReLU, Sigmoid, Tanh, Softmax
- [ ] Understand gradient flow

**Day 3-4:** Loss Functions
- [ ] MSE
- [ ] Binary cross-entropy
- [ ] Cross-entropy + Softmax

**Day 5-6:** Optimizers
- [ ] SGD
- [ ] Momentum
- [ ] Adam

**Day 7:** Complete Network
- [ ] Build end-to-end network
- [ ] Train on real dataset
- [ ] Visualize training

### Week 4: Supporting Functions & Practice (Priority 3)

**Day 1-2:** Data Processing
- [ ] Train-test split
- [ ] Feature scaling
- [ ] One-hot encoding
- [ ] Batch processing

**Day 3-4:** Metrics & Evaluation
- [ ] Accuracy, Precision, Recall, F1
- [ ] Confusion matrix
- [ ] AUC/ROC from scratch
- [ ] Cross-validation

**Day 5-7:** Mock Interviews
- [ ] 3-5 complete mock interviews
- [ ] Time yourself (45 mins per problem)
- [ ] Practice thinking aloud
- [ ] Review and improve

---

## Must-Know Algorithms

### 1. Linear Regression - Complete Implementation

```python
class LinearRegression:
    def __init__(self):
        self.theta = None

    def fit(self, X, y):
        # Add intercept
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        # Normal equation: θ = (X^T X)^(-1) X^T y
        self.theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        return self

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.theta
```

**Key Discussion Points:**
- Normal equation: O(n³) due to matrix inversion
- Gradient descent: O(m×n×iterations), better for large n
- When X^T X is singular, use regularization or SVD

### 2. Logistic Regression - Complete Implementation

```python
class LogisticRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.theta = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        m, n = X_b.shape
        self.theta = np.zeros(n)

        for _ in range(self.n_iters):
            h = self.sigmoid(X_b @ self.theta)
            gradient = (1/m) * X_b.T @ (h - y)
            self.theta -= self.lr * gradient

        return self

    def predict(self, X, threshold=0.5):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return (self.sigmoid(X_b @ self.theta) >= threshold).astype(int)
```

**Key Discussion Points:**
- Why cross-entropy instead of MSE?
- Handling class imbalance
- Numerical stability with sigmoid

### 3. K-Nearest Neighbors - Optimized Implementation

```python
class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X):
        # Vectorized distance computation
        X_train_sq = np.sum(self.X_train ** 2, axis=1)
        X_test_sq = np.sum(X ** 2, axis=1)[:, np.newaxis]
        cross_term = -2 * X @ self.X_train.T
        distances = np.sqrt(X_test_sq + X_train_sq + cross_term)

        # Get k nearest neighbors
        k_indices = np.argpartition(distances, self.k, axis=1)[:, :self.k]
        k_labels = self.y_train[k_indices]

        # Majority vote
        return np.array([np.bincount(labels).argmax() for labels in k_labels])
```

**Key Discussion Points:**
- Time complexity: O(n×d) per query
- How to speed up: KD-trees, Ball trees, LSH
- Curse of dimensionality

### 4. K-Means - Complete Implementation

```python
class KMeans:
    def __init__(self, n_clusters=3, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.centroids = None

    def fit(self, X):
        # K-Means++ initialization
        self.centroids = self._kmeans_plus_plus(X)

        for _ in range(self.max_iters):
            # Assignment step
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)

            # Update step
            old_centroids = self.centroids.copy()
            for k in range(self.n_clusters):
                if np.sum(labels == k) > 0:
                    self.centroids[k] = X[labels == k].mean(axis=0)

            # Check convergence
            if np.allclose(self.centroids, old_centroids):
                break

        return self

    def _kmeans_plus_plus(self, X):
        centroids = [X[np.random.randint(len(X))]]
        for _ in range(1, self.n_clusters):
            distances = np.array([min(np.linalg.norm(x - c) ** 2 for c in centroids) for x in X])
            probs = distances / distances.sum()
            centroids.append(X[np.random.choice(len(X), p=probs)])
        return np.array(centroids)
```

**Key Discussion Points:**
- K-Means++ vs random initialization
- How to choose k (elbow method, silhouette)
- Limitations (spherical clusters, sensitive to outliers)

---

## Common Interview Questions

### Category 1: Algorithm Implementation (60%)

**Q1: "Implement linear regression using gradient descent. Include feature scaling."**

**Approach:**
1. Implement StandardScaler
2. Implement gradient descent with MSE loss
3. Track cost history for convergence
4. Test on toy dataset

**Q2: "Implement logistic regression from scratch. Handle numerical stability."**

**Approach:**
1. Sigmoid with clipping to prevent overflow
2. Binary cross-entropy with epsilon
3. Gradient descent
4. Discuss regularization

**Q3: "Implement K-Means clustering with K-Means++ initialization."**

**Approach:**
1. Start with K-Means++ for better initialization
2. Lloyd's algorithm (assign → update)
3. Handle empty clusters
4. Convergence check

**Q4: "Implement k-NN classifier. Optimize for large datasets."**

**Approach:**
1. Vectorized distance computation
2. Use argpartition instead of argsort
3. Discuss KD-trees for further optimization

### Category 2: Neural Network Components (20%)

**Q5: "Implement ReLU activation with forward and backward pass."**

**Q6: "Implement softmax with numerical stability."**

**Q7: "Implement cross-entropy loss combined with softmax."**

**Q8: "Implement Adam optimizer from scratch."**

### Category 3: Data Processing (10%)

**Q9: "Implement train-test split with stratification."**

**Q10: "Implement StandardScaler (fit and transform)."**

**Q11: "Implement data collator for variable-length sequences."**

### Category 4: Metrics (10%)

**Q12: "Implement precision, recall, F1 score from scratch."**

**Q13: "Implement AUC-ROC calculation from scratch."**

**Q14: "Implement k-fold cross-validation."**

---

## Coding Patterns

### Pattern 1: Matrix Operations

**Problem:** Linear layer forward pass

```python
# Efficient matrix multiplication
def linear_forward(X, W, b):
    # X: (batch, in_features)
    # W: (out_features, in_features)
    # b: (out_features, 1)
    return (W @ X.T).T + b.T  # (batch, out_features)
```

**Key:** Understand broadcasting and transpose operations

### Pattern 2: Numerical Stability

**Problem:** Softmax overflow

```python
def softmax_stable(x):
    # Subtract max to prevent overflow
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
```

**Key:** Always subtract max before exp

### Pattern 3: Vectorization

**Problem:** Computing pairwise distances

```python
# Instead of loops
def pairwise_distances_slow(X, Y):
    distances = np.zeros((len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            distances[i, j] = np.linalg.norm(X[i] - Y[j])
    return distances

# Use vectorization
def pairwise_distances_fast(X, Y):
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2x·y
    X_sq = np.sum(X ** 2, axis=1)[:, np.newaxis]
    Y_sq = np.sum(Y ** 2, axis=1)
    return np.sqrt(X_sq + Y_sq - 2 * X @ Y.T)
```

**Key:** Use matrix operations instead of loops (10-100x faster)

### Pattern 4: Gradient Computation

**Problem:** Backpropagation through layers

```python
def layer_backward(grad_output, x, W):
    # grad_output: gradient from next layer
    # x: cached input from forward pass
    # W: weights

    batch_size = x.shape[0]

    # Gradient w.r.t. weights
    dW = (grad_output.T @ x) / batch_size

    # Gradient w.r.t. bias
    db = np.sum(grad_output, axis=0, keepdims=True) / batch_size

    # Gradient w.r.t. input (pass to previous layer)
    dx = grad_output @ W

    return dx, dW, db
```

**Key:** Cache forward pass values for backward pass

### Pattern 5: Iterative Optimization

**Problem:** Gradient descent convergence

```python
def gradient_descent_with_early_stopping(
    X, y, learning_rate=0.01, max_iters=1000, tolerance=1e-6
):
    theta = np.zeros(X.shape[1])
    prev_cost = float('inf')

    for iteration in range(max_iters):
        # Compute gradient
        predictions = X @ theta
        errors = predictions - y
        gradient = X.T @ errors / len(y)

        # Update
        theta -= learning_rate * gradient

        # Compute cost
        cost = np.mean(errors ** 2)

        # Early stopping
        if abs(prev_cost - cost) < tolerance:
            print(f"Converged at iteration {iteration}")
            break

        prev_cost = cost

    return theta
```

**Key:** Implement early stopping to avoid unnecessary iterations

---

## Interview Tips

### Before the Interview

**1. Review Core Algorithms (Week before)**
- [ ] Can implement all 4 core algorithms in < 20 minutes each
- [ ] Understand time/space complexity
- [ ] Know when to use each algorithm

**2. Practice Coding Without IDE (3 days before)**
- [ ] Use online judges (LeetCode, HackerRank)
- [ ] Practice on whiteboard or paper
- [ ] Get comfortable with syntax errors

**3. Prepare Questions to Ask (1 day before)**
- [ ] About allowed libraries
- [ ] Input/output format
- [ ] Performance requirements
- [ ] Edge cases to handle

### During the Interview

**1. Clarify Requirements (First 5 minutes)**
```
Example questions to ask:
- "Can I use NumPy? What about SciPy?"
- "What's the expected input format - is it a NumPy array or Python list?"
- "Are there any constraints on time/space complexity?"
- "Should I handle missing values or can I assume clean data?"
- "What should the function return - predictions or probabilities?"
```

**2. Explain Approach (Next 5-10 minutes)**
```
Template:
1. "I'm going to implement [algorithm name]"
2. "The approach is: [high-level steps]"
3. "Time complexity will be O(...), space complexity O(...)"
4. "Potential issues: [edge cases, numerical stability]"
5. "Let me know if this sounds good or if you'd like a different approach"
```

**3. Write Clean Code (Next 20-30 minutes)**

**Best Practices:**
```python
# Good: Clear variable names
def linear_regression_gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    m, n = X.shape
    theta = np.zeros(n)

    for iteration in range(n_iterations):
        predictions = X @ theta
        errors = predictions - y
        gradient = (1/m) * X.T @ errors
        theta -= learning_rate * gradient

    return theta

# Bad: Unclear names
def lr_gd(X, y, lr=0.01, n=1000):
    m, n = X.shape
    t = np.zeros(n)
    for i in range(n):
        p = X @ t
        e = p - y
        g = (1/m) * X.T @ e
        t -= lr * g
    return t
```

**4. Test Your Code (Last 5-10 minutes)**

```python
# Create simple test case
X_test = np.array([[1, 2], [3, 4], [5, 6]])
y_test = np.array([5, 11, 17])  # y = 2*x1 + 1*x2

model = LinearRegression()
model.fit(X_test, y_test)
predictions = model.predict(X_test)

print(f"Predictions: {predictions}")
print(f"True values: {y_test}")
print(f"Error: {np.mean((predictions - y_test) ** 2):.4f}")

# Walk through one example
print("\nManual check:")
print(f"For X=[1, 2], prediction = {predictions[0]:.2f}, expected = 5")
```

**5. Discuss Extensions (If time remains)**

```
Example extensions:
- "To handle large datasets, I would implement mini-batch processing"
- "For regularization, I would add L2 penalty: lambda * ||theta||^2"
- "To improve convergence, I could add learning rate scheduling"
- "For production, I would add input validation and error handling"
```

### Common Mistakes to Avoid

**1. Not Asking Clarifying Questions**
❌ Jump straight to coding
✅ Ask about constraints, format, edge cases

**2. Writing Inefficient Code**
❌ Use nested loops for matrix operations
✅ Vectorize with NumPy operations

**3. Ignoring Numerical Stability**
❌ `np.exp(large_number)` → overflow
✅ Subtract max before exp, clip values

**4. No Error Handling**
❌ Assume perfect inputs
✅ Check shapes, handle edge cases

**5. Not Testing**
❌ Assume code works
✅ Test with simple example

**6. Poor Variable Names**
❌ `a, b, c, tmp, x1, x2`
✅ `predictions, errors, gradient, learning_rate`

---

## Resources Created

### 1. ML Algorithms From Scratch (`ml-algorithms-from-scratch.ipynb`)

**What's Included:**
- ✅ Linear Regression (Normal Equation + Gradient Descent)
- ✅ Logistic Regression (with numerical stability)
- ✅ K-Nearest Neighbors (basic + optimized vectorized version)
- ✅ K-Means Clustering (Lloyd's algorithm + K-Means++)
- ✅ Supporting functions (train-test split, feature scaling, cross-validation)

**When to Use:**
- Primary resource for algorithm implementation
- Reference during practice
- Day 1-10 of preparation

### 2. Neural Network Components From Scratch (`neural-network-components-from-scratch.ipynb`)

**What's Included:**
- ✅ Linear (Dense) Layer - forward/backward pass
- ✅ Activation Functions - ReLU, Sigmoid, Tanh, Softmax
- ✅ Loss Functions - MSE, BCE, Cross-Entropy
- ✅ Optimizers - SGD, Momentum, Adam
- ✅ Complete Neural Network - end-to-end implementation

**When to Use:**
- Deep learning specific questions
- Transformer/attention mechanism components
- Week 3 of preparation

### 3. ML Coding Problems (`ml-coding-problems.ipynb`)

**What's Included:**
- ✅ 7 practice problems with solutions
- ✅ Attention mechanism
- ✅ Cosine similarity
- ✅ Data collation for fine-tuning
- ✅ Gradient descent variants
- ✅ Confusion matrix and metrics

**When to Use:**
- Cisco-specific preparation (has attention mechanism)
- Practice problems
- Week 2-4 of preparation

---

## Final Checklist

### 1 Week Before Interview

- [ ] Can implement Linear Regression in < 15 minutes
- [ ] Can implement Logistic Regression in < 20 minutes
- [ ] Can implement k-NN in < 15 minutes
- [ ] Can implement K-Means in < 25 minutes
- [ ] Know all activation functions and their derivatives
- [ ] Can implement softmax with numerical stability
- [ ] Understand backpropagation conceptually

### 3 Days Before Interview

- [ ] Solved 10+ practice problems end-to-end
- [ ] Practiced explaining approach aloud
- [ ] Comfortable with NumPy syntax
- [ ] Know common edge cases
- [ ] Prepared questions to ask interviewer

### 1 Day Before Interview

- [ ] Review time/space complexities
- [ ] Review common pitfalls (numerical stability, vectorization)
- [ ] Get good sleep
- [ ] Prepare environment (quiet space, stable internet)

### Day of Interview

- [ ] Arrive 10 minutes early
- [ ] Have pen and paper ready
- [ ] Test microphone and camera
- [ ] Stay calm and think aloud
- [ ] Ask clarifying questions
- [ ] Test your code with examples

---

## Conclusion

**You're Ready If You Can:**

1. ✅ Implement all 4 core ML algorithms from scratch in < 25 minutes each
2. ✅ Implement neural network components (layers, activations, loss, optimizers)
3. ✅ Write vectorized NumPy code (no nested loops for matrix operations)
4. ✅ Handle numerical stability issues (softmax, sigmoid)
5. ✅ Explain time/space complexity
6. ✅ Test your code with simple examples
7. ✅ Discuss trade-offs and optimizations

**Your Advantage:**

- ✅ 3 comprehensive notebooks with 20+ implementations
- ✅ Practice problems with solutions
- ✅ Real interview questions from 2025
- ✅ Step-by-step preparation timeline
- ✅ Interview tips and common patterns

**Remember:**
- Companies want to see **problem-solving**, not memorization
- **Think aloud** - explain your approach
- **Ask questions** - clarify requirements
- **Test your code** - walk through examples
- **Stay calm** - you've practiced this!

---

**Good Luck! 🚀**

You've got this! Your preparation is thorough, your implementations are solid, and you understand the concepts deeply. Go ace that Cisco ML Engineer coding round!

