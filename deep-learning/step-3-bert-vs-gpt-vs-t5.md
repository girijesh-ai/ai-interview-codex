# 🎨 Visual Comparison: BERT vs GPT vs T5

**A side-by-side visual guide to understand transformer architecture variants**

---

## 📊 Quick Comparison Table

| Feature | BERT (Encoder-Only) | GPT (Decoder-Only) | T5 (Encoder-Decoder) |
|---------|-------------------|-------------------|---------------------|
| **Architecture** | Stacked Encoders | Stacked Decoders | Encoder + Decoder |
| **Attention** | ✅ Bidirectional | ⏩ Causal (Left-to-right) | Both |
| **Training** | MLM (Masked tokens) | CLM (Next token) | Span corruption |
| **Generation** | ❌ No | ✅ Yes | ✅ Yes |
| **Understanding** | ⭐⭐⭐ Excellent | ⭐⭐ Good | ⭐⭐⭐ Excellent |
| **Best For** | Classification, embeddings | Text generation, chat | Translation, seq2seq |
| **2025 Popularity** | 📉 Declining | 📈 Dominant | 📊 Moderate |
| **Examples** | BERT, RoBERTa | GPT-4, LLaMA, Claude | T5, BART |

---

## 🔍 Attention Pattern Visualization

### BERT (Bidirectional Attention)

```
Input: "The cat sat on the mat"

Attention Pattern (each token sees ALL tokens):
        The  cat  sat  on   the  mat
The   [ ■    ■    ■    ■    ■    ■  ]  → Can see everything
cat   [ ■    ■    ■    ■    ■    ■  ]  → Can see everything  
sat   [ ■    ■    ■    ■    ■    ■  ]  → Can see everything
on    [ ■    ■    ■    ■    ■    ■  ]  → Can see everything
the   [ ■    ■    ■    ■    ■    ■  ]  → Can see everything
mat   [ ■    ■    ■    ■    ■    ■  ]  → Can see everything

✅ Advantage: Rich contextual understanding
❌ Disadvantage: Cannot generate text naturally
```

### GPT (Causal Attention)

```
Input: "The cat sat on the mat"

Attention Pattern (each token sees ONLY previous tokens):
        The  cat  sat  on   the  mat
The   [ ■    □    □    □    □    □  ]  → Sees only "The"
cat   [ ■    ■    □    □    □    □  ]  → Sees "The cat"
sat   [ ■    ■    ■    □    □    □  ]  → Sees "The cat sat"
on    [ ■    ■    ■    ■    □    □  ]  → Sees "The cat sat on"
the   [ ■    ■    ■    ■    ■    □  ]  → Sees "The cat sat on the"
mat   [ ■    ■    ■    ■    ■    ■  ]  → Sees everything before

■ = Can attend    □ = Masked (cannot attend)

✅ Advantage: Perfect for text generation
✅ Advantage: Simple, scalable architecture
❌ Disadvantage: Less rich context than bidirectional
```

### T5 (Encoder: Bidirectional, Decoder: Causal + Cross-Attention)

```
Encoder (Bidirectional):
Input: "translate English to German: Hello"
        translate  English  to  German  :  Hello
translate  [ ■         ■      ■     ■     ■    ■  ]
English    [ ■         ■      ■     ■     ■    ■  ]
...

Decoder (Causal + Cross-Attention to Encoder):
Output: "Hallo Welt"
        Hallo  Welt
Hallo [ ■      □  ]  + Cross-attention to ALL encoder outputs
Welt  [ ■      ■  ]  + Cross-attention to ALL encoder outputs

✅ Advantage: Best of both worlds
❌ Disadvantage: More complex, harder to scale
```

---

## 🎯 Training Objectives Explained

### BERT: Masked Language Modeling (MLM)

```
Original:  "The cat sat on the mat"
Masked:    "The [MASK] sat on the [MASK]"
Target:    Predict "cat" and "mat"

How it works:
1. Randomly mask 15% of tokens
2. Model predicts masked tokens using bidirectional context
3. Loss = CrossEntropy(predicted, true_masked_tokens)

Example:
Input:     "The [MASK] is sleeping"
Context:   Can see "The", "is", "sleeping" (all directions)
Predict:   "cat" (most likely), "dog", "baby", etc.

Why bidirectional?
"The cat is" → "cat" could be anything
"is sleeping" → likely an animate being
Together →  probably "cat", "dog", "baby"
```

**Real training statistics:**
- 15% of tokens are selected for masking:
  - 80% replaced with [MASK]
  - 10% replaced with random token (makes model robust)
  - 10% kept unchanged (reduces train/test mismatch)

### GPT: Causal Language Modeling (CLM)

```
Input sequence:  "The cat sat on the"
Model predicts:  "cat sat on the mat"
                  ↑   ↑   ↑   ↑   ↑
Each position predicts the NEXT token

How it works:
1. Feed sequence left-to-right
2. At each position, predict next token
3. Loss = Σ CrossEntropy(predicted_i, actual_i+1)

Example:
Input:  "The cat sat on"
        ↓   ↓   ↓   ↓
Predict:"cat sat on the"

Position 0: "The"           → Predict "cat"
Position 1: "The cat"       → Predict "sat"
Position 2: "The cat sat"   → Predict "on"
Position 3: "The cat sat on"→ Predict "the"

All predictions happen in parallel during training!
```

**Why this works for generation:**
```
At inference:
Start:  "The"
Gen 1:  "The cat"      (model predicted "cat")
Gen 2:  "The cat sat"  (model predicted "sat")
Gen 3:  "The cat sat on" (model predicted "on")
...
```

### T5: Span Corruption

```
Original: "The cat sat on the mat in the sun"
Corrupt:  "The cat <X> on <Y> mat <Z> sun"
Target:   "<X> sat <Y> the <Z> in the <eos>"

How it works:
1. Mask random SPANS (not individual tokens)
2. Replace with sentinel tokens <X>, <Y>, <Z>
3. Model predicts masked spans in order

Why better than MLM?
- More realistic (phrases get masked, not random words)
- Learns to generate multiple tokens
- Works for seq2seq tasks naturally
```

---

## 🏗️ Architecture Deep Dive

### BERT Encoder Block

```
Input (e.g., "cat")
       ↓
┌──────────────────────┐
│  Input Embedding     │  768-dim vector
│  + Position Embed    │
└──────────────────────┘
       ↓
┌──────────────────────┐
│ Multi-Head Attention │  12 heads × 64 dim = 768
│ (Bidirectional)      │  Attends to ALL tokens
└──────────────────────┘
       ↓
    Add & Norm           Residual connection
       ↓
┌──────────────────────┐
│  Feed-Forward (FFN)  │  768 → 3072 → 768
│  ReLU                │  Position-wise
└──────────────────────┘
       ↓
    Add & Norm           Residual connection
       ↓
    Output (768-dim)
```

**Layer organization:**
- BERT-base: 12 encoder blocks
- BERT-large: 24 encoder blocks

**When to use:**
- Sentence classification (spam detection, sentiment)
- Named Entity Recognition (NER)
- Question answering (when answer is in context)
- Sentence similarity/embeddings

### GPT Decoder Block

```
Input (e.g., "cat")
       ↓
┌──────────────────────┐
│  Input Embedding     │  
│  + Position Embed    │  (RoPE in modern GPT)
└──────────────────────┘
       ↓
┌──────────────────────┐
│ Masked Self-Attention│  Causal mask applied
│ (Causal)             │  Only sees ← previous
└──────────────────────┘
       ↓
    Add & Norm           Pre-LN in modern GPT
       ↓
┌──────────────────────┐
│  Feed-Forward (FFN)  │  Often SwiGLU in 2025
│  4x expansion        │  (e.g., 4096→16384→4096)
└──────────────────────┘
       ↓
    Add & Norm
       ↓
    Output
```

**Key difference from BERT:**
- ❌ NO cross-attention (not needed for language modeling)
- ✅ Causal masking in self-attention
- ✅ Optimized for generation

**When to use:**
- Text generation (stories, articles)
- Chat/dialogue (ChatGPT, Claude)
- Code generation (Copilot)
- Instruction following

### T5 Full Architecture

```
ENCODER SIDE:
Input: "translate: Hello"
       ↓
┌──────────────────────┐
│  Encoder Block  1    │  Bidirectional
│  Encoder Block  2    │  attention
│  ...                 │
│  Encoder Block 12    │
└──────────────────────┘
       ↓
  Encoder Output (context)
       ↓
       ↓ (fed to decoder via cross-attention)
       ↓
DECODER SIDE:
Input: "<start> Hallo"
       ↓
┌──────────────────────┐
│ Masked Self-Attention│  Causal
└──────────────────────┘
       ↓
┌──────────────────────┐
│   Cross-Attention    │  Attends to encoder output
│   Q: from decoder    │  K,V: from encoder
└──────────────────────┘
       ↓
┌──────────────────────┐
│  Feed-Forward (FFN)  │
└──────────────────────┘
       ↓
    Output: "Hallo"
```

**When to use:**
- Machine translation (En→De, etc.)
- Summarization (long → short)
- Question answering (generate answer)
- Any task that's truly seq2seq

---

## 💡 Key Insights for Each Architecture

### BERT Insights

```python
# Why BERT is great for classification:
sentence = "This movie was terrible!"

# BERT sees the ENTIRE sentence bidirectionally
# "terrible" affects understanding of "This movie was"
# "This movie was" affects understanding of "terrible"

# Output: [CLS] token = sentence representation
cls_token = bert(sentence)[0]  # Rich, contextual embedding
classifier = linear(cls_token)  # Simple classifier on top
```

**Limitation:**
```python
# BERT cannot naturally do this:
prompt = "Once upon a time"
next_word = bert.generate()  # ❌ Not designed for this!

# Why? Because BERT sees EVERYTHING at once.
# It's not trained to predict what comes NEXT.
```

### GPT Insights

```python
# GPT predicts one token at a time
prompt = "The capital of France is"

# Step 1: "The"                      → predict "capital"
# Step 2: "The capital"              → predict "of"  
# Step 3: "The capital of"           → predict "France"
# Step 4: "The capital of France"    → predict "is"
# Step 5: "The capital of France is" → predict "Paris"

output = gpt.generate(prompt)
# Output: "The capital of France is Paris"
```

**Why it scales:**
```
Simple architecture → Easy to scale → 175B (GPT-3), 1.7T (GPT-4) parameters
Unified objective → Works for everything with prompting
```

### T5 Insights

```python
# T5 frames everything as text-to-text

# Translation
input = "translate English to German: Hello"
output = t5(input)  # "Hallo"

# Summarization
input = "summarize: [long article]"
output = t5(input)  # "[summary]"

# Classification (weird but works!)
input = "sentiment: This movie is great!"
output = t5(input)  # "positive"

# Why? Encoder understands input deeply (bidirectional)
#       Decoder generates output flexibly
```

**Trade-off:**
- ✅ Task flexibility
- ✅ Excellent understanding + generation
- ❌ More parameters for same capability
- ❌ Harder to scale to 100B+ parameters

---

## 🎯 Decision Tree: Which Architecture?

```
Your Task:
│
├─ Need to GENERATE text?
│  │
│  ├─ Yes → Need to understand INPUT deeply first?
│  │  │
│  │  ├─ Yes (Translation, Summarization)
│  │  │  └─ Use: T5 / Encoder-Decoder
│  │  │
│  │  └─ No (Chat, Creative writing, Code gen)
│  │     └─ Use: GPT / Decoder-Only ⭐ (2025 standard)
│  │
│  └─ No → Just classify/embed/extract?
│     └─ Use: BERT / Encoder-Only
│
└─ Special cases:
   - Embeddings for search: BERT-based (e.g., sentence-transformers)
   - Long-form generation: GPT
   - Multi-task learning: T5 or GPT with prompting
```

---

## 📈 Evolution Timeline

```
2017: Original Transformer (Encoder-Decoder)
      └─ "Attention is All You Need"

2018: BERT (Encoder-Only)
      └─ State-of-art on 11 NLP tasks
      └─ Everyone uses BERT for everything

2018-2019: GPT-1, GPT-2 (Decoder-Only)
           └─ "Interesting but niche"

2019: T5 (Encoder-Decoder, unified framework)
      └─ Text-to-text revolution

2020: GPT-3 (175B parameters)
      └─ In-context learning discovered!
      └─ Paradigm shift: Decoder-only dominates

2023-2025: LLaMA, GPT-4, Claude, Qwen
           └─ Only decoder-only at scale
           └─ BERT/T5 for specialized tasks only

Why Decoder-Only Won?
  ✓ Simpler architecture
  ✓ Scales better (proven to 1T+ params)
  ✓ Emergent abilities at scale
  ✓ Prompting solves most tasks
  ✓ Chat/instruction following natural fit
```

---

## 🧪 Hands-On: Experience the Differences

### Try with Hugging Face

```python
from transformers import pipeline

# BERT (Encoder-Only): Fill in the blank
fill_mask = pipeline("fill-mask", model="bert-base-uncased")
result = fill_mask("The capital of France is [MASK].")
print(result)
# Output: [{'token_str': 'paris', 'score': 0.9}]

# GPT (Decoder-Only): Text generation  
generator = pipeline("text-generation", model="gpt2")
result = generator("The capital of France is", max_length=10)
print(result)
# Output: [{'generated_text': 'The capital of France is Paris'}]

# T5 (Encoder-Decoder): Translation
translator = pipeline("translation_en_to_de", model="t5-small")
result = translator("The capital of France is Paris")
print(result)
# Output: [{'translation_text': 'Die Hauptstadt Frankreichs ist Paris'}]
```

---

## Summary: The Bottom Line

### BERT (Encoder-Only)
- **Best at:** Understanding text, embeddings, classification
- **Cannot:** Generate text naturally
- **Use when:** You need rich bidirectional context
- **2025 status:** Still used for embeddings/classification, but declining

### GPT (Decoder-Only)
- **Best at:** Generating text, chat, anything with prompting
- **Decent at:** Understanding (especially with large scale)
- **Use when:** General purpose, generation, chat
- **2025 status:** 🏆 Dominant architecture

### T5 (Encoder-Decoder)
- **Best at:** Seq2seq tasks (translation, summarization)
- **Good at:** Both understanding and generation
- **Use when:** True input→output transformation needed
- **2025 status:** Niche use cases, not scaling like decoder-only

**The 2025 Reality:**
> "Decoder-only transformers (GPT-style) have become the default choice. They're simpler, scale better, and with sufficient size and prompting, match or exceed specialized architectures on most tasks."

---

## 🎓 Next Steps

1. **Implement attention**: Do the hands-on exercises
2. **Run examples**: Try BERT vs GPT on same task
3. **Visualize attention**: Use tools like BertViz
4. **Read papers**: 
   - BERT: "BERT: Pre-training of Deep Bidirectional Transformers"
   - GPT: "Language Models are Few-Shot Learners" (GPT-3)
   - T5: "Exploring the Limits of Transfer Learning"

**Remember:** Architecture matters less than you think. Scale, data, and training matter more! 🚀
