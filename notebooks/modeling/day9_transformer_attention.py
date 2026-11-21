"""
Day 9 – Transformer Architecture & Self-Attention Demo
=======================================================
Topics covered:
  - Tokenization concepts
  - Embeddings (toy numerical representations)
  - Positional encoding (additive sinusoidal)
  - Scaled dot-product Self-Attention
  - Causal masking (decoder-only / GPT style)
  - Softmax normalization
  - Multi-head attention concept
  - KV Cache explanation

Domain: Investment transaction instruction processing
        Tokens: investment order words from portfolio manager emails.

NOTE: These are educational toy implementations using NumPy.
      They demonstrate the mathematical mechanics of attention,
      not a production-grade or trained language model.
      Requires: pip install numpy
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

SEPARATOR = "─" * 60


# ===========================================================================
# Part 1 – Tokenization
# ===========================================================================

print(SEPARATOR)
print("PART 1 – Tokenization")
print(SEPARATOR)

# Simplified tokenization of an investment instruction sentence
sentence = "BUY 500 ABC P1001"
tokens = sentence.split()          # real tokenizers use subword units (BPE)

# Toy token-ID lookup table
TOKEN_VOCAB: dict[str, int] = {
    "BUY": 101, "SELL": 102, "500": 200, "200": 201,
    "ABC": 301, "XYZ": 302, "P1001": 401, "P1002": 402,
}
token_ids = [TOKEN_VOCAB[t] for t in tokens]

print(f"Sentence   : {sentence}")
print(f"Tokens     : {tokens}")
print(f"Token IDs  : {token_ids}")
print("(IDs are illustrative – not real GPT tokenizer values)\n")


# ===========================================================================
# Part 2 – Toy Embeddings
# ===========================================================================

print(SEPARATOR)
print("PART 2 – Embeddings")
print(SEPARATOR)

# 4 tokens, each represented by a 4-dimensional vector.
# In a real LLM these are learned during pre-training.
X = np.array([
    [1.0, 0.0, 0.5, 0.2],   # BUY
    [0.0, 1.0, 0.3, 0.1],   # 500
    [1.0, 1.0, 0.8, 0.6],   # ABC
    [0.5, 0.5, 0.4, 0.9],   # P1001
], dtype=np.float64)

print("Token embeddings (shape = 4 tokens × 4 dimensions):")
for token, vec in zip(tokens, X):
    print(f"  {token:<8} → {vec}")
print()


# ===========================================================================
# Part 3 – Positional Encoding (sinusoidal)
# ===========================================================================

print(SEPARATOR)
print("PART 3 – Positional Encoding")
print(SEPARATOR)


def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """
    Standard sinusoidal positional encoding from 'Attention Is All You Need'.
    Returns shape (seq_len, d_model).
    """
    pe = np.zeros((seq_len, d_model))
    positions = np.arange(seq_len)[:, np.newaxis]       # (seq_len, 1)
    dims = np.arange(0, d_model, 2)                     # even dimensions

    pe[:, 0::2] = np.sin(positions / (10_000 ** (dims / d_model)))
    pe[:, 1::2] = np.cos(positions / (10_000 ** (dims / d_model)))
    return pe


seq_len, d_model = X.shape
PE = sinusoidal_positional_encoding(seq_len, d_model)
X_pe = X + PE  # add positional information to embeddings

print("Positional encoding (sinusoidal):")
print(PE)
print()
print("Embeddings + positional encoding (input to attention):")
print(X_pe)
print()
print("Why needed: 'BUY 500' and '500 BUY' contain the same tokens")
print("but position encoding gives the model sequence-order information.\n")


# ===========================================================================
# Part 4 – Scaled Dot-Product Self-Attention
# ===========================================================================

print(SEPARATOR)
print("PART 4 – Scaled Dot-Product Self-Attention")
print(SEPARATOR)


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    causal_mask: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Standard attention: Softmax( Q K^T / sqrt(d_k) ) V

    Args:
        Q: Query matrix  (seq_len, d_k)
        K: Key matrix    (seq_len, d_k)
        V: Value matrix  (seq_len, d_v)
        causal_mask: if True, prevent attending to future tokens

    Returns:
        output: context-aware representations (seq_len, d_v)
        weights: attention weight matrix       (seq_len, seq_len)
    """
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)          # shape (seq_len, seq_len)

    if causal_mask:
        # Set upper-triangle (future positions) to -inf before softmax
        seq = scores.shape[0]
        mask = np.triu(np.ones((seq, seq)), k=1) * -1e9
        scores = scores + mask

    # Numerically stable softmax
    scores_shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores_shifted)
    weights = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-9)

    output = weights @ V
    return output, weights


# --- Learned projection matrices (fixed for this demo; trained in real LLMs)
d_input = X_pe.shape[1]  # 4

W_Q = np.array([
    [1.0, 0.0, 0.5, 0.2],
    [0.5, 1.0, 0.3, 0.1],
    [0.2, 0.3, 1.0, 0.4],
    [0.1, 0.2, 0.4, 1.0],
], dtype=np.float64)

W_K = np.array([
    [1.0, 0.5, 0.2, 0.1],
    [0.0, 1.0, 0.3, 0.2],
    [0.5, 0.3, 1.0, 0.4],
    [0.2, 0.1, 0.4, 1.0],
], dtype=np.float64)

W_V = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)  # Identity for V: output = weighted input

Q_mat = X_pe @ W_Q
K_mat = X_pe @ W_K
V_mat = X_pe @ W_V

print("Query (Q):")
print(Q_mat)
print()
print("Key (K):")
print(K_mat)
print()
print("Value (V):")
print(V_mat)
print()

# --- Self-attention with causal masking (decoder / GPT style)
attn_output, attn_weights = scaled_dot_product_attention(
    Q_mat, K_mat, V_mat, causal_mask=True
)

print("Attention weights (with causal mask):")
print("  Rows = query position, Cols = key position")
print("  Zero upper-triangle means future tokens are masked.")
print()
print(attn_weights)
print()

print("Attention weight interpretation (causal):")
for i, token in enumerate(tokens):
    visible = tokens[: i + 1]
    dominant_idx = np.argmax(attn_weights[i])
    print(
        f"  Token '{token}' attends most to '{tokens[dominant_idx]}' "
        f"(visible context: {visible})"
    )
print()

print("Self-attention output (context-aware token representations):")
print(attn_output)
print()


# ===========================================================================
# Part 5 – Without causal mask (encoder style / BERT)
# ===========================================================================

print(SEPARATOR)
print("PART 5 – Attention WITHOUT causal mask (encoder / BERT style)")
print(SEPARATOR)

attn_output_full, attn_weights_full = scaled_dot_product_attention(
    Q_mat, K_mat, V_mat, causal_mask=False
)

print("Attention weights (bidirectional – no mask):")
print(attn_weights_full)
print()
print("Each token can now attend to ALL other tokens.")
print("Useful for classification/understanding tasks (not text generation).\n")


# ===========================================================================
# Part 6 – Multi-Head Attention (concept)
# ===========================================================================

print(SEPARATOR)
print("PART 6 – Multi-Head Attention (2 heads)")
print(SEPARATOR)


def multi_head_attention(
    X: np.ndarray, n_heads: int = 2, causal_mask: bool = True
) -> np.ndarray:
    """
    Simplified Multi-Head Attention.
    Splits d_model evenly across heads; each head runs independent attention.
    """
    seq_len, d_model = X.shape
    d_head = d_model // n_heads
    head_outputs = []

    for head in range(n_heads):
        # Each head uses a different slice of the feature space
        # (in real LLMs these are distinct learned projection matrices)
        rng = np.random.default_rng(seed=head)
        Wq = rng.standard_normal((d_model, d_head)) * 0.5
        Wk = rng.standard_normal((d_model, d_head)) * 0.5
        Wv = rng.standard_normal((d_model, d_head)) * 0.5

        Qh = X @ Wq
        Kh = X @ Wk
        Vh = X @ Wv

        out, w = scaled_dot_product_attention(Qh, Kh, Vh, causal_mask=causal_mask)
        head_outputs.append(out)
        print(f"  Head {head + 1} attention weights:")
        print(f"  {w}")
        print()

    # Concatenate head outputs
    combined = np.concatenate(head_outputs, axis=-1)
    return combined


mha_output = multi_head_attention(X_pe, n_heads=2, causal_mask=True)
print("Multi-head attention output (concatenated heads):")
print(mha_output)
print()
print("Each head can specialize in different relationships:")
print("  Head 1 might focus on 'transaction type ↔ quantity'")
print("  Head 2 might focus on 'security ↔ portfolio'\n")


# ===========================================================================
# Part 7 – Practical challenges (from Day 9 docs)
# ===========================================================================

print(SEPARATOR)
print("PART 7 – Practical challenges")
print(SEPARATOR)

# Challenge 1: change BUY to SELL
tokens_sell = ["SELL", "500", "ABC", "P1001"]
X_sell = np.array([
    [0.0, 1.0, 0.5, 0.2],   # SELL  (different embedding from BUY)
    [0.0, 1.0, 0.3, 0.1],
    [1.0, 1.0, 0.8, 0.6],
    [0.5, 0.5, 0.4, 0.9],
])
Q_sell = X_sell @ W_Q
K_sell = X_sell @ W_K
V_sell = X_sell @ W_V
_, w_sell = scaled_dot_product_attention(Q_sell, K_sell, V_sell)
print("Challenge 1 – SELL instead of BUY:")
print(f"Tokens: {tokens_sell}")
print("Attention weights:")
print(w_sell)
print()

# Challenge 2: change quantity 500 → 1000
X_qty = X_pe.copy()
X_qty[1] = [0.0, 2.0, 0.6, 0.2]   # higher quantity embedding
Q_qty = X_qty @ W_Q
K_qty = X_qty @ W_K
V_qty = X_qty @ W_V
_, w_qty = scaled_dot_product_attention(Q_qty, K_qty, V_qty)
print("Challenge 2 – Different quantity vector [0.0, 2.0, 0.6, 0.2]:")
print("Attention weights:")
print(w_qty)
print()

# Challenge 3: remove causal mask
_, w_no_mask = scaled_dot_product_attention(Q_mat, K_mat, V_mat, causal_mask=False)
print("Challenge 3 – No causal mask (tokens can see the full sequence):")
print("Attention weights:")
print(w_no_mask)
print()


# ===========================================================================
# Part 8 – KV Cache explanation
# ===========================================================================

print(SEPARATOR)
print("PART 8 – KV Cache concept")
print(SEPARATOR)

print("""
KV Cache (Key-Value Cache):

During autoregressive generation the model produces one token at a time.
At each step it needs K and V for all previously generated tokens.

Without KV Cache:
  Step 3 (generating "ABC"): recompute K,V for tokens [BUY, 500, ABC]
  Step 4 (generating "P1001"): recompute K,V for [BUY, 500, ABC, P1001]
  → O(n²) redundant computation

With KV Cache:
  Store K,V from each previous step.
  At each new step, only compute K,V for the NEW token.
  Append to the cache. Use cached K,V for attention over past tokens.
  → Significantly reduces inference cost for long sequences.

Relevance for investment emails:
  Longer portfolio manager emails produce more tokens.
  KV caching keeps per-token inference latency manageable at scale.
""")

print(SEPARATOR)
print("Day 9 complete. Key takeaways:")
print(SEPARATOR)
print("""
1. Tokenization converts text to numerical token IDs.
2. Embeddings map token IDs to learned vector representations.
3. Positional encoding adds sequence-order information.
4. Self-Attention (Q, K, V) lets each token relate to others.
5. Causal masking prevents decoder models from seeing future tokens.
6. Softmax turns raw scores into attention weights that sum to 1.
7. Multi-Head Attention runs several attention heads in parallel.
8. KV Cache avoids recomputing Keys/Values for earlier tokens.
""")
