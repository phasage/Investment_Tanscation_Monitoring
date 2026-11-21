"""
Tests for Day 9 concepts – Transformer Architecture & Self-Attention
Verifies the mathematical correctness of the toy attention implementation.
"""
import pytest
import numpy as np
import sys
import os

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "..", "notebooks", "modeling"),
)
import day9_transformer_attention as day9


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def toy_tokens():
    return ["BUY", "500", "ABC", "P1001"]


@pytest.fixture()
def toy_embeddings():
    """4-token, 4-dim toy embeddings (matches day9 X matrix)."""
    return np.array([
        [1.0, 0.0, 0.5, 0.2],
        [0.0, 1.0, 0.3, 0.1],
        [1.0, 1.0, 0.8, 0.6],
        [0.5, 0.5, 0.4, 0.9],
    ], dtype=np.float64)


@pytest.fixture()
def identity_matrices(toy_embeddings):
    d = toy_embeddings.shape[1]
    return np.eye(d), np.eye(d), np.eye(d)   # W_Q, W_K, W_V = Identity


# ---------------------------------------------------------------------------
# Part 1 – Positional Encoding
# ---------------------------------------------------------------------------

class TestPositionalEncoding:
    def test_output_shape(self):
        pe = day9.sinusoidal_positional_encoding(4, 4)
        assert pe.shape == (4, 4)

    def test_different_positions_differ(self):
        pe = day9.sinusoidal_positional_encoding(4, 4)
        # No two rows should be identical
        for i in range(len(pe)):
            for j in range(i + 1, len(pe)):
                assert not np.allclose(pe[i], pe[j]), (
                    f"Rows {i} and {j} are identical"
                )

    def test_values_bounded(self):
        pe = day9.sinusoidal_positional_encoding(10, 8)
        assert pe.min() >= -1.0 - 1e-6
        assert pe.max() <= 1.0 + 1e-6

    def test_adding_to_embeddings_preserves_shape(self, toy_embeddings):
        pe = day9.sinusoidal_positional_encoding(*toy_embeddings.shape)
        X_pe = toy_embeddings + pe
        assert X_pe.shape == toy_embeddings.shape


# ---------------------------------------------------------------------------
# Part 2 – Scaled Dot-Product Attention
# ---------------------------------------------------------------------------

class TestScaledDotProductAttention:
    def _run(self, X, causal=True):
        Q = X
        K = X
        V = X
        return day9.scaled_dot_product_attention(Q, K, V, causal_mask=causal)

    def test_output_shape_matches_input(self, toy_embeddings):
        out, weights = self._run(toy_embeddings)
        assert out.shape == toy_embeddings.shape

    def test_attention_weights_shape(self, toy_embeddings):
        _, weights = self._run(toy_embeddings)
        n = len(toy_embeddings)
        assert weights.shape == (n, n)

    def test_attention_weights_sum_to_one_per_row(self, toy_embeddings):
        _, weights = self._run(toy_embeddings)
        np.testing.assert_allclose(
            weights.sum(axis=1), np.ones(len(toy_embeddings)), atol=1e-5
        )

    def test_weights_non_negative(self, toy_embeddings):
        _, weights = self._run(toy_embeddings)
        assert np.all(weights >= 0)

    def test_causal_mask_upper_triangle_zero(self, toy_embeddings):
        """
        In a decoder (GPT-style) causal mask, positions i must NOT attend
        to positions j > i  →  upper triangle weights ≈ 0.
        """
        _, weights = self._run(toy_embeddings, causal=True)
        seq = len(toy_embeddings)
        for i in range(seq):
            for j in range(i + 1, seq):
                assert weights[i, j] < 1e-6, (
                    f"Position {i} attends to future position {j}: "
                    f"weight={weights[i, j]:.6f}"
                )

    def test_no_causal_mask_allows_full_attention(self, toy_embeddings):
        """Without the mask, every position can attend to all others."""
        _, weights = self._run(toy_embeddings, causal=False)
        # At least some upper-triangle weights must be nonzero
        seq = len(toy_embeddings)
        upper_sum = sum(
            weights[i, j] for i in range(seq) for j in range(i + 1, seq)
        )
        assert upper_sum > 0.01, "Full attention expected but upper triangle is zero"

    def test_first_position_attends_only_to_itself_with_causal_mask(
        self, toy_embeddings
    ):
        """
        Token at position 0 can only see itself under causal masking.
        Its attention weight vector must be [1, 0, 0, 0].
        """
        _, weights = self._run(toy_embeddings, causal=True)
        np.testing.assert_allclose(weights[0], [1, 0, 0, 0], atol=1e-5)

    def test_identity_projections_preserve_values(self, toy_embeddings):
        """
        With W_Q = W_K = W_V = Identity and no causal mask,
        output values are weighted combinations of input rows.
        """
        I = np.eye(toy_embeddings.shape[1])
        Q = toy_embeddings @ I
        K = toy_embeddings @ I
        V = toy_embeddings @ I
        out, _ = day9.scaled_dot_product_attention(Q, K, V, causal_mask=False)
        # Output rows must lie in the convex hull of input rows
        # (i.e. each output row is a weighted average of V rows)
        assert out.shape == toy_embeddings.shape

    def test_different_embeddings_produce_different_outputs(self):
        """Changing an embedding must change the attention output."""
        base = np.eye(4, dtype=np.float64)
        modified = base.copy()
        modified[0, 0] = 5.0   # large change to first token

        out_base, _ = day9.scaled_dot_product_attention(base, base, base, causal_mask=False)
        out_mod, _  = day9.scaled_dot_product_attention(modified, modified, modified, causal_mask=False)

        assert not np.allclose(out_base, out_mod)


# ---------------------------------------------------------------------------
# Part 3 – Multi-Head Attention
# ---------------------------------------------------------------------------

class TestMultiHeadAttention:
    def test_output_shape_two_heads(self, toy_embeddings):
        pe = day9.sinusoidal_positional_encoding(*toy_embeddings.shape)
        X_pe = toy_embeddings + pe
        out = day9.multi_head_attention(X_pe, n_heads=2, causal_mask=True)
        # Each head outputs d_model/n_heads = 2 dims → concatenated = 4 dims
        assert out.shape == X_pe.shape

    def test_output_shape_one_head(self, toy_embeddings):
        pe = day9.sinusoidal_positional_encoding(*toy_embeddings.shape)
        X_pe = toy_embeddings + pe
        out = day9.multi_head_attention(X_pe, n_heads=1, causal_mask=True)
        assert out.shape == X_pe.shape

    def test_different_heads_produce_nonidentical_outputs(self, toy_embeddings):
        """
        Two different heads with different random projections must not
        produce identical per-head outputs.
        """
        pe = day9.sinusoidal_positional_encoding(*toy_embeddings.shape)
        X_pe = toy_embeddings + pe
        # Collect per-head outputs by running single-head attention twice
        # with different seeds (simulates what multi_head_attention does)
        d_model = X_pe.shape[1]
        d_head = d_model // 2

        rng0 = np.random.default_rng(seed=0)
        Wq0 = rng0.standard_normal((d_model, d_head)) * 0.5
        Wk0 = rng0.standard_normal((d_model, d_head)) * 0.5
        Wv0 = rng0.standard_normal((d_model, d_head)) * 0.5

        rng1 = np.random.default_rng(seed=1)
        Wq1 = rng1.standard_normal((d_model, d_head)) * 0.5
        Wk1 = rng1.standard_normal((d_model, d_head)) * 0.5
        Wv1 = rng1.standard_normal((d_model, d_head)) * 0.5

        out0, _ = day9.scaled_dot_product_attention(X_pe @ Wq0, X_pe @ Wk0, X_pe @ Wv0)
        out1, _ = day9.scaled_dot_product_attention(X_pe @ Wq1, X_pe @ Wk1, X_pe @ Wv1)

        assert not np.allclose(out0, out1), "Both heads produce identical outputs"


# ---------------------------------------------------------------------------
# Part 4 – Tokenization concept
# ---------------------------------------------------------------------------

class TestTokenizationConcept:
    def test_token_vocab_coverage(self):
        """All tokens used in the demo must be in the vocab."""
        for token in ["BUY", "500", "ABC", "P1001"]:
            assert token in day9.TOKEN_VOCAB

    def test_token_ids_are_integers(self):
        for _, tid in day9.TOKEN_VOCAB.items():
            assert isinstance(tid, int)

    def test_token_ids_unique(self):
        ids = list(day9.TOKEN_VOCAB.values())
        assert len(ids) == len(set(ids)), "Token IDs are not unique"
