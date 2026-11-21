"""
+==================================================================+
|  PERSONAL AI MENTOR  -  Phase 1, Lesson 4                       |
|  Neural Networks from scratch - the foundation of all LLMs      |
|  Audience : 16+ yrs .NET / AWS enterprise developer             |
|  Duration : ~75 minutes                                          |
+==================================================================+

WHAT YOU WILL LEARN
-------------------
1. What a neuron is (and why the name is misleading)
2. Layers, activations and forward propagation
3. Backpropagation - the algorithm that trains every LLM
4. Build a 2-layer neural network from scratch using NumPy only
5. The same network with PyTorch (what production uses)
6. Why deep networks learn hierarchical features
7. The direct line from this lesson -> Transformers -> GPT-4

RUN THIS FILE
-------------
    pip install numpy torch
    python lesson04_neural_networks.py
"""

import numpy as np
SEP = "-" * 65

# ==================================================================
# CONCEPT: What IS a neural network?
# ==================================================================
"""
A neural network is a function composition:

    output = f_L( f_{L-1}( ... f_2( f_1(input) ) ... ) )

Each f_i is: activation( weights_i @ input + bias_i )

That's it. The 'neural' metaphor is a historical accident.
Think of it as a stack of linear transformations with non-linear
activations between them.

.NET analogy: a middleware pipeline where each layer transforms
the data, with a non-linear function (like clamping or gating)
applied after each transformation.

WHY DO WE NEED NON-LINEAR ACTIVATIONS?
  Without them, stacking N linear layers is mathematically equivalent
  to ONE linear layer. No matter how deep, you'd just have logistic
  regression. Non-linearities let the network learn curves, boundaries
  that aren't straight lines, and complex feature interactions.

COMMON ACTIVATIONS:
  ReLU     -> max(0, x)                 -- most common hidden layer activation
  Sigmoid  -> 1/(1+e^-x)               -- binary output (0-1)
  Softmax  -> normalised exponentials   -- multi-class output (probabilities)
  Tanh     -> (e^x - e^-x)/(e^x+e^-x)  -- zero-centred, used in RNNs
  GELU     -> smooth ReLU variant       -- used in Transformers / GPT
"""

# ==================================================================
# SECTION 1 - Build a Neural Network layer by layer from scratch
# ==================================================================
"""
PROBLEM: Classify investment transactions as high-risk or normal.

ARCHITECTURE:
  Input layer:   6 features (transaction data)
  Hidden layer:  16 neurons, ReLU activation
  Hidden layer:  8  neurons, ReLU activation
  Output layer:  1 neuron,   Sigmoid activation -> probability

This is a 2-hidden-layer feedforward neural network (MLP).
"""

print(SEP)
print("LESSON 4 - Neural Networks from Scratch")
print(SEP)

# -- Activation functions ------------------------------------------

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def relu_derivative(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    s = sigmoid(x)
    return s * (1 - s)

def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-9
    return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))


# -- Data ----------------------------------------------------------

def make_data(n: int = 1000, seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    n_risk = int(n * 0.1)
    n_norm = n - n_risk

    X_norm = np.column_stack([
        rng.lognormal(10, 1, n_norm),
        rng.beta(2, 8, n_norm),
        rng.binomial(1, 0.05, n_norm),
        rng.binomial(1, 0.08, n_norm),
        rng.beta(2, 10, n_norm),
        rng.poisson(2, n_norm).astype(float),
    ])
    X_risk = np.column_stack([
        rng.lognormal(13, 1.5, n_risk),
        rng.beta(8, 2, n_risk),
        rng.binomial(1, 0.4, n_risk),
        rng.binomial(1, 0.5, n_risk),
        rng.beta(8, 2, n_risk),
        rng.poisson(12, n_risk).astype(float),
    ])

    X = np.vstack([X_norm, X_risk])
    y = np.concatenate([np.zeros(n_norm), np.ones(n_risk)])

    # Normalise features to mean=0, std=1 (critical for neural networks)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)

    idx = rng.permutation(n)
    X, y = X[idx], y[idx]

    split = int(n * 0.8)
    return X[:split], X[split:], y[:split], y[split:]


X_train, X_test, y_train, y_test = make_data(1000)
n_features = X_train.shape[1]
print(f"Data: {len(X_train)} train, {len(X_test)} test | {n_features} features")
print()


# -- Neural Network class (from scratch) --------------------------

class TwoLayerNeuralNetwork:
    """
    A 2-hidden-layer feedforward network trained with backpropagation.

    Architecture:
        Input (6) -> Dense(16, ReLU) -> Dense(8, ReLU) -> Dense(1, Sigmoid)

    .NET analogy: think of each layer as a middleware handler with
    learnable parameters, and backpropagation as the reverse pass that
    computes gradients to update those parameters.
    """

    def __init__(
        self,
        input_dim: int,
        hidden1: int = 16,
        hidden2: int = 8,
        learning_rate: float = 0.01,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)

        # He initialisation -- scales weights by sqrt(2/fan_in)
        # Prevents vanishing/exploding gradients in ReLU networks
        self.W1 = rng.normal(0, np.sqrt(2.0 / input_dim), (input_dim, hidden1))
        self.b1 = np.zeros(hidden1)
        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden1), (hidden1, hidden2))
        self.b2 = np.zeros(hidden2)
        self.W3 = rng.normal(0, np.sqrt(2.0 / hidden2), (hidden2, 1))
        self.b3 = np.zeros(1)

        self.lr = learning_rate
        self._cache = {}   # stores intermediate values needed for backprop

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass: compute predictions.
        Cache intermediate values -- backprop needs them.
        """
        # Layer 1: linear + ReLU
        z1 = X @ self.W1 + self.b1
        a1 = relu(z1)

        # Layer 2: linear + ReLU
        z2 = a1 @ self.W2 + self.b2
        a2 = relu(z2)

        # Output layer: linear + Sigmoid
        z3 = a2 @ self.W3 + self.b3
        a3 = sigmoid(z3)

        self._cache = dict(X=X, z1=z1, a1=a1, z2=z2, a2=a2, z3=z3, a3=a3)
        return a3.squeeze()

    def backward(self, y_true: np.ndarray) -> None:
        """
        Backpropagation: compute gradients and update weights.

        The chain rule of calculus applied layer by layer in reverse.
        Each layer's gradient depends on the layer above it.

        CONCEPT (don't memorise, understand the direction):
          dL/dW3 = how much the loss changes when W3 changes
          dL/dW2 = dL/da2 * da2/dz2 * dz2/dW2  (chain rule)
          dL/dW1 = dL/da1 * da1/dz1 * dz1/dW1  (chain rule, one more step)

        This is what PyTorch's autograd does automatically.
        """
        X, z1, a1, z2, a2, z3, a3 = (
            self._cache[k] for k in ("X", "z1", "a1", "z2", "a2", "z3", "a3")
        )
        n = len(y_true)
        y_true_col = y_true.reshape(-1, 1)

        # Output layer gradient
        dz3 = a3 - y_true_col                              # (n, 1)
        dW3 = a2.T @ dz3 / n
        db3 = dz3.mean(axis=0)

        # Layer 2 gradient
        da2 = dz3 @ self.W3.T                              # (n, 8)
        dz2 = da2 * relu_derivative(z2)
        dW2 = a1.T @ dz2 / n
        db2 = dz2.mean(axis=0)

        # Layer 1 gradient
        da1 = dz2 @ self.W2.T                              # (n, 16)
        dz1 = da1 * relu_derivative(z1)
        dW1 = X.T @ dz1 / n
        db1 = dz1.mean(axis=0)

        # Gradient descent weight updates
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 500) -> None:
        print("Training neural network from scratch:")
        print(f"  {'Epoch':>6}  {'Loss':>10}  {'Accuracy':>10}")
        print(f"  {'-'*6}  {'-'*10}  {'-'*10}")

        for epoch in range(epochs):
            predictions = self.forward(X)
            loss = binary_cross_entropy(y, predictions)
            self.backward(y)

            if epoch % 100 == 0 or epoch == epochs - 1:
                acc = np.mean((predictions >= 0.5) == y)
                print(f"  {epoch:>6}  {loss:>10.4f}  {acc:>10.2%}")

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> tuple:
        proba = self.forward(X)
        return (proba >= threshold).astype(int), proba


# -- Train and evaluate --------------------------------------------

nn = TwoLayerNeuralNetwork(input_dim=n_features, hidden1=16, hidden2=8, learning_rate=0.05)
nn.train(X_train, y_train, epochs=500)

preds, proba = nn.predict(X_test)
accuracy = np.mean(preds == y_test)
tp = int(((preds == 1) & (y_test == 1)).sum())
fp = int(((preds == 1) & (y_test == 0)).sum())
fn = int(((preds == 0) & (y_test == 1)).sum())
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0

print()
print(f"From-scratch NN results:")
print(f"  Accuracy  : {accuracy:.2%}")
print(f"  Precision : {precision:.2%}")
print(f"  Recall    : {recall:.2%}")


# ==================================================================
# SECTION 2 - The same network in PyTorch
# ==================================================================

print()
print(SEP)
print("SECTION 2 - Same architecture in PyTorch")
print(SEP)

"""
PyTorch is to neural networks what .NET is to web APIs.
It handles:
  - Automatic differentiation (autograd) -- no manual backprop
  - GPU acceleration -- training on CUDA
  - Model serialisation -- save/load trained models
  - Production deployment -- TorchScript, ONNX export

The code looks almost identical to your from-scratch version
but with zero manual gradient computation.
"""

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader

    # Convert NumPy arrays to PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
    y_test_t  = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1)

    # Mini-batch DataLoader (like paginating a database query)
    dataset = TensorDataset(X_train_t, y_train_t)
    loader  = DataLoader(dataset, batch_size=32, shuffle=True)

    # Define the same 2-layer architecture using nn.Sequential
    # .NET analogy: builder pattern for a middleware pipeline
    model = nn.Sequential(
        nn.Linear(n_features, 16),  # Dense(16)
        nn.ReLU(),                   # ReLU activation
        nn.Linear(16, 8),            # Dense(8)
        nn.ReLU(),
        nn.Linear(8, 1),             # Dense(1)
        nn.Sigmoid(),                # sigmoid output
    )

    # Loss function and optimiser
    criterion = nn.BCELoss()        # binary cross-entropy
    optimizer = optim.Adam(         # Adam: adaptive gradient descent
        model.parameters(),
        lr=0.001,
    )

    print("Training PyTorch model:")
    print(f"  {'Epoch':>6}  {'Loss':>10}  {'Accuracy':>10}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}")

    for epoch in range(50):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()         # clear gradients
            output = model(X_batch)       # forward pass
            loss = criterion(output, y_batch)
            loss.backward()               # autograd computes ALL gradients
            optimizer.step()              # update weights
            epoch_loss += loss.item()

        if epoch % 10 == 0 or epoch == 49:
            model.eval()
            with torch.no_grad():
                test_out = model(X_test_t)
                test_loss = criterion(test_out, y_test_t).item()
                test_acc = ((test_out >= 0.5).float() == y_test_t).float().mean().item()
            print(f"  {epoch:>6}  {test_loss:>10.4f}  {test_acc:>10.2%}")

    model.eval()
    with torch.no_grad():
        pt_proba = model(X_test_t).squeeze().numpy()
        pt_preds = (pt_proba >= 0.5).astype(int)

    pt_tp = int(((pt_preds == 1) & (y_test == 1)).sum())
    pt_fp = int(((pt_preds == 1) & (y_test == 0)).sum())
    pt_fn = int(((pt_preds == 0) & (y_test == 1)).sum())
    pt_p = pt_tp / (pt_tp + pt_fp) if (pt_tp + pt_fp) > 0 else 0
    pt_r = pt_tp / (pt_tp + pt_fn) if (pt_tp + pt_fn) > 0 else 0
    print()
    print(f"PyTorch NN results:")
    print(f"  Accuracy  : {((pt_preds == y_test).mean()):.2%}")
    print(f"  Precision : {pt_p:.2%}")
    print(f"  Recall    : {pt_r:.2%}")

except ImportError:
    print("PyTorch not installed. Run: pip install torch")
    print("Skipping PyTorch section.")


# ==================================================================
# SECTION 3 - The direct path from this -> Transformers -> GPT-4
# ==================================================================

print()
print(SEP)
print("SECTION 3 - The road from MLP to GPT-4")
print(SEP)

EVOLUTION = [
    ("1980s", "Perceptron / MLP",
     "What you built in this lesson. Single hidden layers.\n"
     "    Limitation: can't process sequences or long-range dependencies."),
    ("1990s", "Recurrent Neural Network (RNN)",
     "Added a loop -- the network's output feeds back as input.\n"
     "    Limitation: vanishing gradients over long sequences."),
    ("1997", "LSTM (Long Short-Term Memory)",
     "Gates control what to remember and forget in sequences.\n"
     "    Used in early NLP, speech, time series."),
    ("2017", "Transformer (Attention Is All You Need)",
     "Replaced recurrence with self-attention.\n"
     "    Every token can attend to every other token simultaneously.\n"
     "    Parallelisable -- trains 10-100x faster than RNNs."),
    ("2018", "BERT",
     "Transformer encoder pre-trained on massive text.\n"
     "    Bidirectional -- sees both left and right context."),
    ("2018-2024", "GPT-1 -> GPT-4o",
     "Transformer decoder only. Generates text token by token.\n"
     "    Scaled to billions of parameters and billions of training tokens."),
    ("2025", "Claude Opus 5 / GPT-5 / Gemini 2",
     "Same Transformer architecture -- just dramatically larger,\n"
     "    better data, RLHF alignment, multimodal inputs."),
]

for year, name, description in EVOLUTION:
    print(f"\n  {year:>8}  {name}")
    for line in description.split("\n"):
        print(f"            {line.strip()}")

print()
print("  KEY INSIGHT:")
print("  Every LLM you will use -- GPT-4, Claude, Llama -- is at its")
print("  core the same forward-pass + backpropagation you implemented")
print("  in this lesson, applied at 100,000x the scale with the")
print("  Transformer architecture replacing the simple MLP layers.")


# ==================================================================
# SECTION 4 - Five Interview Questions
# ==================================================================

print()
print(SEP)
print("SECTION 4 - Interview Questions")
print(SEP)

QA = [
    ("Q1. What is backpropagation?",
     """Backpropagation is the algorithm that computes the gradient of the loss
   function with respect to every weight in the network.
   It applies the chain rule of calculus from the output layer backward
   to the input layer. The gradients are then used by gradient descent
   to update the weights.
   In PyTorch, loss.backward() does this automatically via autograd.
   Understanding it matters because gradient vanishing/exploding issues
   are the reason specific initialisations and activations were invented."""),
    ("Q2. Why do we need activation functions?",
     """Without non-linear activations, stacking N linear layers is
   mathematically equivalent to a single linear layer -- the network
   can only learn linear decision boundaries.
   Activations like ReLU introduce non-linearity, allowing the network
   to approximate any continuous function (Universal Approximation Theorem).
   ReLU is preferred over sigmoid in hidden layers because it doesn't
   suffer from vanishing gradients for positive inputs."""),
    ("Q3. What is the vanishing gradient problem?",
     """When training deep networks, gradients are multiplied through
   many layers (chain rule). If each multiplication factor is < 1
   (like sigmoid derivatives, which max at 0.25), the gradient
   shrinks exponentially and early layers stop learning.
   Solutions: ReLU activation, residual connections (ResNet),
   layer normalisation, careful weight initialisation (He/Xavier)."""),
    ("Q4. What is the difference between batch, mini-batch and stochastic gradient descent?",
     """Batch GD: compute gradient on the FULL dataset. Accurate but slow.
   Stochastic GD: compute gradient on ONE sample. Fast but noisy.
   Mini-batch GD: compute gradient on a subset (e.g. 32 samples).
   Best trade-off -- vectorised computation on GPUs, noisy enough
   to escape local minima.
   Production default: mini-batch with batch size 32-256.
   Smaller batches -> noisier gradients -> sometimes better generalisation."""),
    ("Q5. How does a neural network relate to an LLM?",
     """An LLM is a very large neural network with the Transformer architecture.
   The core mechanics are identical: forward pass, loss computation,
   backpropagation, gradient descent.
   What differs: architecture (attention instead of fully connected layers),
   scale (billions of parameters), training data (internet-scale text),
   and training objective (predict next token instead of a fixed label).
   The from-scratch network in this lesson does the same forward/backward
   computation as GPT-4 -- just with 6 input features instead of 100k tokens."""),
]

for question, answer in QA:
    print(f"\n  {question}")
    print(f"  {'-' * (len(question) - 2)}")
    for line in answer.strip().split("\n"):
        print(f"  {line}")

print()
print(SEP)
print("EXERCISES")
print(SEP)
print("""
  EXERCISE 1: Add a third hidden layer.
    Change the architecture to: Input -> 32 -> 16 -> 8 -> 1
    Does training accuracy improve? What about test accuracy?
    If test accuracy drops, you're seeing overfitting.

  EXERCISE 2: Try different learning rates: 0.001, 0.01, 0.1, 1.0.
    Plot the loss at each epoch (print it and observe).
    What happens with lr=1.0? What is 'exploding gradients'?

  EXERCISE 3: Replace ReLU with Sigmoid in hidden layers.
    Does the network train as well? Can you observe vanishing gradients?
    (The loss stops decreasing after a few epochs.)

  EXERCISE 4 (PyTorch): Save and load the model.
    torch.save(model.state_dict(), 'transaction_nn.pth')
    model.load_state_dict(torch.load('transaction_nn.pth'))
    model.eval()
    This is the equivalent of serialising a trained model for deployment.
""")

print(SEP)
print("NEXT LESSON: lesson05_model_evaluation.py")
print("  -> Full evaluation pipeline for production AI systems")
print("  -> Confusion matrix, AUC, PR curve")
print("  -> Classification thresholds and business cost of errors")
print("  -> Model drift monitoring")
print(SEP)
