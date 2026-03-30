# ReflectAI

**Neuro-symbolic reasoning with abductive reflection** — first open-source implementation of ABL-Refl (AAAI 2026 Outstanding Paper).

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-153%20passed-brightgreen.svg)]()

> **Paper**: *"Efficient Rectification of Neuro-Symbolic Reasoning Inconsistencies by Abductive Reflection"* (AAAI 2026 Outstanding Paper)

## What is ReflectAI?

ReflectAI combines neural networks with logical reasoning to solve constraint-satisfaction problems. The key innovation is the **reflection head** — a binary error detector that learns which neural predictions are likely wrong, enabling targeted correction instead of expensive brute-force search.

### Architecture

```
Input (e.g., Sudoku digit images)
  |
  v
[Body Block] ── shared feature extractor (CNN/MLP)
  |         \
  v          v
[Output Head]  [Reflection Head]    <-- core innovation
  |               |
  v               v
Predictions    Error Flags (binary: 1=suspect, 0=trusted)
  |               |
  +-------+-------+
          |
          v
[Abductive Solver] ── corrects ONLY flagged positions
          |
          v
   Corrected Labels (constraint-consistent)
```

### Why ReflectAI?

| Feature | Traditional ABL | ReflectAI (ABL-Refl) |
|---------|----------------|---------------------|
| Error detection | Search all positions | Reflection head targets suspects |
| Speed | Baseline | **10,000-15,000x faster** |
| Data efficiency | 20,000 labels | **2,000 labels** (10x less) |
| Sudoku accuracy | 76.5% | **97.4%** |
| Error detection recall | - | **99.04%** |

## Installation

```bash
# Clone the repository
git clone https://github.com/wjddusrb03/reflectai.git
cd reflectai

# Install (editable mode)
pip install -e .

# With Z3 SMT solver (recommended for complex constraints)
pip install -e ".[z3]"

# With Web UI (Gradio)
pip install -e ".[web]"

# With all optional backends + Web UI
pip install -e ".[all]"

# Development (with tests)
pip install -e ".[dev]"
```

## Quick Start

### Web UI (Recommended!)

```bash
pip install -e ".[web]"    # Install Gradio
reflectai web              # Open http://localhost:7860
```

![Web UI](https://img.shields.io/badge/Web_UI-Gradio-orange.svg)

The web UI provides:
- Interactive Sudoku solver with color-coded grids
- Digit addition task visualization
- Benchmark runner
- Step-by-step pipeline explanation

### CLI Demo

```bash
# Run interactive demo
reflectai demo --difficulty medium --verbose

# Benchmark on multiple puzzles
reflectai benchmark --num-puzzles 20 --difficulty hard
```

### Python API

```python
import numpy as np
from reflectai.knowledge import build_sudoku_kb
from reflectai.tasks.sudoku import generate_sudoku, simulate_noisy_predictions
from reflectai.pipeline import solve_from_predictions

# Generate a Sudoku puzzle
puzzle, solution = generate_sudoku("medium", seed=42)

# Simulate noisy neural predictions (15% error rate)
prediction = simulate_noisy_predictions(solution, error_rate=0.15, seed=42)

# Build knowledge base (27 constraints: rows + cols + boxes)
kb = build_sudoku_kb()

# Simulate reflection scores (in practice, learned by the reflection head)
errors = prediction.labels != solution
reflection_scores = np.where(errors, 0.8, 0.1)  # High for errors

# Run abductive reasoning on flagged positions
result = solve_from_predictions(
    prediction.labels,
    prediction.probabilities,
    reflection_scores,
    kb,
    threshold=0.5,
    solver_type="backtrack",
)

print(f"Prediction accuracy: {(prediction.labels == solution).mean():.1%}")
print(f"Correction accuracy: {(result.final_labels == solution).mean():.1%}")
print(f"Consistent: {result.is_consistent}")
```

### Training a ReflectAI Model

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from reflectai.perception import MLPBody, PerceptionModule
from reflectai.reflection import ReflectionHead
from reflectai.trainer import ReflectAIModel, Trainer
from reflectai.models import TrainConfig, KnowledgeBase

# Build model
body = MLPBody(input_dim=784, hidden_dim=128)
perception = PerceptionModule(body, num_classes=10)
reflection_head = ReflectionHead(hidden_dim=128, num_classes=10)
model = ReflectAIModel(perception, reflection_head)

# Configure training with three-loss objective
config = TrainConfig(
    epochs=50,
    learning_rate=1e-3,
    lambda_consistency=1.0,       # Weight for consistency loss
    lambda_reflection_size=0.1,   # Weight for reflection size regularizer
    reflection_target_rate=0.2,   # Target ~20% flagging rate (C=0.8)
)

# Train
kb = KnowledgeBase(num_classes=10)
trainer = Trainer(model, kb, config)
history = trainer.train(train_loader, callback=lambda s: print(s.to_dict()))
```

## Components

### Core Modules

| Module | Description |
|--------|-------------|
| `perception.py` | Neural body (CNN/MLP) + output head |
| `reflection.py` | Reflection head — binary error detector (core innovation) |
| `reasoner.py` | Abductive solvers (backtrack, Z3) |
| `knowledge.py` | Knowledge base builders |
| `trainer.py` | Three-loss training loop |
| `pipeline.py` | End-to-end inference pipeline |
| `cli.py` | Command-line interface |

### Built-in Tasks

| Task | Constraints | Description |
|------|------------|-------------|
| Sudoku (9x9) | 27 all_distinct | Row, column, and box uniqueness |
| Sudoku (4x4) | 12 all_distinct | Mini Sudoku for testing |
| MNIST Addition | 1 sum_equals | Two digits must sum to target |
| Equation | 1 in_range | Handwritten equation recognition |
| N-Queens | 1 all_distinct | Queen placement on N x N board |

### Three-Loss Training

```
L_total = L_supervised + lambda_c * L_consistency + lambda_r * L_reflection_size
```

1. **L_supervised**: Standard cross-entropy on labeled data
2. **L_consistency**: REINFORCE reward when correction improves constraint satisfaction
3. **L_reflection_size**: Regularizer to maintain ~20% flagging rate (prevents trivial solutions)

## CLI Commands

```bash
# Launch web UI (recommended)
reflectai web
reflectai web --port 8080 --share    # Custom port + public link

# Show system info and available backends
reflectai info

# Run Sudoku demo
reflectai demo --difficulty hard --error-rate 0.2 --solver backtrack -v

# Benchmark pipeline performance
reflectai benchmark --num-puzzles 50 --difficulty medium
```

## Project Structure

```
reflectai/
├── src/reflectai/
│   ├── __init__.py           # Package exports
│   ├── models.py             # Core data structures
│   ├── perception.py         # Neural networks (CNN/MLP)
│   ├── reflection.py         # Reflection head (core innovation)
│   ├── reasoner.py           # Abductive solvers
│   ├── knowledge.py          # Knowledge base builders
│   ├── trainer.py            # Three-loss training loop
│   ├── pipeline.py           # End-to-end pipeline
│   ├── cli.py                # CLI interface
│   ├── web.py                # Gradio web UI
│   └── tasks/
│       ├── sudoku.py         # Sudoku puzzle utilities
│       ├── mnist_add.py      # MNIST digit addition
│       └── equation.py       # Equation recognition
├── tests/                    # 153 tests
├── pyproject.toml
├── LICENSE
└── README.md
```

## Key Concepts

### Reflection Vector

The reflection vector `r` is a binary mask over all prediction positions:
- `r[i] = 1`: Position `i` is flagged as a potential error
- `r[i] = 0`: Position `i` is trusted

Only flagged positions are sent to the constraint solver, dramatically reducing search space.

### Abductive Reasoning

Given predictions `p` and reflection flags `r`, find corrected labels `c` such that:
1. `c[i] = p[i]` for all trusted positions (where `r[i] = 0`)
2. All constraints in the knowledge base are satisfied
3. For flagged positions, prefer values with higher neural network probability

### Why "Abductive"?

Abduction = reasoning from observations to the best explanation. The solver "explains" constraint violations by finding the minimal corrections to flagged predictions.

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- NumPy >= 1.24

Optional:
- z3-solver >= 4.12 (for Z3 backend)
- python-sat >= 1.8 (for SAT backend)
- torchvision >= 0.15 (for image tasks)

## Citation

```bibtex
@inproceedings{abl-refl-2026,
  title={Efficient Rectification of Neuro-Symbolic Reasoning Inconsistencies by Abductive Reflection},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2026},
  note={Outstanding Paper Award}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
