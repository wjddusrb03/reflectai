"""Built-in puzzle tasks for ReflectAI.

Available tasks:
- sudoku: 9x9 Sudoku with 27 constraints
- mnist_add: MNIST digit addition
- equation: Handwritten equation recognition
"""

from .sudoku import generate_sudoku, simulate_noisy_predictions
from .mnist_add import generate_addition_samples, simulate_addition_predictions
from .equation import generate_equations, simulate_equation_predictions

__all__ = [
    "generate_sudoku",
    "simulate_noisy_predictions",
    "generate_addition_samples",
    "simulate_addition_predictions",
    "generate_equations",
    "simulate_equation_predictions",
]
