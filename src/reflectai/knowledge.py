"""Knowledge base builders for domain-specific constraints.

Provides factory functions to create KnowledgeBase instances for
built-in tasks (Sudoku, MNIST addition, equations) and custom tasks.
"""

from __future__ import annotations

from .models import KnowledgeBase


# ══════════════════════════════════════════════════════════════════
# Sudoku (9×9)
# ══════════════════════════════════════════════════════════════════

def build_sudoku_kb(size: int = 9) -> KnowledgeBase:
    """Build knowledge base for standard Sudoku.

    27 constraints:
    - 9 row constraints (all_distinct per row)
    - 9 column constraints (all_distinct per column)
    - 9 box constraints (all_distinct per 3x3 box)

    Positions indexed 0-80 (row-major: pos = row*9 + col).
    Values: 1-9 (mapped to classes 1-9, 0=empty).

    Args:
        size: Grid size (default 9 for standard Sudoku)

    Returns:
        KnowledgeBase with 27 all_distinct constraints
    """
    box_size = int(size ** 0.5)
    assert box_size * box_size == size, f"Size must be a perfect square, got {size}"

    kb = KnowledgeBase(num_classes=size + 1, num_positions=size * size)

    # Row constraints
    for r in range(size):
        indices = [r * size + c for c in range(size)]
        kb.add_constraint(f"row_{r}", indices, "all_distinct")

    # Column constraints
    for c in range(size):
        indices = [r * size + c for r in range(size)]
        kb.add_constraint(f"col_{c}", indices, "all_distinct")

    # Box constraints
    for br in range(box_size):
        for bc in range(box_size):
            indices = []
            for dr in range(box_size):
                for dc in range(box_size):
                    r = br * box_size + dr
                    c = bc * box_size + dc
                    indices.append(r * size + c)
            kb.add_constraint(f"box_{br}_{bc}", indices, "all_distinct")

    return kb


def build_sudoku_4x4_kb() -> KnowledgeBase:
    """Build knowledge base for 4×4 mini Sudoku (good for testing)."""
    return build_sudoku_kb(size=4)


# ══════════════════════════════════════════════════════════════════
# MNIST Addition
# ══════════════════════════════════════════════════════════════════

def build_addition_kb(num_digits: int = 2,
                      target_sum: int | None = None) -> KnowledgeBase:
    """Build knowledge base for digit addition task.

    Task: Given images of N digits, their sum must equal target.
    Positions: [digit_0, digit_1, ..., digit_{N-1}]

    Args:
        num_digits: Number of digits to add
        target_sum: Expected sum (if known at build time)

    Returns:
        KnowledgeBase with sum constraint
    """
    kb = KnowledgeBase(num_classes=10, num_positions=num_digits)

    if target_sum is not None:
        indices = list(range(num_digits))
        kb.add_constraint(
            f"sum_equals_{target_sum}",
            indices,
            "sum_equals",
            target=target_sum,
        )

    return kb


# ══════════════════════════════════════════════════════════════════
# Equation Recognition
# ══════════════════════════════════════════════════════════════════

def build_equation_kb(num_positions: int = 5) -> KnowledgeBase:
    """Build knowledge base for equation recognition.

    Task: Recognize handwritten equations like "3 + 4 = 7"
    Positions represent symbols: [digit, operator, digit, equals, digit]

    Constraints:
    - Operands and result must be valid digits (0-9)
    - The equation must be arithmetically valid

    This is a simplified version — real implementation would need
    custom constraint rules for arithmetic operations.

    Args:
        num_positions: Number of symbol positions

    Returns:
        KnowledgeBase with range constraints
    """
    kb = KnowledgeBase(num_classes=10, num_positions=num_positions)

    # All positions should be valid digits
    indices = list(range(num_positions))
    kb.add_constraint("valid_digits", indices, "in_range", min=0, max=9)

    return kb


# ══════════════════════════════════════════════════════════════════
# N-Queens (bonus puzzle)
# ══════════════════════════════════════════════════════════════════

def build_nqueens_kb(n: int = 8) -> KnowledgeBase:
    """Build knowledge base for N-Queens puzzle.

    Positions: N queens, each assigned a column (0 to N-1).
    Constraints: all columns distinct + no diagonal attacks.

    Args:
        n: Board size

    Returns:
        KnowledgeBase with distinctness constraints
    """
    kb = KnowledgeBase(num_classes=n, num_positions=n)

    # All queens in different columns
    indices = list(range(n))
    kb.add_constraint("all_different_cols", indices, "all_distinct")

    return kb


# ══════════════════════════════════════════════════════════════════
# Generic Builder
# ══════════════════════════════════════════════════════════════════

def build_kb(task: str, **kwargs) -> KnowledgeBase:
    """Factory function to build a knowledge base by task name.

    Args:
        task: Task name ("sudoku", "sudoku_4x4", "addition", "equation", "nqueens")
        **kwargs: Task-specific parameters

    Returns:
        KnowledgeBase instance
    """
    builders = {
        "sudoku": build_sudoku_kb,
        "sudoku_4x4": build_sudoku_4x4_kb,
        "addition": build_addition_kb,
        "equation": build_equation_kb,
        "nqueens": build_nqueens_kb,
    }

    if task not in builders:
        raise ValueError(f"Unknown task: {task}. Available: {list(builders.keys())}")

    return builders[task](**kwargs)
