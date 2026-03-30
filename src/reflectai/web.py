"""Web UI for ReflectAI — interactive Sudoku solver with reflection visualization.

Provides a Gradio-based web interface for:
1. Sudoku puzzle generation and solving
2. MNIST Addition task
3. Reflection visualization (which cells are flagged)
4. Step-by-step pipeline demonstration

Usage:
    reflectai web                    # Launch on localhost:7860
    reflectai web --port 8080        # Custom port
    reflectai web --share            # Create public link
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .knowledge import build_sudoku_kb, build_addition_kb
from .models import Reflection
from .reasoner import create_solver
from .tasks.sudoku import (
    generate_sudoku,
    simulate_noisy_predictions,
    evaluate_sudoku_correction,
)
from .tasks.mnist_add import simulate_addition_predictions, evaluate_addition


# ══════════════════════════════════════════════════════════════════
# Color helpers
# ══════════════════════════════════════════════════════════════════

_CSS = """
.reflectai-header { text-align: center; margin-bottom: 10px; }
.sudoku-grid table { border-collapse: collapse; margin: 0 auto; font-family: monospace; }
.sudoku-grid td {
    width: 42px; height: 42px; text-align: center; vertical-align: middle;
    font-size: 18px; font-weight: bold; border: 1px solid #999;
}
.sudoku-grid tr:nth-child(3n) td { border-bottom: 2.5px solid #333; }
.sudoku-grid td:nth-child(3n) { border-right: 2.5px solid #333; }
.sudoku-grid tr:first-child td { border-top: 2.5px solid #333; }
.sudoku-grid td:first-child { border-left: 2.5px solid #333; }
.cell-correct { background: #d4edda; color: #155724; }
.cell-error   { background: #f8d7da; color: #721c24; }
.cell-fixed   { background: #cce5ff; color: #004085; }
.cell-empty   { background: #f8f9fa; color: #999; }
.cell-flagged { background: #fff3cd; color: #856404; }
"""


def _grid_html(values: np.ndarray, highlights: dict | None = None,
               title: str = "") -> str:
    """Render 9x9 Sudoku grid as HTML table.

    Args:
        values: [81] array
        highlights: dict mapping index -> css class name
        title: Grid title
    """
    grid = values.reshape(9, 9)
    html = f'<div class="sudoku-grid"><h4 style="text-align:center">{title}</h4><table>'
    for r in range(9):
        html += "<tr>"
        for c in range(9):
            idx = r * 9 + c
            val = int(grid[r, c])
            display = str(val) if val > 0 else "&middot;"
            cls = ""
            if highlights and idx in highlights:
                cls = f' class="{highlights[idx]}"'
            html += f"<td{cls}>{display}</td>"
        html += "</tr>"
    html += "</table></div>"
    return html


def _4x4_grid_html(values: list, highlights: dict | None = None,
                   title: str = "") -> str:
    """Render a small 4x4 grid."""
    html = f'<div><h4 style="text-align:center">{title}</h4><table style="border-collapse:collapse;margin:0 auto;font-family:monospace;">'
    for r in range(4):
        html += "<tr>"
        for c in range(4):
            idx = r * 4 + c
            if idx < len(values):
                val = int(values[idx])
                display = str(val) if val > 0 else "&middot;"
            else:
                display = "&middot;"
            style = "width:42px;height:42px;text-align:center;font-size:18px;font-weight:bold;border:1px solid #999;"
            if r % 2 == 1:
                style += "border-bottom:2.5px solid #333;"
            if c % 2 == 1:
                style += "border-right:2.5px solid #333;"
            if r == 0:
                style += "border-top:2.5px solid #333;"
            if c == 0:
                style += "border-left:2.5px solid #333;"
            bg = ""
            if highlights and idx in highlights:
                cls = highlights[idx]
                if cls == "cell-correct":
                    bg = "background:#d4edda;color:#155724;"
                elif cls == "cell-error":
                    bg = "background:#f8d7da;color:#721c24;"
                elif cls == "cell-flagged":
                    bg = "background:#fff3cd;color:#856404;"
                elif cls == "cell-fixed":
                    bg = "background:#cce5ff;color:#004085;"
            html += f'<td style="{style}{bg}">{display}</td>'
        html += "</tr>"
    html += "</table></div>"
    return html


# ══════════════════════════════════════════════════════════════════
# Sudoku solver logic
# ══════════════════════════════════════════════════════════════════

def solve_sudoku(difficulty: str, error_rate: float, threshold: float,
                 solver_type: str, seed: int):
    """Full Sudoku pipeline — returns HTML grids + metrics text."""
    # Generate
    puzzle, solution = generate_sudoku(difficulty=difficulty, seed=int(seed))
    kb = build_sudoku_kb()

    # Simulate noisy predictions
    prediction = simulate_noisy_predictions(
        solution, error_rate=error_rate, seed=int(seed)
    )
    errors = prediction.labels != solution

    # Reflection simulation
    scores = 1.0 - prediction.confidence
    for i in range(81):
        if errors[i]:
            scores[i] = min(scores[i] + 0.3, 1.0)

    flags = (scores >= threshold).astype(int)
    reflection = Reflection(flags=flags, scores=scores, threshold=threshold)

    # Solve
    solver = create_solver(solver_type)
    t0 = time.perf_counter()
    correction = solver.solve(prediction, reflection, kb, timeout_ms=10000)
    solve_ms = (time.perf_counter() - t0) * 1000

    # Evaluate
    metrics = evaluate_sudoku_correction(prediction, solution, correction.labels)

    # ---- Build HTML grids ----
    # 1) Puzzle
    puzzle_hl = {}
    for i in range(81):
        puzzle_hl[i] = "cell-empty" if puzzle[i] == 0 else ""
    grid_puzzle = _grid_html(puzzle, puzzle_hl, "Puzzle (given)")

    # 2) Neural prediction — green=correct, red=error
    pred_hl = {}
    for i in range(81):
        pred_hl[i] = "cell-correct" if not errors[i] else "cell-error"
    grid_pred = _grid_html(prediction.labels, pred_hl,
                           f"Neural Prediction ({errors.sum()} errors)")

    # 3) Reflection — yellow=flagged
    ref_hl = {}
    for i in range(81):
        if flags[i]:
            ref_hl[i] = "cell-flagged"
        else:
            ref_hl[i] = "cell-correct" if not errors[i] else ""
    grid_ref = _grid_html(prediction.labels, ref_hl,
                          f"Reflection ({reflection.num_flagged} flagged)")

    # 4) Correction result
    corr_errors = correction.labels != solution
    corr_hl = {}
    for i in range(81):
        if correction.changed_mask[i] and not corr_errors[i]:
            corr_hl[i] = "cell-fixed"    # Fixed by solver
        elif corr_errors[i]:
            corr_hl[i] = "cell-error"     # Still wrong
        else:
            corr_hl[i] = "cell-correct"   # Correct
    grid_corr = _grid_html(correction.labels, corr_hl,
                           f"After Correction ({corr_errors.sum()} remaining errors)")

    # 5) Ground truth
    grid_truth = _grid_html(solution, {}, "Ground Truth")

    # Combined HTML
    grids_html = f"""<style>{_CSS}</style>
    <div style="display:flex; flex-wrap:wrap; gap:20px; justify-content:center;">
        {grid_puzzle}{grid_pred}{grid_ref}{grid_corr}{grid_truth}
    </div>
    <div style="text-align:center; margin-top:10px; font-size:13px; color:#666;">
        <span style="display:inline-block;width:16px;height:16px;background:#d4edda;border:1px solid #ccc;vertical-align:middle;"></span> Correct &nbsp;
        <span style="display:inline-block;width:16px;height:16px;background:#f8d7da;border:1px solid #ccc;vertical-align:middle;"></span> Error &nbsp;
        <span style="display:inline-block;width:16px;height:16px;background:#fff3cd;border:1px solid #ccc;vertical-align:middle;"></span> Flagged &nbsp;
        <span style="display:inline-block;width:16px;height:16px;background:#cce5ff;border:1px solid #ccc;vertical-align:middle;"></span> Fixed by solver
    </div>"""

    # Metrics text
    result_text = (
        f"## Results\n\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| Prediction Accuracy | {metrics['prediction_accuracy']:.1%} |\n"
        f"| Correction Accuracy | {metrics['correction_accuracy']:.1%} |\n"
        f"| Improvement | **+{metrics['improvement']:.1%}** |\n"
        f"| Errors Fixed | {metrics['errors_fixed']} |\n"
        f"| Errors Introduced | {metrics['errors_introduced']} |\n"
        f"| Total Original Errors | {metrics['total_original_errors']} |\n"
        f"| Constraint Consistent | {'Yes' if correction.consistent else 'No'} |\n"
        f"| Solver Time | {solve_ms:.1f} ms |\n"
        f"| Positions Flagged | {reflection.num_flagged} ({reflection.flag_rate:.0%}) |\n"
    )

    return grids_html, result_text


# ══════════════════════════════════════════════════════════════════
# Addition solver logic
# ══════════════════════════════════════════════════════════════════

def solve_addition(digit_a: int, digit_b: int, error_rate: float,
                   solver_type: str, seed: int):
    """Addition pipeline — returns HTML + metrics."""
    digits = np.array([int(digit_a), int(digit_b)])
    target_sum = int(digits.sum())

    prediction = simulate_addition_predictions(digits, error_rate=error_rate,
                                                seed=int(seed))

    # Reflection: flag digits with low confidence
    scores = 1.0 - prediction.confidence
    errors = prediction.labels != digits
    for i in range(len(scores)):
        if errors[i]:
            scores[i] = min(scores[i] + 0.4, 1.0)

    flags = (scores >= 0.5).astype(int)
    reflection = Reflection(flags=flags, scores=scores, threshold=0.5)

    kb = build_addition_kb(num_digits=2, target_sum=target_sum)
    solver = create_solver(solver_type)
    t0 = time.perf_counter()
    correction = solver.solve(prediction, reflection, kb, timeout_ms=5000)
    solve_ms = (time.perf_counter() - t0) * 1000

    metrics = evaluate_addition(prediction, digits, correction.labels, target_sum)

    # Visual
    def _digit_box(label: str, value: int, css_class: str = "") -> str:
        bg = {"correct": "#d4edda", "error": "#f8d7da", "fixed": "#cce5ff"}.get(css_class, "#f8f9fa")
        return (f'<div style="display:inline-block;width:80px;height:90px;'
                f'border:2px solid #333;border-radius:8px;text-align:center;'
                f'background:{bg};margin:5px;padding-top:8px;">'
                f'<div style="font-size:11px;color:#666;">{label}</div>'
                f'<div style="font-size:36px;font-weight:bold;">{value}</div></div>')

    # Prediction row
    pred_a_cls = "correct" if prediction.labels[0] == digits[0] else "error"
    pred_b_cls = "correct" if prediction.labels[1] == digits[1] else "error"
    pred_html = (
        f'<div style="text-align:center;margin:10px 0;">'
        f'<h4>Neural Prediction</h4>'
        f'{_digit_box("Digit A", prediction.labels[0], pred_a_cls)}'
        f'<span style="font-size:36px;vertical-align:middle;"> + </span>'
        f'{_digit_box("Digit B", prediction.labels[1], pred_b_cls)}'
        f'<span style="font-size:36px;vertical-align:middle;"> = </span>'
        f'{_digit_box("Sum", int(prediction.labels.sum()))}'
        f'</div>'
    )

    # Correction row
    corr_a_cls = "fixed" if correction.changed_mask[0] else ("correct" if correction.labels[0] == digits[0] else "error")
    corr_b_cls = "fixed" if correction.changed_mask[1] else ("correct" if correction.labels[1] == digits[1] else "error")
    corr_html = (
        f'<div style="text-align:center;margin:10px 0;">'
        f'<h4>After Correction</h4>'
        f'{_digit_box("Digit A", correction.labels[0], corr_a_cls)}'
        f'<span style="font-size:36px;vertical-align:middle;"> + </span>'
        f'{_digit_box("Digit B", correction.labels[1], corr_b_cls)}'
        f'<span style="font-size:36px;vertical-align:middle;"> = </span>'
        f'{_digit_box("Sum", int(correction.labels.sum()))}'
        f'</div>'
    )

    # Truth row
    truth_html = (
        f'<div style="text-align:center;margin:10px 0;">'
        f'<h4>Ground Truth</h4>'
        f'{_digit_box("Digit A", digits[0], "correct")}'
        f'<span style="font-size:36px;vertical-align:middle;"> + </span>'
        f'{_digit_box("Digit B", digits[1], "correct")}'
        f'<span style="font-size:36px;vertical-align:middle;"> = </span>'
        f'{_digit_box("Sum", target_sum, "correct")}'
        f'</div>'
    )

    visual_html = pred_html + corr_html + truth_html

    result_text = (
        f"## Results\n\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| True Digits | {digits[0]} + {digits[1]} = {target_sum} |\n"
        f"| Predicted | {prediction.labels[0]} + {prediction.labels[1]} = {int(prediction.labels.sum())} |\n"
        f"| Corrected | {correction.labels[0]} + {correction.labels[1]} = {int(correction.labels.sum())} |\n"
        f"| Prediction Correct | {'Yes' if metrics['prediction_correct'] else 'No'} |\n"
        f"| Correction Correct | {'Yes' if metrics['correction_correct'] else 'No'} |\n"
        f"| Digit Accuracy Before | {metrics['digit_accuracy_before']:.0%} |\n"
        f"| Digit Accuracy After | {metrics['digit_accuracy_after']:.0%} |\n"
        f"| Flagged Positions | {reflection.num_flagged} |\n"
        f"| Solver Time | {solve_ms:.1f} ms |\n"
    )

    return visual_html, result_text


# ══════════════════════════════════════════════════════════════════
# Batch benchmark
# ══════════════════════════════════════════════════════════════════

def run_benchmark(num_puzzles: int, difficulty: str, error_rate: float,
                  solver_type: str):
    """Run benchmark and return summary."""
    kb = build_sudoku_kb()
    solver = create_solver(solver_type)

    pred_accs, corr_accs, improvements, times = [], [], [], []
    consistent_count = 0

    for i in range(int(num_puzzles)):
        seed = i * 137 + 42
        _, solution = generate_sudoku(difficulty=difficulty, seed=seed)
        prediction = simulate_noisy_predictions(solution, error_rate=error_rate, seed=seed)

        scores = 1.0 - prediction.confidence
        errors = prediction.labels != solution
        for j in range(len(scores)):
            if errors[j]:
                scores[j] = min(scores[j] + 0.3, 1.0)

        flags = (scores >= 0.5).astype(int)
        reflection = Reflection(flags=flags, scores=scores, threshold=0.5)

        t0 = time.perf_counter()
        correction = solver.solve(prediction, reflection, kb, timeout_ms=5000)
        elapsed = (time.perf_counter() - t0) * 1000

        metrics = evaluate_sudoku_correction(prediction, solution, correction.labels)
        pred_accs.append(metrics["prediction_accuracy"])
        corr_accs.append(metrics["correction_accuracy"])
        improvements.append(metrics["improvement"])
        times.append(elapsed)
        if correction.consistent:
            consistent_count += 1

    n = int(num_puzzles)
    result = (
        f"## Benchmark Results ({n} puzzles, {difficulty})\n\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| Prediction Accuracy | {np.mean(pred_accs):.1%} +/- {np.std(pred_accs):.1%} |\n"
        f"| Correction Accuracy | {np.mean(corr_accs):.1%} +/- {np.std(corr_accs):.1%} |\n"
        f"| Average Improvement | **+{np.mean(improvements):.1%}** |\n"
        f"| Consistency Rate | {consistent_count}/{n} ({consistent_count/n:.0%}) |\n"
        f"| Avg Solve Time | {np.mean(times):.1f} ms |\n"
        f"| Min Solve Time | {np.min(times):.1f} ms |\n"
        f"| Max Solve Time | {np.max(times):.1f} ms |\n"
    )
    return result


# ══════════════════════════════════════════════════════════════════
# Gradio app builder
# ══════════════════════════════════════════════════════════════════

def create_app():
    """Create and return the Gradio app."""
    try:
        import gradio as gr
    except ImportError:
        raise ImportError(
            "Web UI requires gradio. Install with:\n"
            "  pip install gradio"
        )

    with gr.Blocks(
        title="ReflectAI — Neuro-Symbolic Reasoning",
    ) as app:

        gr.Markdown(
            "# ReflectAI\n"
            "### Neuro-symbolic reasoning with abductive reflection\n"
            "*First open-source implementation of ABL-Refl (AAAI 2026 Outstanding Paper)*\n\n"
            "**How it works:** Neural network predicts &rarr; Reflection head flags errors "
            "&rarr; Abductive solver corrects only flagged positions"
        )

        # ---- Tab 1: Sudoku ----
        with gr.Tab("Sudoku Solver"):
            gr.Markdown("Generate a Sudoku puzzle, simulate noisy neural predictions, "
                        "and watch the reflection + abductive reasoning pipeline correct errors.")

            with gr.Row():
                with gr.Column(scale=1):
                    sud_difficulty = gr.Dropdown(
                        ["easy", "medium", "hard"], value="medium",
                        label="Difficulty",
                    )
                    sud_error = gr.Slider(
                        0.0, 0.5, value=0.15, step=0.01,
                        label="Error Rate (simulated neural network mistakes)",
                    )
                    sud_threshold = gr.Slider(
                        0.1, 0.9, value=0.5, step=0.05,
                        label="Reflection Threshold (higher = fewer flags)",
                    )
                    sud_solver = gr.Dropdown(
                        ["backtrack", "z3"], value="backtrack",
                        label="Solver Backend",
                    )
                    sud_seed = gr.Number(value=42, label="Random Seed", precision=0)
                    sud_btn = gr.Button("Solve Sudoku", variant="primary", size="lg")

                with gr.Column(scale=2):
                    sud_grids = gr.HTML(label="Grids")
                    sud_metrics = gr.Markdown(label="Results")

            sud_btn.click(
                fn=solve_sudoku,
                inputs=[sud_difficulty, sud_error, sud_threshold, sud_solver, sud_seed],
                outputs=[sud_grids, sud_metrics],
            )

        # ---- Tab 2: Addition ----
        with gr.Tab("Digit Addition"):
            gr.Markdown("Simulate recognizing two handwritten digits and verifying "
                        "their sum with abductive reasoning.")

            with gr.Row():
                with gr.Column(scale=1):
                    add_a = gr.Slider(0, 9, value=3, step=1, label="Digit A")
                    add_b = gr.Slider(0, 9, value=4, step=1, label="Digit B")
                    add_error = gr.Slider(
                        0.0, 1.0, value=0.5, step=0.05,
                        label="Error Rate",
                    )
                    add_solver = gr.Dropdown(
                        ["backtrack", "z3"], value="backtrack",
                        label="Solver",
                    )
                    add_seed = gr.Number(value=10, label="Random Seed", precision=0)
                    add_btn = gr.Button("Solve Addition", variant="primary", size="lg")

                with gr.Column(scale=2):
                    add_visual = gr.HTML(label="Visual")
                    add_metrics = gr.Markdown(label="Results")

            add_btn.click(
                fn=solve_addition,
                inputs=[add_a, add_b, add_error, add_solver, add_seed],
                outputs=[add_visual, add_metrics],
            )

        # ---- Tab 3: Benchmark ----
        with gr.Tab("Benchmark"):
            gr.Markdown("Run the pipeline on multiple puzzles and see aggregate statistics.")

            with gr.Row():
                with gr.Column(scale=1):
                    bench_n = gr.Slider(5, 100, value=20, step=5,
                                        label="Number of Puzzles")
                    bench_diff = gr.Dropdown(
                        ["easy", "medium", "hard"], value="medium",
                        label="Difficulty",
                    )
                    bench_error = gr.Slider(0.0, 0.5, value=0.15, step=0.01,
                                             label="Error Rate")
                    bench_solver = gr.Dropdown(
                        ["backtrack", "z3"], value="backtrack", label="Solver",
                    )
                    bench_btn = gr.Button("Run Benchmark", variant="primary", size="lg")

                with gr.Column(scale=2):
                    bench_result = gr.Markdown(label="Results")

            bench_btn.click(
                fn=run_benchmark,
                inputs=[bench_n, bench_diff, bench_error, bench_solver],
                outputs=[bench_result],
            )

        # ---- Tab 4: How It Works ----
        with gr.Tab("How It Works"):
            gr.Markdown("""
## ABL-Refl Architecture

**ReflectAI** implements the ABL-Refl framework from the AAAI 2026 Outstanding Paper.

### Pipeline Steps

1. **Perception (Neural Network)**
   - A CNN or MLP processes input (e.g., digit images)
   - Outputs class probabilities for each position

2. **Reflection Head (Core Innovation)**
   - A binary classifier that shares features with the output head
   - Learns to predict which positions are likely **wrong**
   - Outputs a binary mask: 1 = suspect, 0 = trusted
   - Trained to maintain ~20% flagging rate (not too many, not too few)

3. **Abductive Reasoning**
   - Takes predicted labels + reflection flags
   - Fixes **only flagged positions** to satisfy constraints
   - Uses backtracking or Z3 SMT solver
   - This targeted search is **10,000-15,000x faster** than brute-force

### Three-Loss Training

```
L_total = L_supervised + lambda_c * L_consistency + lambda_r * L_reflection_size
```

- **L_supervised**: Cross-entropy on labeled data
- **L_consistency**: REINFORCE reward when correction improves constraint satisfaction
- **L_reflection_size**: Regularizer to keep flagging rate around 20%

### Key Results (from paper)

| Metric | Value |
|--------|-------|
| Sudoku Accuracy | 97.4% (vs 76.5% baseline) |
| Error Detection Recall | 99.04% |
| Speed Improvement | 10,000-15,000x |
| Data Efficiency | 10x fewer labels needed |

### Color Legend (Sudoku grids)
- **Green**: Correctly predicted
- **Red**: Neural network error
- **Yellow**: Flagged by reflection head
- **Blue**: Successfully corrected by solver
            """)

    return app


def launch_app(port: int = 7860, share: bool = False):
    """Create and launch the Gradio app."""
    import gradio as gr
    app = create_app()
    app.launch(server_port=port, share=share, theme=gr.themes.Soft(), css=_CSS)
