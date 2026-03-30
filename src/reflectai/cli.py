"""Command-line interface for ReflectAI.

Commands:
  reflectai demo       — Run a demo with synthetic Sudoku
  reflectai solve      — Solve a Sudoku puzzle from file
  reflectai benchmark  — Benchmark the pipeline
  reflectai info       — Show system information
"""

from __future__ import annotations

import json
import sys
import time

import click
import numpy as np

from . import __version__


@click.group()
@click.version_option(__version__, prog_name="reflectai")
def main():
    """ReflectAI — Neuro-symbolic reasoning with abductive reflection."""
    pass


@main.command()
@click.option("--difficulty", "-d", default="easy",
              type=click.Choice(["easy", "medium", "hard"]),
              help="Sudoku difficulty level")
@click.option("--error-rate", "-e", default=0.15, type=float,
              help="Simulated neural network error rate")
@click.option("--seed", "-s", default=42, type=int, help="Random seed")
@click.option("--solver", default="backtrack",
              type=click.Choice(["backtrack", "z3"]),
              help="Constraint solver backend")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
def demo(difficulty: str, error_rate: float, seed: int,
         solver: str, verbose: bool):
    """Run a demo with synthetic Sudoku puzzle."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        console = Console()
    except ImportError:
        console = None

    from .tasks.sudoku import (
        generate_sudoku, simulate_noisy_predictions, evaluate_sudoku_correction,
    )
    from .knowledge import build_sudoku_kb
    from .models import Reflection
    from .reasoner import create_solver

    # Generate puzzle
    if console:
        console.print(Panel(
            f"[bold cyan]ReflectAI Demo[/bold cyan] — Sudoku ({difficulty})\n"
            f"Error rate: {error_rate:.0%} | Solver: {solver} | Seed: {seed}",
            title="🧩 ReflectAI",
        ))

    puzzle, solution = generate_sudoku(difficulty=difficulty, seed=seed)
    kb = build_sudoku_kb()

    # Simulate noisy predictions
    prediction = simulate_noisy_predictions(
        solution, error_rate=error_rate, seed=seed
    )

    errors = (prediction.labels != solution)
    num_errors = errors.sum()

    if console:
        console.print(f"\n[bold]Prediction:[/bold] {num_errors} errors in {len(solution)} cells")
    else:
        click.echo(f"Prediction: {num_errors} errors in {len(solution)} cells")

    # Simulate reflection (flag low-confidence + known error patterns)
    scores = 1.0 - prediction.confidence
    # Boost scores for actual errors (simulating trained reflection head)
    for i in range(len(scores)):
        if errors[i]:
            scores[i] = min(scores[i] + 0.3, 1.0)

    threshold = 0.5
    flags = (scores >= threshold).astype(int)
    reflection = Reflection(flags=flags, scores=scores, threshold=threshold)

    if console:
        console.print(f"[bold]Reflection:[/bold] {reflection.num_flagged} positions flagged "
                       f"({reflection.flag_rate:.0%} flag rate)")
    else:
        click.echo(f"Reflection: {reflection.num_flagged} positions flagged")

    # Abductive reasoning
    abductive_solver = create_solver(solver)
    t0 = time.perf_counter()
    correction = abductive_solver.solve(prediction, reflection, kb, timeout_ms=5000)
    solve_time = (time.perf_counter() - t0) * 1000

    # Evaluate
    metrics = evaluate_sudoku_correction(prediction, solution, correction.labels)

    if console:
        table = Table(title="Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Prediction Accuracy", f"{metrics['prediction_accuracy']:.1%}")
        table.add_row("Correction Accuracy", f"{metrics['correction_accuracy']:.1%}")
        table.add_row("Improvement", f"+{metrics['improvement']:.1%}")
        table.add_row("Errors Fixed", str(metrics['errors_fixed']))
        table.add_row("Errors Introduced", str(metrics['errors_introduced']))
        table.add_row("Consistent", "Yes" if correction.consistent else "No")
        table.add_row("Solver Time", f"{solve_time:.1f}ms")

        console.print(table)

        if verbose:
            console.print("\n[bold]Puzzle Grid:[/bold]")
            _print_grid(console, puzzle, "Puzzle (0=empty)")
            _print_grid(console, prediction.labels, "Neural Predictions")
            _print_grid(console, correction.labels, "After Correction")
            _print_grid(console, solution, "Ground Truth")
    else:
        click.echo(f"\nPrediction Accuracy: {metrics['prediction_accuracy']:.1%}")
        click.echo(f"Correction Accuracy: {metrics['correction_accuracy']:.1%}")
        click.echo(f"Improvement: +{metrics['improvement']:.1%}")
        click.echo(f"Consistent: {correction.consistent}")
        click.echo(f"Solver Time: {solve_time:.1f}ms")


@main.command()
@click.option("--num-puzzles", "-n", default=10, type=int,
              help="Number of puzzles to benchmark")
@click.option("--difficulty", "-d", default="medium",
              type=click.Choice(["easy", "medium", "hard"]))
@click.option("--error-rate", "-e", default=0.15, type=float)
@click.option("--solver", default="backtrack",
              type=click.Choice(["backtrack", "z3"]))
def benchmark(num_puzzles: int, difficulty: str, error_rate: float,
              solver: str):
    """Benchmark the ReflectAI pipeline on multiple puzzles."""
    try:
        from rich.console import Console
        from rich.progress import Progress
        console = Console()
        use_rich = True
    except ImportError:
        console = None
        use_rich = False

    from .tasks.sudoku import (
        generate_sudoku, simulate_noisy_predictions, evaluate_sudoku_correction,
    )
    from .knowledge import build_sudoku_kb
    from .models import Reflection
    from .reasoner import create_solver

    kb = build_sudoku_kb()
    abductive_solver = create_solver(solver)

    all_metrics = []
    times = []

    iterator = range(num_puzzles)
    if use_rich:
        from rich.progress import Progress
        progress = Progress(console=console)
        task = progress.add_task("Benchmarking...", total=num_puzzles)
        progress.start()

    for i in iterator:
        seed = i * 137 + 42
        _, solution = generate_sudoku(difficulty=difficulty, seed=seed)
        prediction = simulate_noisy_predictions(solution, error_rate=error_rate, seed=seed)

        # Simple reflection simulation
        scores = 1.0 - prediction.confidence
        errors = prediction.labels != solution
        for j in range(len(scores)):
            if errors[j]:
                scores[j] = min(scores[j] + 0.3, 1.0)

        flags = (scores >= 0.5).astype(int)
        reflection = Reflection(flags=flags, scores=scores, threshold=0.5)

        t0 = time.perf_counter()
        correction = abductive_solver.solve(prediction, reflection, kb, timeout_ms=5000)
        elapsed = (time.perf_counter() - t0) * 1000

        metrics = evaluate_sudoku_correction(prediction, solution, correction.labels)
        metrics["solve_time_ms"] = elapsed
        metrics["consistent"] = correction.consistent
        all_metrics.append(metrics)
        times.append(elapsed)

        if use_rich:
            progress.update(task, advance=1)

    if use_rich:
        progress.stop()

    # Aggregate
    pred_accs = [m["prediction_accuracy"] for m in all_metrics]
    corr_accs = [m["correction_accuracy"] for m in all_metrics]
    improvements = [m["improvement"] for m in all_metrics]
    consistent = sum(1 for m in all_metrics if m["consistent"])

    msg = (
        f"\n{'='*50}\n"
        f"Benchmark Results ({num_puzzles} puzzles, {difficulty})\n"
        f"{'='*50}\n"
        f"Prediction Accuracy: {np.mean(pred_accs):.1%} ± {np.std(pred_accs):.1%}\n"
        f"Correction Accuracy: {np.mean(corr_accs):.1%} ± {np.std(corr_accs):.1%}\n"
        f"Improvement:         +{np.mean(improvements):.1%}\n"
        f"Consistency Rate:    {consistent}/{num_puzzles} ({consistent/num_puzzles:.0%})\n"
        f"Avg Solve Time:      {np.mean(times):.1f}ms\n"
        f"{'='*50}"
    )

    if console:
        console.print(msg)
    else:
        click.echo(msg)


@main.command()
def info():
    """Show system information and available backends."""
    try:
        from rich.console import Console
        from rich.table import Table
        console = Console()
    except ImportError:
        console = None

    checks = {}

    # PyTorch
    try:
        import torch
        checks["PyTorch"] = f"{torch.__version__}"
        if torch.cuda.is_available():
            checks["CUDA"] = torch.version.cuda
        else:
            checks["CUDA"] = "Not available"
    except ImportError:
        checks["PyTorch"] = "Not installed"

    # NumPy
    try:
        checks["NumPy"] = np.__version__
    except Exception:
        checks["NumPy"] = "Not installed"

    # Z3
    try:
        import z3
        checks["Z3 Solver"] = "Available"
    except ImportError:
        checks["Z3 Solver"] = "Not installed (pip install reflectai[z3])"

    # PySAT
    try:
        from pysat.solvers import Solver
        checks["PySAT"] = "Available"
    except ImportError:
        checks["PySAT"] = "Not installed (pip install reflectai[sat])"

    # torchvision
    try:
        import torchvision
        checks["torchvision"] = torchvision.__version__
    except ImportError:
        checks["torchvision"] = "Not installed (pip install reflectai[vision])"

    if console:
        table = Table(title=f"ReflectAI v{__version__}")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")

        for k, v in checks.items():
            style = "green" if "Not" not in v else "yellow"
            table.add_row(k, f"[{style}]{v}[/{style}]")

        console.print(table)
    else:
        click.echo(f"ReflectAI v{__version__}")
        for k, v in checks.items():
            click.echo(f"  {k}: {v}")


@main.command()
@click.option("--port", "-p", default=7860, type=int, help="Server port")
@click.option("--share", is_flag=True, help="Create a public Gradio link")
def web(port: int, share: bool):
    """Launch the interactive web UI."""
    try:
        from .web import launch_app
    except ImportError:
        click.echo("Web UI requires gradio. Install with:")
        click.echo("  pip install gradio")
        raise SystemExit(1)

    click.echo(f"Starting ReflectAI Web UI on port {port}...")
    if share:
        click.echo("Creating public share link...")
    launch_app(port=port, share=share)


def _print_grid(console, values: np.ndarray, title: str):
    """Print a 9x9 Sudoku grid."""
    from rich.table import Table

    table = Table(title=title, show_lines=True)
    for c in range(9):
        table.add_column(str(c + 1), width=3, justify="center")

    grid = values.reshape(9, 9)
    for r in range(9):
        row = [str(v) if v > 0 else "." for v in grid[r]]
        table.add_row(*row)

    console.print(table)


if __name__ == "__main__":
    main()
