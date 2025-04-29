import re
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.cm as cm
import numpy as np
import argparse
import os
from collections import deque

# --- Log Parsing ---


def parse_training_log(log_path):
    """
    Parses a training log file to extract evaluation loss at each step.
    Focuses on lines containing 'Step ... eval_loss:...'
    Returns a list of tuples: (step, eval_loss).
    """
    data_points = []
    # Pattern for optimizer steps with eval_loss
    # Example: Step 3642 | train_loss:0.84... eval_loss:0.89... lr:...
    eval_pattern = re.compile(r"Step\s+(\d+)\s+\|.*eval_loss:([\d.e+-]+)")

    try:
        with open(log_path, "r") as f:
            for line_num, line in enumerate(f):
                match = eval_pattern.search(line)
                if match:
                    step = int(match.group(1))
                    eval_loss = float(match.group(2))
                    data_points.append((step, eval_loss))

    except FileNotFoundError:
        print(f"Error: Log file not found at {log_path}")
        return None
    except Exception as e:
        print(f"Error reading log file {log_path}: {e}")
        return None

    if not data_points:
        print(f"Warning: No eval_loss entries found in {log_path}")
        return []

    # Sort by step number just in case logs are out of order
    data_points.sort(key=lambda x: x[0])

    return data_points


# --- Plotting ---


def _setup_plot_style(ax, title="", xlabel="Step", ylabel="Evaluation Loss"):
    """Applies common styling to the plot axes."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.linewidth"] = 1.5
    plt.rcParams["lines.linewidth"] = 2  # Line width for plots
    plt.rcParams["axes.labelsize"] = 18  # Axis label size
    plt.rcParams["xtick.labelsize"] = 14  # X-tick label size
    plt.rcParams["ytick.labelsize"] = 14  # Y-tick label size
    plt.rcParams["legend.fontsize"] = 14  # Legend font size
    plt.rcParams["axes.titlesize"] = 18  # Plot title size
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="both", which="major", width=1.5, length=6)
    ax.set_title(title)
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.5)


def _save_plot(fig, output_path):
    """Saves the plot to a file."""
    try:
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_path}")
    except Exception as e:
        print(f"Error saving plot to {output_path}: {e}")
    plt.close(fig)


def calculate_moving_average(data, window_size):
    """Calculates the moving average for a list of (step, value) tuples."""
    if window_size <= 1 or not data or len(data) < window_size:
        return data  # Return original data if window is too small or no data
    steps = np.array([p[0] for p in data])
    values = np.array([p[1] for p in data])
    # Use 'valid' mode so output size is len(data) - window_size + 1
    moving_avg = np.convolve(values, np.ones(window_size) / window_size, mode="valid")
    # Adjust steps to align with the end of the window
    ma_steps = steps[window_size - 1 :]
    return list(zip(ma_steps, moving_avg))


def plot_comparison(
    run_data_list,
    output_path,
    title="Evaluation Loss Comparison",
    max_steps=None,
    moving_average_window=0,
):
    """
    Generates and saves a plot comparing eval_loss for multiple runs.

    Args:
        run_data_list: List of dicts, each {'label': str, 'data': list_of_tuples(step, loss)}.
        output_path: Path to save the plot image.
        title: Title for the plot.
        max_steps: Maximum step number to include in the plot.
        moving_average_window: Window size for moving average (0 to disable).
    """
    if not run_data_list:
        print("No run data provided to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    _setup_plot_style(ax, title=title)

    colors = cm.viridis(np.linspace(0.1, 0.9, len(run_data_list)))
    legend_handles = []
    all_steps_plot, all_losses_plot = [], []

    for i, run in enumerate(run_data_list):
        label = run["label"]
        raw_data = run["data"]
        color = colors[i]

        if not raw_data:
            print(f"Skipping run '{label}' due to no data.")
            continue

        # Filter by max_steps
        filtered_data = (
            [p for p in raw_data if p[0] <= max_steps]
            if max_steps is not None
            else raw_data
        )

        if not filtered_data:
            print(f"Skipping run '{label}' after filtering by max_steps={max_steps}.")
            continue

        # Apply moving average if requested
        plot_data = (
            calculate_moving_average(filtered_data, moving_average_window)
            if moving_average_window > 0
            else filtered_data
        )

        if not plot_data:
            print(
                f"Skipping run '{label}' after moving average calculation (window={moving_average_window})."
            )
            continue

        steps, losses = zip(*plot_data)

        # Plot the data
        (line,) = ax.plot(steps, losses, color=color, linestyle="-", label=label)
        legend_handles.append(line)

        # Collect points for overall axis limits (use filtered data before MA)
        steps_lim, losses_lim = zip(*filtered_data)
        all_steps_plot.extend(steps_lim)
        all_losses_plot.extend(losses_lim)

    if not legend_handles:
        print("No data was plotted.")
        plt.close(fig)
        return

    # --- Finalize Plot ---
    if all_steps_plot:
        min_step, max_step_val = min(all_steps_plot), max(all_steps_plot)
        xlim_left = max(0, min_step - (max_step_val - min_step) * 0.02)
        xlim_right = max_step_val * 1.05 if max_step_val > 0 else 1.0
        ax.set_xlim(left=xlim_left, right=xlim_right)
    if all_losses_plot:
        min_loss, max_loss = min(all_losses_plot), max(all_losses_plot)
        loss_range = max_loss - min_loss
        buffer = (
            loss_range * 0.05 if loss_range > 1e-9 else 0.1
        )  # Add buffer unless range is tiny
        # Ensure buffer doesn't push ylim below 0 if all losses are non-negative
        bottom_lim = max(0, min_loss - buffer) if min_loss >= 0 else min_loss - buffer
        top_lim = max_loss + buffer
        ax.set_ylim(bottom=bottom_lim, top=top_lim)
    else:  # Handle case where maybe only one point exists
        ax.set_ylim(bottom=0)  # Default bottom

    ax.legend(handles=legend_handles, frameon=True, edgecolor="lightgrey", loc="best")
    plt.tight_layout()
    _save_plot(fig, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot evaluation loss comparison from multiple training log files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log-files",
        required=True,
        nargs="+",
        help="Paths to the training log files.",
    )
    parser.add_argument(
        "--labels",
        required=True,
        nargs="+",
        help="Labels for each log file/run, corresponding to --log-files order.",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        default="eval_loss_comparison.png",
        help="Path to save the output plot image.",
    )
    parser.add_argument(
        "-t",
        "--title",
        default="Evaluation Loss Comparison",
        help="Title for the plot.",
    )
    parser.add_argument(
        "-n",
        "--max-steps",
        type=int,
        default=None,
        help="Only plot data up to this maximum step number.",
    )
    parser.add_argument(
        "-w",
        "--moving-average-window",
        type=int,
        default=1,  # Default to 1 (no smoothing)
        help="Window size for moving average smoothing. Set to 1 or 0 to disable.",
    )

    args = parser.parse_args()

    if len(args.log_files) != len(args.labels):
        print("Error: The number of log files must match the number of labels.")
        exit(1)

    print("Parsing log files...")
    all_run_data_parsed = []
    for log_path, label in zip(args.log_files, args.labels):
        print(f"  Parsing: {log_path} (Label: {label})")
        parsed_data = parse_training_log(log_path)
        if parsed_data is not None:  # Handle case where parsing returns None on error
            all_run_data_parsed.append({"label": label, "data": parsed_data})
        else:
            print(f"  Failed to parse or empty log: {log_path}")

    if not all_run_data_parsed:
        print("No valid data found in any log file. Exiting.")
        exit(1)

    # Adjust MA window if 0 is provided
    ma_window = args.moving_average_window if args.moving_average_window > 0 else 1

    print(f"\nGenerating plot: {args.output_path}")
    plot_comparison(
        run_data_list=all_run_data_parsed,
        output_path=args.output_path,
        title=args.title,
        max_steps=args.max_steps,
        moving_average_window=ma_window,
    )

    print("\nScript finished.")
