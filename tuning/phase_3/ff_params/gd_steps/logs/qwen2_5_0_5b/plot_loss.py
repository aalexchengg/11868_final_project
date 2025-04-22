import re
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.cm as cm  # Import colormap
import numpy as np  # For color generation
import argparse
import os
from collections import deque


# --- Unified Parsing ---


def parse_log_file(log_path):
    """
    Parses a log file, detecting if it's regular or FF training.
    Returns a dictionary containing parsed data and run type.
    """
    log_entries = []
    opt_pattern = re.compile(r"Step (\d+) \| loss:([\d.e+-]+)")
    ff_pattern = re.compile(r"Step (\d+) \| ff_loss:([\d.e+-]+) ff_step:(\d+)")
    has_ff_steps = False

    # --- Pass 1: Read all relevant entries ---
    try:
        with open(log_path, "r") as f:
            for line_num, line in enumerate(f):  # Use line_num for stable sort
                opt_match = opt_pattern.search(line)
                ff_match = ff_pattern.search(line)
                log_step = -1
                entry_type = None
                data = None

                if opt_match:
                    log_step = int(opt_match.group(1))
                    loss = float(opt_match.group(2))
                    entry_type = "opt"
                    data = loss
                elif ff_match:
                    log_step = int(ff_match.group(1))
                    entry_type = "ff"
                    data = None  # Loss value from FF steps not used directly here
                    has_ff_steps = True

                if entry_type:
                    log_entries.append(
                        {
                            "log_step": log_step,
                            "line_num": line_num,
                            "type": entry_type,
                            "loss": data,
                        }
                    )
    except FileNotFoundError:
        print(f"Error: Log file not found at {log_path}")
        return None
    except Exception as e:
        print(f"Error reading log file {log_path}: {e}")
        return None

    if not log_entries:
        print(f"Warning: No relevant log entries found in {log_path}")
        return {"type": "empty", "data": [], "ff_connections": []}

    # Sort primarily by log_step, secondarily by line number for stability
    log_entries.sort(key=lambda x: (x["log_step"], x["line_num"]))

    # --- Pass 2: Calculate cumulative steps and identify points ---
    optimizer_points_cumulative = []  # List of tuples: (cumulative_step, loss)
    optimizer_log_info = []  # List of tuples: (log_step, cumulative_step, loss)
    regular_steps_losses = []  # List of tuples: (original_step, loss)
    cumulative_step = 0
    current_log_step = -1

    for entry in log_entries:
        # Use original step number for regular, cumulative for FF plotting axis
        is_new_opt_step = (
            entry["type"] == "opt" and entry["log_step"] != current_log_step
        )
        if is_new_opt_step:
            current_log_step = entry["log_step"]
            regular_steps_losses.append((entry["log_step"], entry["loss"]))

        # Cumulative step always increments
        cumulative_step += 1
        entry["cumulative_step"] = cumulative_step

        if entry["type"] == "opt":
            loss = entry["loss"]
            optimizer_points_cumulative.append((cumulative_step, loss))
            optimizer_log_info.append((entry["log_step"], cumulative_step, loss))

    # --- Pass 3: Identify FF connecting pairs (only if FF steps occurred) ---
    ff_connecting_pairs = []
    if has_ff_steps and len(optimizer_log_info) >= 2:
        for i in range(len(optimizer_log_info) - 1):
            opt1_log_step, opt1_cs, opt1_loss = optimizer_log_info[i]
            opt2_log_step, opt2_cs, opt2_loss = optimizer_log_info[i + 1]

            # Check if any FF entry exists strictly between the log steps of these two optimizer steps
            found_ff_between = False
            for entry in log_entries:
                if (
                    opt1_log_step < entry["log_step"] < opt2_log_step
                    and entry["type"] == "ff"
                ):
                    found_ff_between = True
                    break

            if found_ff_between:
                ff_connecting_pairs.append(((opt1_cs, opt1_loss), (opt2_cs, opt2_loss)))

    # --- Determine run type and return structured data ---
    if has_ff_steps:
        return {
            "type": "ff",
            "data": optimizer_points_cumulative,  # (cumulative_step, loss) for scatter points
            "ff_connections": ff_connecting_pairs,
        }
    else:
        # Use original steps for regular runs
        return {
            "type": "regular",
            "data": regular_steps_losses,  # (original_step, loss) for line plot
            "ff_connections": [],
        }


# --- Plotting ---


def _setup_plot_style(ax, title=""):
    """Applies common styling to the plot axes."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.linewidth"] = 1.5
    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["scatter.marker"] = "."  # Default marker, can be overridden
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams["xtick.labelsize"] = 16
    plt.rcParams["ytick.labelsize"] = 16
    plt.rcParams["legend.fontsize"] = (
        14  # Slightly smaller legend for potentially many items
    )
    plt.rcParams["axes.titlesize"] = 18

    ax.set_xlabel("Step")  # Use generic "Step"
    ax.set_ylabel("Loss")
    ax.tick_params(axis="both", which="major", width=1.5, length=6)
    ax.set_title(title)

    # Make plot area border visible
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.5)


def _save_plot(fig, output_path):
    """Saves the plot to a file."""
    try:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_path}")
    except Exception as e:
        print(f"Error saving plot to {output_path}: {e}")
    plt.close(fig)  # Close the figure to free memory


def calculate_moving_average(data, window_size):
    """Calculates the moving average for a list of (step, value) tuples."""
    if window_size <= 0 or not data or len(data) < window_size:
        return data  # Return original data if window is invalid or too large

    steps = np.array([p[0] for p in data])
    values = np.array([p[1] for p in data])

    # Use convolution for efficient moving average calculation
    # 'valid' mode ensures output length matches input length - window_size + 1
    moving_avg = np.convolve(values, np.ones(window_size) / window_size, mode="valid")

    # The steps corresponding to the moving average values start after the first window/2 points
    # For simplicity and plotting, we align the MA point with the *end* of the window.
    ma_steps = steps[window_size - 1 :]

    return list(zip(ma_steps, moving_avg))


def plot_multiple_runs(
    run_data_list, output_path, max_steps=None, moving_average_window=0
):
    """
    Generates and saves a plot comparing multiple training runs.

    Args:
        run_data_list: A list of dictionaries, where each dict has 'label' and 'parsed_data'.
                       'parsed_data' is the output from parse_log_file.
        output_path: The path to save the output plot image.
        max_steps: Optional maximum step to plot up to.
        moving_average_window: Optional window size for calculating moving average.
    """
    if not run_data_list:
        print("No run data provided to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))  # Wider plot for multiple lines
    plot_title = "Training Loss Comparison"
    if moving_average_window and moving_average_window > 0:
        plot_title += f" (MA Window: {moving_average_window})"
    _setup_plot_style(ax, title=plot_title)

    colors = cm.viridis(np.linspace(0, 1, len(run_data_list)))  # Generate colors
    handles = []  # For legend
    all_steps = []
    all_losses = []
    any_ff_runs = False

    # --- Filter data first ---
    filtered_run_data_list = []
    for i, run in enumerate(run_data_list):
        parsed_data = run["parsed_data"]
        label = run["label"]
        run_type = parsed_data.get("type", "empty")

        if run_type == "empty" or not parsed_data.get("data"):
            print(f"Skipping run '{label}' due to no data.")
            continue

        data_points = list(parsed_data["data"])  # Make a copy to filter
        ff_connections = list(parsed_data.get("ff_connections", []))

        if max_steps is not None:
            # Filter based on step (first element of tuples in data)
            original_count = len(data_points)
            data_points = [p for p in data_points if p[0] <= max_steps]

            if run_type == "ff":
                # Also filter connections: both points must be <= max_steps
                ff_connections = [
                    pair
                    for pair in ff_connections
                    if pair[0][0] <= max_steps and pair[1][0] <= max_steps
                ]
                # Note: We filter based on cumulative steps for FF runs
                if original_count > 0 and not data_points:
                    print(
                        f"Warning: Run '{label}' (FF) has no optimizer points <= {max_steps} steps."
                    )

            elif run_type == "regular":
                # Note: We filter based on original steps for regular runs
                if original_count > 0 and not data_points:
                    print(
                        f"Warning: Run '{label}' (Regular) has no points <= {max_steps} steps."
                    )

        if not data_points:
            print(f"Skipping run '{label}' after filtering by max_steps={max_steps}.")
            continue

        # Store filtered data
        run["parsed_data"]["data"] = data_points
        run["parsed_data"]["ff_connections"] = ff_connections
        filtered_run_data_list.append(run)

        # Calculate moving average if requested
        plot_data = data_points
        if moving_average_window and moving_average_window > 0:
            plot_data = calculate_moving_average(data_points, moving_average_window)
            run["plot_data"] = plot_data  # Store for plotting phase
        else:
            run["plot_data"] = data_points

        # Collect all points for axis limits (use *original* filtered data for limits)
        steps = [p[0] for p in data_points]
        losses = [p[1] for p in data_points]
        all_steps.extend(steps)
        all_losses.extend(losses)
        # Collect moving average points for limits as well if calculated
        if moving_average_window and moving_average_window > 0:
            ma_steps = [p[0] for p in plot_data]
            ma_losses = [p[1] for p in plot_data]
            all_steps.extend(ma_steps)
            all_losses.extend(ma_losses)

        if run_type == "ff":
            any_ff_runs = True

    if not filtered_run_data_list:
        print(f"No data left to plot after filtering with max_steps={max_steps}.")
        plt.close(fig)
        return

    # --- Plot filtered data ---
    ff_legend_added = False
    for i, run in enumerate(filtered_run_data_list):
        label = run["label"]
        parsed_data = run["parsed_data"]
        run_type = parsed_data["type"]
        plot_data = run["plot_data"]  # Use pre-calculated (potentially MA) data
        color = colors[i]

        if not plot_data:
            print(
                f"Skipping plot for run '{label}' as it has no data points (check MA window/filtering)."
            )
            continue

        steps = [p[0] for p in plot_data]
        losses = [p[1] for p in plot_data]

        if run_type == "regular":
            (line,) = ax.plot(steps, losses, color=color, label=label)
            handles.append(line)
        elif run_type == "ff":
            # Plot only the (moving average) line for FF runs for clarity
            ff_label = f"{label} (FF Run)"
            # If using MA, add indication
            # ff_label += f" (MA {moving_average_window})" if moving_average_window and moving_average_window > 0 else ""
            (line,) = ax.plot(steps, losses, color=color, linestyle="-", label=ff_label)
            handles.append(line)

    # --- Finalize Plot ---
    if all_steps:
        ax.set_xlim(left=0, right=max(all_steps) * 1.05)
    if all_losses:
        # Add buffer only if min loss is positive
        min_loss = min(all_losses)
        max_loss = max(all_losses)
        bottom_lim = min_loss * 0.95 if min_loss > 0 else min_loss * 1.05
        top_lim = max_loss * 1.05 if max_loss > 0 else max_loss * 0.95
        # Handle case where min and max are the same or very close
        if abs(top_lim - bottom_lim) < 1e-9:
            top_lim += 0.1
            bottom_lim -= 0.1
        ax.set_ylim(bottom=bottom_lim, top=top_lim)

    ax.legend(handles=handles, frameon=True, edgecolor="lightgrey", loc="upper right")
    plt.tight_layout()
    _save_plot(fig, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot and compare training loss from multiple log files (regular or FF).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log-files",
        required=True,
        nargs="+",  # Accept multiple log files
        help="Paths to the log files for training runs.",
    )
    parser.add_argument(
        "--labels",
        required=True,
        nargs="+",  # Accept multiple labels
        help="Labels for each corresponding log file provided via --log-files.",
    )
    parser.add_argument(
        "-o",
        "--output-path",  # Changed from prefix to single path
        default="loss_comparison_plot.png",
        help="Path for the output comparison plot image file (e.g., 'comparison.png').",
    )
    parser.add_argument(
        "-n",
        "--max-steps",
        type=int,
        default=None,
        help="Only plot data up to this maximum step number (uses cumulative step for FF, original step for regular).",
    )
    parser.add_argument(
        "-w",
        "--moving-average-window",
        type=int,
        default=0,
        help="Window size for moving average smoothing. If 0 or less, no smoothing is applied.",
    )

    args = parser.parse_args()

    if len(args.log_files) != len(args.labels):
        print("Error: The number of log files must match the number of labels.")
        exit(1)

    # --- Parse all log files ---
    all_run_data = []
    for log_path, label in zip(args.log_files, args.labels):
        print(f"Parsing log file: {log_path} with label: {label}")
        parsed_data = parse_log_file(log_path)
        if parsed_data:
            all_run_data.append({"label": label, "parsed_data": parsed_data})
        else:
            print(f"Failed to parse or empty log file: {log_path}. Skipping.")

    if not all_run_data:
        print("No valid log data found to plot.")
        exit(1)

    # --- Generate combined plot ---
    # Filtering is now done inside plot_multiple_runs
    plot_multiple_runs(
        all_run_data, args.output_path, args.max_steps, args.moving_average_window
    )

    print("Script finished.")
