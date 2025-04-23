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
    Extracts eval_loss from regular steps and ff_loss from FF steps.
    Returns dict with run type and data points: (cumul_step, value, origin_type, log_step).
    """
    log_entries = []
    # Pattern for FF steps
    ff_pattern = re.compile(r"Step (\d+) \| ff_loss:([\d.e+-]+) ff_step:(\d+)")
    # Pattern for regular optimizer steps with eval_loss
    opt_eval_pattern = re.compile(
        r"Step (\d+) \| train_loss:[\d.e+-]+ eval_loss:([\d.e+-]+)"
    )
    # Pattern for regular optimizer steps WITHOUT eval_loss (Fallback)
    opt_pattern = re.compile(r"Step (\d+) \| loss:([\d.e+-]+)")
    has_ff_steps = False

    # --- Pass 1: Read all relevant entries ---
    try:
        with open(log_path, "r") as f:
            for line_num, line in enumerate(f):  # Use line_num for stable sort
                opt_eval_match = opt_eval_pattern.search(line)
                ff_match = ff_pattern.search(line)
                opt_match = opt_pattern.search(line)  # Check old format too

                log_step = -1
                entry_type = None
                value = None  # Store eval_loss, ff_loss, or loss here

                # Determine type and value
                if opt_eval_match:
                    log_step = int(opt_eval_match.group(1))
                    value = float(opt_eval_match.group(2))
                    entry_type = "opt_eval"  # Origin is eval loss
                elif ff_match:
                    log_step = int(ff_match.group(1))
                    value = float(ff_match.group(2))
                    entry_type = "ff"  # Origin is ff loss
                    has_ff_steps = True
                elif opt_match:  # Fallback to old 'loss' if 'eval_loss' not found
                    log_step = int(opt_match.group(1))
                    value = float(opt_match.group(2))
                    entry_type = "opt"  # Origin is regular loss

                if entry_type:
                    log_entries.append(
                        {
                            "log_step": log_step,
                            "line_num": line_num,
                            "type": entry_type,
                            "value": value,  # Store the relevant loss value
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
        return {"type": "empty", "data": []}

    # Sort primarily by log_step, secondarily by line number for stability
    log_entries.sort(key=lambda x: (x["log_step"], x["line_num"]))

    # --- Pass 2: Calculate cumulative steps and create unified data list ---
    all_points = []  # List of tuples: (cumulative_step, value, origin_type, log_step)
    cumulative_step = 0

    for entry in log_entries:
        cumulative_step += 1
        value = entry["value"]
        origin_type = entry["type"]
        log_step = entry["log_step"]
        if value is not None:
            all_points.append((cumulative_step, value, origin_type, log_step))

    # --- Determine run type and return structured data ---
    run_type = "regular" if not has_ff_steps else "ff"
    if not all_points:
        run_type = "empty"

    return {
        "type": run_type,
        "data": all_points,  # All points: (cumulative_step, value, origin_type, log_step)
    }


# --- Plotting Helper ---


def _setup_plot_style(ax, title="", xlabel="Step"):
    """Applies common styling to the plot axes."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.linewidth"] = 1.5
    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams["xtick.labelsize"] = 16
    plt.rcParams["ytick.labelsize"] = 16
    plt.rcParams["legend.fontsize"] = 14
    plt.rcParams["axes.titlesize"] = 18
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Loss")
    ax.tick_params(axis="both", which="major", width=1.5, length=6)
    ax.set_title(title)
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
    plt.close(fig)


def calculate_moving_average(data, window_size):
    """Calculates the moving average for a list of (step, value) tuples."""
    if window_size <= 0 or not data or len(data) < window_size:
        return data
    steps = np.array([p[0] for p in data])
    values = np.array([p[1] for p in data])
    moving_avg = np.convolve(values, np.ones(window_size) / window_size, mode="valid")
    ma_steps = steps[window_size - 1 :]
    return list(zip(ma_steps, moving_avg))


def _generate_plot(
    run_data_list, output_path, plot_type, max_steps=None, moving_average_window=0
):
    """
    Internal helper to generate and save a specific type of plot.
    'cumulative': Segmented FF runs, MA for regular.
    'gd_steps_only': Only GD steps (sequential count), MA applied to all.
    """
    if not run_data_list:
        print("No run data provided to _generate_plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    title = f"Training Loss Comparison ({plot_type.replace('_', ' ').title()})"
    xlabel = "Cumulative Step" if plot_type == "cumulative" else "GD Step Count"
    _setup_plot_style(ax, title=title, xlabel=xlabel)

    colors = cm.viridis(np.linspace(0.1, 0.9, len(run_data_list)))
    legend_handles = []
    all_steps_plot, all_losses_plot = [], []

    plot_data_prepared = []
    for i, run_orig in enumerate(run_data_list):
        run = run_orig.copy()
        run["parsed_data"] = run_orig["parsed_data"].copy()
        run["parsed_data"]["data"] = list(run_orig["parsed_data"].get("data", []))
        run_type = run["parsed_data"].get("type", "empty")

        if run_type == "empty" or not run["parsed_data"]["data"]:
            continue

        data_points = run["parsed_data"]["data"]
        # These hold the data *after* filtering for the specific plot type
        plot_points_final_full = []  # Holds full tuples for cumulative FF plot
        line_plot_pairs = []  # Holds (x, y) pairs for line plotting / MA
        plot_run_type_for_plot = run_type
        x_axis_points_for_limits = []
        y_axis_points_for_limits = []

        if plot_type == "gd_steps_only":
            gd_points_raw = [p for p in data_points if p[2] != "ff"]
            if max_steps is not None:
                gd_points_filtered = [p for p in gd_points_raw if p[3] <= max_steps]
            else:
                gd_points_filtered = gd_points_raw

            gd_step_count = 0
            for p in gd_points_filtered:
                gd_step_count += 1
                line_plot_pairs.append((gd_step_count, p[1]))
            plot_run_type_for_plot = "regular"
            x_axis_points_for_limits = [p[0] for p in line_plot_pairs]
            y_axis_points_for_limits = [p[1] for p in line_plot_pairs]

        elif plot_type == "cumulative":
            if max_steps is not None:
                cumul_points_filtered = [p for p in data_points if p[0] <= max_steps]
            else:
                cumul_points_filtered = data_points
            # Store full tuples for potential segmented plotting
            plot_points_final_full = cumul_points_filtered
            # Also create the (x,y) pairs needed for regular lines or MA base
            line_plot_pairs = [(p[0], p[1]) for p in cumul_points_filtered]
            x_axis_points_for_limits = [p[0] for p in line_plot_pairs]
            y_axis_points_for_limits = [p[1] for p in line_plot_pairs]
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}")

        # Check if there are any points left to plot for this run
        if not x_axis_points_for_limits:  # If no x-values, skip run for this plot
            print(
                f"Skipping run '{run['label']}' for plot '{plot_type}' after filtering."
            )
            continue

        # Store the prepared data for the plotting loop
        run["plot_points_full"] = plot_points_final_full  # Used by cumulative FF plot
        run["line_plot_pairs"] = line_plot_pairs  # Used by regular plot and MA
        run["plot_run_type"] = plot_run_type_for_plot
        run["color"] = colors[i]
        plot_data_prepared.append(run)

        # Collect points for overall axis limits
        all_steps_plot.extend(x_axis_points_for_limits)
        all_losses_plot.extend(y_axis_points_for_limits)

    if not plot_data_prepared:
        print(f"No data left to plot for plot type '{plot_type}'.")
        plt.close(fig)
        return

    # --- Plotting Logic ---
    for run in plot_data_prepared:
        label = run["label"]
        plot_run_type = run["plot_run_type"]
        # Use the pre-calculated pairs for line plotting
        line_points_for_plot = run["line_plot_pairs"]
        base_color = run["color"]

        if plot_run_type == "regular":  # Handles regular runs and gd_steps_only plot
            plot_steps, plot_losses = [], []
            if line_points_for_plot:
                _apply_ma = moving_average_window > 0
                if _apply_ma:
                    smoothed = calculate_moving_average(
                        line_points_for_plot, moving_average_window
                    )
                    if smoothed:
                        plot_steps, plot_losses = zip(*smoothed)
                else:
                    # This list now guaranteed contains (step, value) pairs
                    if line_points_for_plot:
                        plot_steps, plot_losses = zip(*line_points_for_plot)

            if plot_steps:
                (line,) = ax.plot(
                    plot_steps,
                    plot_losses,
                    color=base_color,
                    linestyle="-",
                    label=label,
                )
                legend_handles.append(line)

        elif plot_run_type == "ff" and plot_type == "cumulative":
            # Segmented plot for cumulative FF - uses plot_points_full
            data_points_ff = run["plot_points_full"]  # Use the full data tuples here
            if moving_average_window > 0:
                print(
                    f"Warning: MA ignored for cumulative FF run '{label}' due to segmented plotting."
                )
            if not data_points_ff:
                continue

            start_index, first_segment_plotted, last_point = 0, False, None
            for i in range(1, len(data_points_ff)):
                current_type = data_points_ff[i][2]
                prev_type = data_points_ff[i - 1][2]
                is_last = i == len(data_points_ff) - 1
                current_seg_type = "ff" if current_type == "ff" else "eval"
                prev_seg_type = "ff" if prev_type == "ff" else "eval"

                if current_seg_type != prev_seg_type or is_last:
                    end_idx = i + 1 if is_last else i
                    seg_raw = data_points_ff[start_index:end_idx]
                    seg_steps = [p[0] for p in seg_raw]  # Use cumul_step (index 0)
                    seg_vals = [p[1] for p in seg_raw]
                    seg_color = "forestgreen" if prev_seg_type == "ff" else base_color
                    if last_point:
                        seg_steps.insert(0, last_point[0])
                        seg_vals.insert(0, last_point[1])

                    seg_label = label if not first_segment_plotted else None
                    (line,) = ax.plot(
                        seg_steps,
                        seg_vals,
                        color=seg_color,
                        linestyle="-",
                        label=seg_label,
                    )
                    if not first_segment_plotted:
                        legend_handles.append(line)
                        first_segment_plotted = True
                    last_point = (seg_steps[-1], seg_vals[-1])
                    start_index = i

            if (
                not first_segment_plotted and data_points_ff
            ):  # Handle single-segment case
                seg_type = "ff" if data_points_ff[0][2] == "ff" else "eval"
                seg_steps = [p[0] for p in data_points_ff]
                seg_vals = [p[1] for p in data_points_ff]
                seg_color = "forestgreen" if seg_type == "ff" else base_color
                (line,) = ax.plot(
                    seg_steps, seg_vals, color=seg_color, linestyle="-", label=label
                )
                legend_handles.append(line)

    # --- Finalize Plot ---
    if all_steps_plot:
        min_step, max_step = min(all_steps_plot), max(all_steps_plot)
        xlim_left = (
            max(0, min_step - (max_step - min_step) * 0.02)
            if max_step > min_step
            else 0
        )
        xlim_right = max_step * 1.05 if max_step > 0 else 1.0
        ax.set_xlim(left=xlim_left, right=xlim_right)
    if all_losses_plot:
        min_loss, max_loss = min(all_losses_plot), max(all_losses_plot)
        loss_range = max_loss - min_loss
        buffer = loss_range * 0.05 if loss_range > 1e-9 else 0.1
        bottom_lim = min_loss - buffer
        top_lim = max_loss + buffer
        if min_loss >= 0 and bottom_lim < 0:
            bottom_lim = 0
        ax.set_ylim(bottom=bottom_lim, top=top_lim)

    if legend_handles:
        ax.legend(
            handles=legend_handles, frameon=True, edgecolor="lightgrey", loc="best"
        )
    else:
        print(f"Warning: No handles for legend in plot '{output_path}'.")

    plt.tight_layout()
    _save_plot(fig, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot training loss comparisons (Cumulative and GD steps).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Add arguments (log_files, labels, output_path, max_steps, moving_average_window)
    parser.add_argument("--log-files", required=True, nargs="+", help="Log files.")
    parser.add_argument(
        "--labels", required=True, nargs="+", help="Labels for log files."
    )
    parser.add_argument(
        "-o",
        "--output-path",
        default="loss_comparison.png",
        help="Base path for output plots.",
    )
    parser.add_argument(
        "-n", "--max-steps", type=int, default=None, help="Max step (cumulative or GD)."
    )
    parser.add_argument(
        "-w",
        "--moving-average-window",
        type=int,
        default=0,
        help="MA window (regular runs/GD plot).",
    )

    args = parser.parse_args()

    if len(args.log_files) != len(args.labels):
        print("Error: Number of log files must match labels.")
        exit(1)

    print("Parsing log files...")
    all_run_data = []
    for log_path, label in zip(args.log_files, args.labels):
        # print(f"Parsing: {log_path} ({label})") # Verbose parsing log
        parsed = parse_log_file(log_path)
        if parsed and parsed.get("type") != "empty":
            all_run_data.append({"label": label, "parsed_data": parsed})
        else:
            print(f"Skipping empty/failed parse: {log_path}")

    if not all_run_data:
        print("No valid data found.")
        exit(1)

    # --- Generate Cumulative Plot ---
    print(f"\nGenerating Cumulative Plot: {args.output_path}")
    _generate_plot(
        run_data_list=all_run_data,
        output_path=args.output_path,
        plot_type="cumulative",
        max_steps=args.max_steps,
        moving_average_window=args.moving_average_window,
    )

    # --- Generate GD Steps Only Plot ---
    base, ext = os.path.splitext(args.output_path)
    gd_output_path = f"{base}_gd_steps{ext if ext else '.png'}"
    print(f"\nGenerating GD Steps Only Plot: {gd_output_path}")
    _generate_plot(
        run_data_list=all_run_data,
        output_path=gd_output_path,
        plot_type="gd_steps_only",
        max_steps=args.max_steps,
        moving_average_window=args.moving_average_window,
    )

    print("\nScript finished.")
