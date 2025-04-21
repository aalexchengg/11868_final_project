import re
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import argparse
import os  # For joining paths


def parse_regular_log(log_path):
    """Parses the regular training log file."""
    regular_steps = []
    pattern = re.compile(r"Step (\d+) \| loss:([\d.e+-]+)")
    try:
        with open(log_path, "r") as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    step = int(match.group(1))
                    loss = float(match.group(2))
                    regular_steps.append((step, loss))
    except FileNotFoundError:
        print(f"Error: Regular log file not found at {log_path}")
        return []
    except Exception as e:
        print(f"Error reading regular log file: {e}")
        return []
    return sorted(regular_steps, key=lambda x: x[0])


def parse_ff_log(log_path):
    """Parses the FF log file to identify optimizer points and pairs to connect with a green line if FF steps occurred between them."""
    log_entries = []
    opt_pattern = re.compile(r"Step (\d+) \| loss:([\d.e+-]+)")
    ff_pattern = re.compile(r"Step (\d+) \| ff_loss:([\d.e+-]+) ff_step:(\d+)")

    # --- Pass 1: Read all relevant entries ---
    try:
        with open(log_path, "r") as f:
            for line in f:
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
                    data = None  # Loss value from FF steps not used here

                if entry_type:
                    log_entries.append(
                        {"log_step": log_step, "type": entry_type, "loss": data}
                    )
    except FileNotFoundError:
        print(f"Error: FF log file not found at {log_path}")
        return [], []
    except Exception as e:
        print(f"Error reading FF log file: {e}")
        return [], []

    if not log_entries:
        return [], []

    log_entries.sort(key=lambda x: x["log_step"])  # Sort by original log step number

    # --- Pass 2: Calculate cumulative steps and identify optimizer points ---
    optimizer_points_cumulative = []  # List of tuples: (cumulative_step, loss)
    optimizer_log_info = []  # List of tuples: (log_step, cumulative_step, loss)
    cumulative_step = 0
    for entry in log_entries:
        cumulative_step += 1
        entry["cumulative_step"] = (
            cumulative_step  # Add cumulative step to original entry
        )
        if entry["type"] == "opt":
            loss = entry["loss"]
            optimizer_points_cumulative.append((cumulative_step, loss))
            optimizer_log_info.append((entry["log_step"], cumulative_step, loss))

    # --- Pass 3: Identify pairs of optimizer steps with FF steps between them ---
    ff_connecting_pairs = []  # List of ((cs1, l1), (cs2, l2))
    if len(optimizer_log_info) >= 2:
        for i in range(len(optimizer_log_info) - 1):
            opt1_log_step, opt1_cs, opt1_loss = optimizer_log_info[i]
            opt2_log_step, opt2_cs, opt2_loss = optimizer_log_info[i + 1]

            # Check if any FF entry exists strictly between the log steps of these two optimizer steps
            found_ff_between = False
            for entry in log_entries:
                # Check log_step range and type
                if (
                    opt1_log_step < entry["log_step"] < opt2_log_step
                    and entry["type"] == "ff"
                ):
                    found_ff_between = True
                    break  # Found one, no need to check further for this pair

            if found_ff_between:
                ff_connecting_pairs.append(((opt1_cs, opt1_loss), (opt2_cs, opt2_loss)))

    return optimizer_points_cumulative, ff_connecting_pairs


def _setup_plot_style(ax, title=""):
    """Applies common styling to the plot axes."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.linewidth"] = 1.5
    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["scatter.marker"] = "."
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams["xtick.labelsize"] = 16
    plt.rcParams["ytick.labelsize"] = 16
    plt.rcParams["legend.fontsize"] = 16
    plt.rcParams["axes.titlesize"] = 18

    ax.set_xlabel("Step")
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


def plot_regular_only(regular_data, output_path):
    """Generates and saves the plot for regular training only."""
    if not regular_data:
        print("No regular training data to plot.")
        return

    steps = [p[0] for p in regular_data]
    losses = [p[1] for p in regular_data]

    fig, ax = plt.subplots(figsize=(8, 5))
    _setup_plot_style(ax, title="Regular Training Loss")

    ax.plot(steps, losses, color="blue", label="Regular training")

    ax.set_xlim(left=0, right=max(steps) * 1.05 if steps else 1)
    if losses:
        ax.set_ylim(bottom=min(losses) * 0.95, top=max(losses) * 1.05)

    ax.legend(frameon=True, edgecolor="lightgrey", loc="upper right")
    plt.tight_layout()
    _save_plot(fig, output_path)


def plot_ff_only(optimizer_points_cumulative, ff_connecting_pairs, output_path):
    """Generates and saves the plot for FF training only (red points, green connections)."""
    if not optimizer_points_cumulative:
        print("No FF optimizer points to plot.")
        return

    opt_steps = [p[0] for p in optimizer_points_cumulative]
    opt_losses = [p[1] for p in optimizer_points_cumulative]

    fig, ax = plt.subplots(figsize=(8, 5))
    _setup_plot_style(ax, title="FF Training Loss")

    # Plot optimizer steps as red points
    ax.scatter(
        opt_steps,
        opt_losses,
        color="red",
        marker="o",
        s=25,
        label="_nolegend_",
        zorder=5,
    )

    # Plot green lines connecting specific optimizer points
    for point1, point2 in ff_connecting_pairs:
        ax.plot(
            [point1[0], point2[0]],
            [point1[1], point2[1]],
            color="green",
            linestyle="-",
            label="_nolegend_",
        )

    # Determine axis limits based on optimizer points
    if opt_steps:
        ax.set_xlim(left=0, right=max(opt_steps) * 1.05)
    if opt_losses:
        ax.set_ylim(bottom=min(opt_losses) * 0.95, top=max(opt_losses) * 1.05)

    # Create legend items manually
    red_marker = mlines.Line2D(
        [],
        [],
        color="red",
        marker="o",
        linestyle="None",
        markersize=6,
        label="optimizer steps",
    )
    green_line = mlines.Line2D(
        [], [], color="green", linewidth=2, label="FF connection"
    )  # Updated label
    handles = [red_marker, green_line]
    ax.legend(handles=handles, frameon=True, edgecolor="lightgrey", loc="upper right")

    plt.tight_layout()
    _save_plot(fig, output_path)


def plot_combined(
    regular_data, optimizer_points_cumulative, ff_connecting_pairs, output_path
):
    """Generates and saves the combined plot (red points, green connections for FF)."""
    if not regular_data and not optimizer_points_cumulative:
        print("No data found for combined plot.")
        return

    # FF Data points
    opt_steps = [p[0] for p in optimizer_points_cumulative]
    opt_losses = [p[1] for p in optimizer_points_cumulative]

    # Regular Data points
    regular_steps = [p[0] for p in regular_data]
    regular_losses = [p[1] for p in regular_data]

    fig, ax = plt.subplots(figsize=(8, 5))
    _setup_plot_style(ax, title="Regular vs FF Training Loss")

    # Plot Regular Training Data
    if regular_steps:
        ax.plot(regular_steps, regular_losses, color="blue", label="_nolegend_")

    # Plot optimizer steps as red points
    if opt_steps:
        ax.scatter(
            opt_steps,
            opt_losses,
            color="red",
            marker="o",
            s=25,
            label="_nolegend_",
            zorder=5,
        )

    # Plot green lines connecting specific optimizer points
    for point1, point2 in ff_connecting_pairs:
        ax.plot(
            [point1[0], point2[0]],
            [point1[1], point2[1]],
            color="green",
            linestyle="-",
            label="_nolegend_",
        )

    # Determine combined axis limits
    all_steps = regular_steps + opt_steps
    all_losses = regular_losses + opt_losses

    if all_steps:
        ax.set_xlim(left=0, right=max(all_steps) * 1.05)
    if all_losses:
        ax.set_ylim(bottom=min(all_losses) * 0.95, top=max(all_losses) * 1.05)

    # Legend
    red_marker = mlines.Line2D(
        [],
        [],
        color="red",
        marker="o",
        linestyle="None",
        markersize=6,
        label="optimizer steps",
    )
    green_line = mlines.Line2D(
        [], [], color="green", linewidth=2, label="FF connection"
    )  # Updated label
    blue_line = mlines.Line2D(
        [], [], color="blue", linewidth=2, label="Regular training"
    )

    handles = [red_marker, green_line, blue_line]
    ax.legend(handles=handles, frameon=True, edgecolor="lightgrey", loc="upper right")

    plt.tight_layout()
    _save_plot(fig, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot and compare regular vs fast-forward training loss from log files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--regular-log",
        required=True,
        help="Path to the log file for regular training.",
    )
    parser.add_argument(
        "--ff-log",
        required=True,
        help="Path to the log file for fast-forward training.",
    )
    parser.add_argument(
        "-o",
        "--output-prefix",
        default="loss_plot",
        help="Prefix for the output plot image files (e.g., 'my_run' will generate 'my_run_regular.png', 'my_run_ff.png', 'my_run_combined.png').",
    )
    parser.add_argument(
        "-n",
        "--max-steps",
        type=int,
        default=None,
        help="Only plot data up to this maximum cumulative step number.",
    )

    args = parser.parse_args()

    # Parse data
    regular_data_full = parse_regular_log(args.regular_log)
    optimizer_points_full, ff_connecting_pairs_full = parse_ff_log(args.ff_log)

    # Filter data based on max_steps if provided
    max_steps = args.max_steps
    regular_data_filtered = regular_data_full
    optimizer_points_filtered = optimizer_points_full
    ff_connecting_pairs_filtered = ff_connecting_pairs_full

    if max_steps is not None:
        print(f"Filtering data to show only up to cumulative step {max_steps}.")
        # Filter regular data based on its own step count
        regular_data_filtered = [p for p in regular_data_full if p[0] <= max_steps]
        # Filter FF optimizer points based on cumulative step
        optimizer_points_filtered = [
            p for p in optimizer_points_full if p[0] <= max_steps
        ]
        # Filter FF connecting pairs: only keep pairs where *both* points are within max_steps
        ff_connecting_pairs_filtered = [
            pair
            for pair in ff_connecting_pairs_full
            if pair[0][0] <= max_steps and pair[1][0] <= max_steps
        ]

    if not regular_data_filtered and not optimizer_points_filtered:
        print("No data left to plot after filtering.")
        exit()

    # Define output paths
    output_path_regular = f"{args.output_prefix}_regular.png"
    output_path_ff = f"{args.output_prefix}_ff.png"
    output_path_combined = f"{args.output_prefix}_combined.png"

    # Generate plots
    plot_regular_only(regular_data_filtered, output_path_regular)
    plot_ff_only(
        optimizer_points_filtered, ff_connecting_pairs_filtered, output_path_ff
    )
    plot_combined(
        regular_data_filtered,
        optimizer_points_filtered,
        ff_connecting_pairs_filtered,
        output_path_combined,
    )
