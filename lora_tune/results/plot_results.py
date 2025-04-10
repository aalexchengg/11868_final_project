import json
import argparse
import os
import re
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np  # For handling potential missing epochs smoothly


def extract_epoch_data(data):
    """Extracts and organizes data by epoch from the loaded JSON."""
    epoch_data = defaultdict(
        lambda: {
            "overall": {"EM": np.nan, "ES": np.nan, "ES RepoEval": np.nan},
            "languages": defaultdict(
                lambda: {"EM": np.nan, "ES": np.nan, "ES RepoEval": np.nan}
            ),
        }
    )
    model_size = None
    epochs_present = set()

    for experiment in data:
        name = experiment["name"]
        # Extract model size (only once)
        if model_size is None:
            match = re.search(r"Qwen2\.5-(\d+\.\d+B)", name)
            if match:
                model_size = match.group(1)

        # Extract epoch
        epoch = 0  # Default for base
        epoch_match = re.search(r"EPOCH\s+(\d+)", name)
        if epoch_match:
            epoch = int(epoch_match.group(1))

        epochs_present.add(epoch)

        # Store overall data
        if "overall" in experiment and experiment["overall"]:
            epoch_data[epoch]["overall"]["EM"] = experiment["overall"].get("EM", np.nan)
            epoch_data[epoch]["overall"]["ES"] = experiment["overall"].get("ES", np.nan)
            epoch_data[epoch]["overall"]["ES RepoEval"] = experiment["overall"].get(
                "ES RepoEval", np.nan
            )

        # Store language data
        if "languages" in experiment:
            for lang, metrics in experiment["languages"].items():
                epoch_data[epoch]["languages"][lang]["EM"] = metrics.get("EM", np.nan)
                epoch_data[epoch]["languages"][lang]["ES"] = metrics.get("ES", np.nan)
                epoch_data[epoch]["languages"][lang]["ES RepoEval"] = metrics.get(
                    "ES RepoEval", np.nan
                )

    # Determine the full epoch range (0 to max epoch found)
    if not epochs_present:
        return None, None, None  # No valid data found

    max_epoch = max(epochs_present)
    all_epochs_sorted = sorted(list(range(max_epoch + 1)))

    # Fill missing epochs with NaN (Matplotlib handles this well for line plots)
    for ep in all_epochs_sorted:
        if ep not in epoch_data:
            # Ensure missing epochs have entries, defaultdict handles nested structure
            _ = epoch_data[ep]

    # Sort data by epoch for plotting
    sorted_epoch_keys = sorted(epoch_data.keys())

    # Check if base epoch (0) exists, add NaN if not
    if 0 not in sorted_epoch_keys:
        print(
            "Warning: Base data (Epoch 0) not found. Plots will start from Epoch 1 or later."
        )

    # Ensure we cover the full range up to max_epoch found for plotting
    full_range_epochs = list(range(max_epoch + 1))

    return epoch_data, model_size, full_range_epochs


def plot_overall_metrics(epoch_data, model_size, epochs, output_dir):
    """Plots overall EM, ES, and ES RepoEval vs. epoch."""
    if not model_size:
        model_size = "UnknownSize"
    os.makedirs(output_dir, exist_ok=True)

    metrics_to_plot = ["EM", "ES", "ES RepoEval"]

    for metric in metrics_to_plot:
        values = [epoch_data[ep]["overall"][metric] for ep in epochs]

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, values, marker="o", linestyle="-")
        plt.title(f"{model_size} - Overall {metric} vs. Epoch")
        plt.xlabel("Training Epoch (0 = Base Model)")
        plt.ylabel(f"Overall {metric} Score")
        plt.xticks(epochs)  # Ensure all epoch numbers are shown
        plt.grid(True)
        plt.ylim(bottom=0)  # Start y-axis at 0 for clarity
        plt.tight_layout(pad=1.5)  # Add padding to prevent title/label cutoff

        filename = os.path.join(output_dir, f"{model_size}_overall_{metric}.png")
        plt.savefig(filename)
        plt.close()
        print(f"Saved plot: {filename}")


def plot_language_metrics(epoch_data, model_size, epochs, output_dir):
    """Plots per-language EM and ES vs. epoch with dual axes."""
    if not model_size:
        model_size = "UnknownSize"
    os.makedirs(output_dir, exist_ok=True)

    # Find all languages present in the data across epochs
    languages = set()
    for ep in epochs:
        languages.update(epoch_data[ep]["languages"].keys())

    if not languages:
        print("No language-specific data found to plot.")
        return

    for lang in sorted(list(languages)):
        em_values = [epoch_data[ep]["languages"][lang]["EM"] for ep in epochs]
        es_values = [epoch_data[ep]["languages"][lang]["ES"] for ep in epochs]

        fig, ax1 = plt.subplots(figsize=(12, 7))

        color_em = "tab:blue"
        ax1.set_xlabel("Training Epoch (0 = Base Model)")
        ax1.set_ylabel(f"{lang.capitalize()} Exact Match (EM) Score", color=color_em)
        ax1.plot(
            epochs,
            em_values,
            marker="o",
            linestyle="-",
            color=color_em,
            label=f"{lang} EM",
        )
        ax1.tick_params(axis="y", labelcolor=color_em)
        ax1.set_xticks(epochs)  # Ensure all epoch numbers are shown
        ax1.grid(True, axis="x")  # Grid only for x-axis to avoid clutter
        ax1.set_ylim(bottom=0)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color_es = "tab:red"
        ax2.set_ylabel(
            f"{lang.capitalize()} Edit Similarity (ES) Score", color=color_es
        )  # we already handled the x-label with ax1
        ax2.plot(
            epochs,
            es_values,
            marker="s",
            linestyle="--",
            color=color_es,
            label=f"{lang} ES",
        )
        ax2.tick_params(axis="y", labelcolor=color_es)
        ax2.set_ylim(bottom=0)

        plt.title(f"{model_size} - {lang.capitalize()} EM & ES vs. Epoch")
        fig.tight_layout()  # adjust layout *after* setting title
        # Adding legends - might need adjustment for position
        # fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

        filename = os.path.join(output_dir, f"{model_size}_{lang}_EM_ES.png")
        plt.savefig(filename)
        plt.close(fig)
        print(f"Saved plot: {filename}")


def plot_consolidated_em_all_languages(epoch_data, model_size, epochs, output_dir):
    """Plots EM scores for all languages on a single graph vs. epoch."""
    if not model_size:
        model_size = "UnknownSize"
    os.makedirs(output_dir, exist_ok=True)

    languages = set()
    for ep in epochs:
        languages.update(epoch_data[ep]["languages"].keys())

    if not languages:
        print("No language-specific data found for consolidated EM plot.")
        return

    plt.figure(figsize=(12, 7))

    # Define markers and linestyles to differentiate languages
    markers = ["o", "s", "^", "d", "v", "<", ">", "p", "*", "h", "H", "+", "x", "D"]
    linestyles = ["-", "--", "-.", ":"]

    for i, lang in enumerate(sorted(list(languages))):
        em_values = [epoch_data[ep]["languages"][lang]["EM"] for ep in epochs]
        plt.plot(
            epochs,
            em_values,
            marker=markers[i % len(markers)],
            linestyle=linestyles[i % len(linestyles)],
            label=f"{lang.capitalize()} EM",
        )

    plt.title(f"{model_size} - Consolidated EM Scores vs. Epoch")
    plt.xlabel("Training Epoch (0 = Base Model)")
    plt.ylabel("Exact Match (EM) Score")
    plt.xticks(epochs)
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.legend()
    plt.tight_layout(pad=1.5)

    filename = os.path.join(output_dir, f"{model_size}_consolidated_EM.png")
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot: {filename}")


def plot_consolidated_es_all_languages(epoch_data, model_size, epochs, output_dir):
    """Plots ES scores for all languages on a single graph vs. epoch."""
    if not model_size:
        model_size = "UnknownSize"
    os.makedirs(output_dir, exist_ok=True)

    languages = set()
    for ep in epochs:
        languages.update(epoch_data[ep]["languages"].keys())

    if not languages:
        print("No language-specific data found for consolidated ES plot.")
        return

    plt.figure(figsize=(12, 7))

    markers = ["o", "s", "^", "d", "v", "<", ">", "p", "*", "h", "H", "+", "x", "D"]
    linestyles = ["-", "--", "-.", ":"]

    for i, lang in enumerate(sorted(list(languages))):
        es_values = [epoch_data[ep]["languages"][lang]["ES"] for ep in epochs]
        plt.plot(
            epochs,
            es_values,
            marker=markers[i % len(markers)],
            linestyle=linestyles[i % len(linestyles)],
            label=f"{lang.capitalize()} ES",
        )

    plt.title(f"{model_size} - Consolidated ES Scores vs. Epoch")
    plt.xlabel("Training Epoch (0 = Base Model)")
    plt.ylabel("Edit Similarity (ES) Score")
    plt.xticks(epochs)
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.legend()
    plt.tight_layout(pad=1.5)

    filename = os.path.join(output_dir, f"{model_size}_consolidated_ES.png")
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot: {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot benchmark results from JSON file."
    )
    parser.add_argument("json_file", help="Path to the JSON results file.")
    parser.add_argument(
        "-o", "--output-dir", default="plots", help="Directory to save the plots."
    )

    args = parser.parse_args()

    try:
        with open(args.json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found: {args.json_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from file: {args.json_file}")
        return
    except Exception as e:
        print(f"Error reading file {args.json_file}: {e}")
        return

    epoch_data, model_size, epochs = extract_epoch_data(data)

    if epoch_data is None:
        print("Could not extract valid data for plotting.")
        return

    # Generate output subdirectory based on model size
    model_specific_output_dir = os.path.join(
        args.output_dir, model_size if model_size else "UnknownSize"
    )

    plot_overall_metrics(epoch_data, model_size, epochs, model_specific_output_dir)
    plot_language_metrics(epoch_data, model_size, epochs, model_specific_output_dir)
    plot_consolidated_em_all_languages(
        epoch_data, model_size, epochs, model_specific_output_dir
    )
    plot_consolidated_es_all_languages(
        epoch_data, model_size, epochs, model_specific_output_dir
    )

    print("\nPlot generation complete.")


if __name__ == "__main__":
    main()
