import re
import json
import argparse
import sys


def parse_benchmark_results(content):
    """Parses benchmark results from the markdown content."""

    # Ignore the OLD section if present
    content = content.split("<!-- ---")[0]

    experiments = []
    # Split by H1 headings, skipping the first empty string if the file starts with #
    sections = re.split(r"^#\s+(.+)", content, flags=re.MULTILINE)[1:]

    if (
        not sections
    ):  # Handle case where there's no H1 heading maybe? Or just rely on format.
        print(f"Warning: No top-level sections found in content.", file=sys.stderr)
        return []

    # Process in pairs: title, body
    for i in range(0, len(sections), 2):
        title = sections[i].strip()
        body = sections[i + 1].strip()

        experiment_data = {"name": title, "languages": {}, "overall": {}}

        # Find language metrics
        lang_matches = re.finditer(
            r"Code Matching for (\w+): EM ([\d.]+), ES ([\d.]+), ES RepoEval ([\d.]+)",
            body,
        )
        for match in lang_matches:
            lang = match.group(1)
            em = float(match.group(2))
            es = float(match.group(3))
            es_repoeval = float(match.group(4))
            experiment_data["languages"][lang] = {
                "EM": em,
                "ES": es,
                "ES RepoEval": es_repoeval,
            }

        # Find overall metrics
        overall_match = re.search(
            r"Overall Results \(Weighted Average\):\s*"
            r"EM: ([\d.]+)\s*"
            r"ES: ([\d.]+)\s*"
            r"ES RepoEval: ([\d.]+)",
            body,
            re.DOTALL,  # Allow . to match newline for the potentially multi-line overall block
        )
        if overall_match:
            experiment_data["overall"] = {
                "EM": float(overall_match.group(1)),
                "ES": float(overall_match.group(2)),
                "ES RepoEval": float(overall_match.group(3)),
            }
        else:
            print(
                f"Warning: Could not find Overall Results for section: {title}",
                file=sys.stderr,
            )

        if experiment_data["languages"] or experiment_data["overall"]:
            experiments.append(experiment_data)
        else:
            print(f"Warning: No data extracted for section: {title}", file=sys.stderr)

    return experiments


def main():
    parser = argparse.ArgumentParser(
        description="Parse benchmark results from Markdown files into JSON."
    )
    parser.add_argument("files", nargs="+", help="Paths to the Markdown result files.")

    args = parser.parse_args()

    all_results = []
    for filename in args.files:
        try:
            with open(filename, "r", encoding="utf-8") as f:
                content = f.read()

            results = parse_benchmark_results(content)
            # Optionally add filename to each result if processing multiple files
            for res in results:
                res["source_file"] = filename
            all_results.extend(results)
        except FileNotFoundError:
            print(f"Error: File not found: {filename}", file=sys.stderr)
        except Exception as e:
            print(f"Error processing file {filename}: {e}", file=sys.stderr)

    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
