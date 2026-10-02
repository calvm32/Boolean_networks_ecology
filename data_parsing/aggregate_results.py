import csv
import json
from pathlib import Path

# Path to data folder CHANGE AS NEEDED
DATA_DIR = Path(__file__).parent.parent / "results_and_data/all_sites_PSO_parameter_data/data"
OUTPUT_CSV = "results_and_data/run_all_sites_PSO/all_sites_parameters.csv"

def aggregate_json_results():
    records = []

    # Find all *_results.json files inside any subfolder under data/
    json_files = sorted(list(DATA_DIR.rglob("*_results.json")))

    if not json_files:
        print(f"No *_results.json files found in {DATA_DIR.resolve()}")
        return

    for file_path in json_files:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            # Extract top-level fields
            row = {
                "site_name": data.get("site_name"),
                "best_loss": data.get("best_loss"),
                "wall_clock_seconds": data.get("wall_clock_seconds"),
            }

            # Unpack / flatten the best_params dictionary into the row
            best_params = data.get("best_params", {})
            row.update(best_params)

            records.append(row)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # Extract all unique column headers across all JSON files
    fieldnames = list(records[0].keys()) if records else []

    # Write to CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(
        f"Successfully aggregated {len(records)} sites into '{OUTPUT_CSV}'."
    )


if __name__ == "__main__":
    aggregate_json_results()