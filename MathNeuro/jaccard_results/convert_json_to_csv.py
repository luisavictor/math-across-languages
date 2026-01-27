# a script which iteratives through all json files in a directory and/or subdirectories
# and converts them to csv files
import json
import csv
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from collections import defaultdict

def convert_json_to_csv(json_dir: str | Path) -> None:
    """
    Converts all JSON files in the specified directory to CSV files.
    Creates separate CSV files for each good_percent value with rows representing layers.
    Aggregates across different seeds and calculates mean and std.

    Args:
        json_dir: Directory containing JSON files.
    """
    json_dir = Path(json_dir)
    
    # Group JSON files by directory
    directories = {}
    for json_file in json_dir.rglob("*.json"):
        if "summary" not in json_file.name:
            continue
        parent_dir = json_file.parent
        if parent_dir not in directories:
            directories[parent_dir] = []
        directories[parent_dir].append(json_file)
    
    # Process each directory
    for directory, json_files in directories.items():
        print(f"\nProcessing directory: {directory}")
        print(f"Found {len(json_files)} JSON file(s)")
        
        # Collect all data grouped by good_percent
        data_by_good_percent = defaultdict(lambda: defaultdict(list))
        
        for json_file in json_files:
            print(f"  Loading: {json_file.name}")
            
            with json_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            
            if not data:
                continue
            
            # Group data by good_percent
            for entry in data:
                good_percent = entry.get("good_percent")
                if good_percent is None:
                    continue
                
                # Extract per-layer data
                per_group = entry.get("per_group", {})
                if not per_group:
                    continue
                
                # Store data for each layer
                for layer_name, layer_data in per_group.items():
                    data_by_good_percent[good_percent][layer_name].append(layer_data)
        
        # Create CSV files for each good_percent
        for good_percent, layers_data in sorted(data_by_good_percent.items()):
            rows = []
            
            for layer_name, layer_entries in sorted(layers_data.items()):
                # Extract layer number from layer name (e.g., "layers.0" -> 0)
                layer_num = layer_name.split(".")[-1] if "." in layer_name else layer_name
                
                # Calculate statistics across seeds
                jaccard_values = [entry.get("jaccard", 0) for entry in layer_entries]
                total_isolated_run1_values = [entry.get("total_isolated_run1", 0) for entry in layer_entries]
                total_isolated_run2_values = [entry.get("total_isolated_run2", 0) for entry in layer_entries]
                intersection_values = [entry.get("intersection", 0) for entry in layer_entries]
                union_values = [entry.get("union", 0) for entry in layer_entries]
                
                row = {
                    "layer": layer_num,
                    "layer_name": layer_name,
                    "total_isolated_run1_mean": np.mean(total_isolated_run1_values),
                    "total_isolated_run1_std": np.std(total_isolated_run1_values) if len(total_isolated_run1_values) > 1 else 0,
                    "total_isolated_run2_mean": np.mean(total_isolated_run2_values),
                    "total_isolated_run2_std": np.std(total_isolated_run2_values) if len(total_isolated_run2_values) > 1 else 0,
                    "intersection_mean": np.mean(intersection_values),
                    "intersection_std": np.std(intersection_values) if len(intersection_values) > 1 else 0,
                    "union_mean": np.mean(union_values),
                    "union_std": np.std(union_values) if len(union_values) > 1 else 0,
                    "jaccard_mean": np.mean(jaccard_values),
                    "jaccard_std": np.std(jaccard_values) if len(jaccard_values) > 1 else 0,
                    "num_tensors": layer_entries[0].get("num_tensors", 0),
                    "num_elements_compared": layer_entries[0].get("num_elements_compared", 0),
                    "num_seeds": len(layer_entries),
                }
                rows.append(row)
            
            # Create CSV filename based on good_percent
            csv_filename = directory / f"aggregated_good_percent_{good_percent}.csv"
            
            # Write to CSV
            if rows:
                df = pd.DataFrame(rows)
                # sort by layer number
                df["layer"] = df["layer"].astype(int)
                df = df.sort_values(by="layer")
                df.to_csv(csv_filename, index=False)
                print(f"  Created: {csv_filename}")

if __name__ == "__main__":
    import argparse

    json_dir = "/raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/jaccard_results"

    convert_json_to_csv(json_dir)