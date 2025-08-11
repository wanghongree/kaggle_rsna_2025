#!/usr/bin/env python3
"""
Count DICOM instances per series in the RSNA Intracranial Aneurysm dataset.

This script scans the data/series directory and counts the number of DICOM files
(.dcm) for each series. The results are saved to a CSV file in data/processed/.
"""

import os
from pathlib import Path
import pandas as pd
from typing import Dict


def count_dcm_files_per_series(series_dir: Path) -> Dict[str, int]:
    """
    Count the number of DICOM files for each series.
    
    Args:
        series_dir: Path to the series directory
        
    Returns:
        Dictionary mapping series IDs to DICOM file counts
    """
    series_counts = {}
    
    if not series_dir.exists():
        print(f"Series directory not found: {series_dir}")
        return series_counts
    
    # Iterate through each series subdirectory
    for series_path in series_dir.iterdir():
        if series_path.is_dir():
            series_id = series_path.name
            # Count .dcm files in this series directory
            dcm_files = list(series_path.glob("*.dcm"))
            series_counts[series_id] = len(dcm_files)
    
    return series_counts


def save_results_to_csv(series_counts: Dict[str, int], output_path: Path) -> None:
    """
    Save the series counts to a CSV file.
    
    Args:
        series_counts: Dictionary mapping series IDs to counts
        output_path: Path where to save the CSV file
    """
    # Create DataFrame
    df = pd.DataFrame([
        {'SeriesInstanceUID': series_id, 'dcm_file_count': count}
        for series_id, count in series_counts.items()
    ])
    
    # Sort by series ID for consistent output
    df = df.sort_values('SeriesInstanceUID').reset_index(drop=True)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")
    print(f"Total series: {len(df)}")
    print(f"DICOM file count statistics:")
    print(df['dcm_file_count'].describe())


def main():
    """Main function to count DICOM files and save results."""
    # Define paths
    data_dir = Path("data")
    series_dir = data_dir / "series"
    processed_dir = data_dir / "processed"
    output_file = processed_dir / "dcm_file_counts_per_series.csv"
    
    print("Counting DICOM files per series...")
    print(f"Series directory: {series_dir}")
    
    # Count DICOM files per series
    series_counts = count_dcm_files_per_series(series_dir)
    
    if not series_counts:
        print("No series found or series directory is empty.")
        return
    
    # Save results
    save_results_to_csv(series_counts, output_file)


if __name__ == "__main__":
    main()