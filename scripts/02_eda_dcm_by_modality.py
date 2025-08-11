#!/usr/bin/env python3
"""
Generate histograms of DICOM file counts by imaging modality.

This script reads train.csv and merges it with the DICOM file counts per series
to analyze the distribution of file counts across different imaging modalities.
Generates histograms and saves them to the reports/ folder.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from typing import Optional


def load_and_merge_data(train_csv_path: Path, dcm_counts_path: Path) -> pd.DataFrame:
    """
    Load train.csv and merge with DICOM file counts.
    
    Args:
        train_csv_path: Path to train.csv
        dcm_counts_path: Path to dcm file counts CSV
        
    Returns:
        Merged DataFrame with modality and DICOM counts
    """
    # Load train data
    train_df = pd.read_csv(train_csv_path)
    print(f"Loaded train.csv: {len(train_df)} records")
    
    # Load DICOM counts
    dcm_counts_df = pd.read_csv(dcm_counts_path)
    print(f"Loaded DICOM counts: {len(dcm_counts_df)} records")
    
    # Merge on SeriesInstanceUID
    merged_df = train_df.merge(
        dcm_counts_df, 
        on='SeriesInstanceUID', 
        how='inner'
    )
    print(f"Merged dataset: {len(merged_df)} records")
    
    # Check for missing values
    missing_counts = len(train_df) - len(merged_df)
    if missing_counts > 0:
        print(f"Warning: {missing_counts} series from train.csv not found in DICOM counts")
    
    return merged_df


def create_modality_histograms(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Create histograms of DICOM file counts for each modality.
    
    Args:
        df: Merged DataFrame with modality and dcm_file_count columns
        output_dir: Directory to save the histogram images
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get unique modalities
    modalities = df['Modality'].unique()
    print(f"Found modalities: {modalities}")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create individual histograms for each modality
    for modality in modalities:
        modality_data = df[df['Modality'] == modality]['dcm_file_count']
        
        plt.figure(figsize=(10, 6))
        plt.hist(modality_data, bins=30, alpha=0.7, edgecolor='black')
        plt.title(f'Distribution of DICOM File Counts - {modality}', fontsize=14, fontweight='bold')
        plt.xlabel('Number of DICOM Files per Series', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f'Count: {len(modality_data)}\nMean: {modality_data.mean():.1f}\nMedian: {modality_data.median():.1f}\nStd: {modality_data.std():.1f}'
        plt.text(0.75, 0.75, stats_text, transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=10)
        
        # Save individual histogram
        output_file = output_dir / f'dcm_count_histogram_{modality}.png'
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved histogram for {modality}: {output_file}")
    
    # Create combined histogram with subplots
    n_modalities = len(modalities)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, modality in enumerate(modalities):
        if i < len(axes):
            modality_data = df[df['Modality'] == modality]['dcm_file_count']
            
            axes[i].hist(modality_data, bins=20, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{modality} (n={len(modality_data)})', fontweight='bold')
            axes[i].set_xlabel('DICOM Files per Series')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(modalities), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('DICOM File Count Distribution by Modality', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save combined histogram
    combined_output = output_dir / 'dcm_count_histograms_combined.png'
    plt.savefig(combined_output, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined histogram: {combined_output}")


def create_summary_boxplot(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Create a boxplot comparing DICOM file counts across modalities.
    
    Args:
        df: Merged DataFrame with modality and dcm_file_count columns
        output_dir: Directory to save the boxplot image
    """
    plt.figure(figsize=(10, 6))
    
    # Create boxplot
    sns.boxplot(data=df, x='Modality', y='dcm_file_count')
    plt.title('DICOM File Count Distribution by Modality', fontsize=14, fontweight='bold')
    plt.xlabel('Imaging Modality', fontsize=12)
    plt.ylabel('Number of DICOM Files per Series', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Save boxplot
    output_file = output_dir / 'dcm_count_boxplot_by_modality.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved boxplot: {output_file}")


def save_merged_data(df: pd.DataFrame, processed_dir: Path) -> None:
    """
    Save the merged DataFrame to data/processed/ directory.
    
    Args:
        df: Merged DataFrame with train data and DICOM counts
        processed_dir: Directory to save the merged data
    """
    # Ensure processed directory exists
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Define output path
    output_path = processed_dir / "train_with_dcm_counts.csv"
    
    # Save merged data
    df.to_csv(output_path, index=False)
    print(f"Saved merged dataset: {output_path}")
    print(f"Shape: {df.shape}")


def print_summary_statistics(df: pd.DataFrame) -> None:
    """
    Print summary statistics for DICOM file counts by modality.
    
    Args:
        df: Merged DataFrame with modality and dcm_file_count columns
    """
    print("\n" + "="*50)
    print("SUMMARY STATISTICS BY MODALITY")
    print("="*50)
    
    summary = df.groupby('Modality')['dcm_file_count'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(2)
    
    print(summary)
    
    print("\nOverall Statistics:")
    print(f"Total series: {len(df)}")
    print(f"Total DICOM files: {df['dcm_file_count'].sum()}")
    print(f"Average files per series: {df['dcm_file_count'].mean():.2f}")


def main():
    """Main function to create histograms of DICOM counts by modality."""
    # Define paths
    data_dir = Path("data")
    train_csv_path = data_dir / "train.csv"
    dcm_counts_path = data_dir / "processed" / "dcm_file_counts_per_series.csv"
    processed_dir = data_dir / "processed"
    reports_dir = Path("reports")
    
    print("Creating DICOM count histograms by modality...")
    print(f"Train CSV: {train_csv_path}")
    print(f"DICOM counts CSV: {dcm_counts_path}")
    print(f"Output directory: {reports_dir}")
    
    # Check if input files exist
    if not train_csv_path.exists():
        print(f"Error: Train CSV not found at {train_csv_path}")
        return
    
    if not dcm_counts_path.exists():
        print(f"Error: DICOM counts CSV not found at {dcm_counts_path}")
        print("Please run 01_eda_dcm_number.py first to generate the DICOM counts.")
        return
    
    # Load and merge data
    merged_df = load_and_merge_data(train_csv_path, dcm_counts_path)
    
    # Save merged data to processed directory
    save_merged_data(merged_df, processed_dir)
    
    # Print summary statistics
    print_summary_statistics(merged_df)
    
    # Create histograms
    create_modality_histograms(merged_df, reports_dir)
    
    # Create boxplot
    create_summary_boxplot(merged_df, reports_dir)
    
    print(f"\nAll visualizations saved to: {reports_dir}")
    print(f"Merged dataset saved to: {processed_dir / 'train_with_dcm_counts.csv'}")


if __name__ == "__main__":
    main()