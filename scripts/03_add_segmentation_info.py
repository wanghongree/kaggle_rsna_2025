#!/usr/bin/env python3
"""
Add segmentation information to the training dataset.

This script scans the segmentations directory, identifies which SeriesInstanceUIDs
have segmentation data (.nii and _cowseg.nii files), and merges this information 
with the existing train_with_dcm_counts.csv dataset.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import os


def scan_segmentation_directory(segmentations_dir: Path) -> pd.DataFrame:
    """
    Scan the segmentations directory and extract segmentation file information.
    
    Args:
        segmentations_dir: Path to the segmentations directory
        
    Returns:
        DataFrame with SeriesInstanceUID and segmentation file information
    """
    segmentation_data = []
    
    if not segmentations_dir.exists():
        print(f"Segmentations directory not found: {segmentations_dir}")
        return pd.DataFrame()
    
    print(f"Scanning segmentation directory: {segmentations_dir}")
    
    # Iterate through each series subdirectory
    for series_path in segmentations_dir.iterdir():
        if series_path.is_dir():
            series_id = series_path.name
            
            # Look for .nii and _cowseg.nii files
            nii_files = list(series_path.glob("*.nii"))
            
            # Separate main .nii file from _cowseg.nii file
            main_nii_file = None
            cowseg_nii_file = None
            
            for nii_file in nii_files:
                if nii_file.name.endswith("_cowseg.nii"):
                    cowseg_nii_file = nii_file.name
                else:
                    main_nii_file = nii_file.name
            
            # Add to segmentation data
            segmentation_data.append({
                'SeriesInstanceUID': series_id,
                'has_segmentation': True,
                'main_nii_file': main_nii_file,
                'cowseg_nii_file': cowseg_nii_file,
                'segmentation_file_count': len(nii_files)
            })
    
    # Create DataFrame
    seg_df = pd.DataFrame(segmentation_data)
    
    print(f"Found {len(seg_df)} series with segmentation data")
    
    return seg_df


def load_train_data(train_data_path: Path) -> pd.DataFrame:
    """
    Load the existing train dataset with DICOM counts.
    
    Args:
        train_data_path: Path to train_with_dcm_counts.csv
        
    Returns:
        DataFrame with training data and DICOM counts
    """
    if not train_data_path.exists():
        raise FileNotFoundError(f"Train data file not found: {train_data_path}")
    
    train_df = pd.read_csv(train_data_path)
    print(f"Loaded training data: {len(train_df)} records")
    
    return train_df


def merge_segmentation_info(train_df: pd.DataFrame, seg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left join the training data with segmentation information.
    
    Args:
        train_df: Training dataset with DICOM counts
        seg_df: Segmentation information dataset
        
    Returns:
        Merged DataFrame with segmentation indicators and file names
    """
    # Perform left join to keep all training records
    merged_df = train_df.merge(seg_df, on='SeriesInstanceUID', how='left')
    
    # Fill NaN values for series without segmentation
    merged_df['has_segmentation'] = merged_df['has_segmentation'].fillna(False)
    merged_df['main_nii_file'] = merged_df['main_nii_file'].fillna('')
    merged_df['cowseg_nii_file'] = merged_df['cowseg_nii_file'].fillna('')
    merged_df['segmentation_file_count'] = merged_df['segmentation_file_count'].fillna(0).astype(int)
    
    print(f"Merged dataset: {len(merged_df)} records")
    print(f"Series with segmentation: {merged_df['has_segmentation'].sum()}")
    print(f"Series without segmentation: {(~merged_df['has_segmentation']).sum()}")
    
    return merged_df


def save_enhanced_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save the enhanced dataset with segmentation information.
    
    Args:
        df: Enhanced DataFrame with segmentation info
        output_path: Path to save the enhanced dataset
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save enhanced dataset
    df.to_csv(output_path, index=False)
    print(f"Enhanced dataset saved: {output_path}")
    print(f"Shape: {df.shape}")


def print_segmentation_summary(df: pd.DataFrame) -> None:
    """
    Print summary statistics about segmentation data.
    
    Args:
        df: Enhanced DataFrame with segmentation info
    """
    print("\n" + "="*60)
    print("SEGMENTATION DATA SUMMARY")
    print("="*60)
    
    # Overall statistics
    total_series = len(df)
    series_with_seg = df['has_segmentation'].sum()
    series_without_seg = total_series - series_with_seg
    
    print(f"Total series: {total_series}")
    print(f"Series with segmentation: {series_with_seg} ({series_with_seg/total_series*100:.1f}%)")
    print(f"Series without segmentation: {series_without_seg} ({series_without_seg/total_series*100:.1f}%)")
    
    # Segmentation by modality
    print("\nSegmentation availability by modality:")
    seg_by_modality = df.groupby('Modality')['has_segmentation'].agg(['count', 'sum'])
    seg_by_modality['percentage'] = (seg_by_modality['sum'] / seg_by_modality['count'] * 100).round(1)
    seg_by_modality.columns = ['Total', 'With_Segmentation', 'Percentage']
    print(seg_by_modality)
    
    # Aneurysm presence vs segmentation
    if 'Aneurysm Present' in df.columns:
        print("\nSegmentation availability by aneurysm presence:")
        seg_by_aneurysm = df.groupby('Aneurysm Present')['has_segmentation'].agg(['count', 'sum'])
        seg_by_aneurysm['percentage'] = (seg_by_aneurysm['sum'] / seg_by_aneurysm['count'] * 100).round(1)
        seg_by_aneurysm.columns = ['Total', 'With_Segmentation', 'Percentage']
        seg_by_aneurysm.index = ['No Aneurysm', 'Has Aneurysm']
        print(seg_by_aneurysm)


def main():
    """Main function to add segmentation information to the training dataset."""
    # Define paths
    data_dir = Path("data")
    segmentations_dir = data_dir / "segmentations"
    train_data_path = data_dir / "processed" / "train_with_dcm_counts.csv"
    processed_dir = data_dir / "processed"
    output_file = processed_dir / "train_with_dcm_and_segmentation.csv"
    
    print("Adding segmentation information to training dataset...")
    print(f"Segmentations directory: {segmentations_dir}")
    print(f"Train data file: {train_data_path}")
    print(f"Output file: {output_file}")
    
    # Check if input files exist
    if not train_data_path.exists():
        print(f"Error: Train data file not found at {train_data_path}")
        print("Please run 02_eda_dcm_by_modality.py first to generate the train data with DICOM counts.")
        return
    
    if not segmentations_dir.exists():
        print(f"Error: Segmentations directory not found at {segmentations_dir}")
        return
    
    # Scan segmentation directory
    seg_df = scan_segmentation_directory(segmentations_dir)
    
    if seg_df.empty:
        print("No segmentation data found. Exiting.")
        return
    
    # Load training data
    train_df = load_train_data(train_data_path)
    
    # Merge segmentation information
    enhanced_df = merge_segmentation_info(train_df, seg_df)
    
    # Print summary statistics
    print_segmentation_summary(enhanced_df)
    
    # Save enhanced dataset
    save_enhanced_dataset(enhanced_df, output_file)
    
    print(f"\nEnhanced dataset saved to: {output_file}")


if __name__ == "__main__":
    main()