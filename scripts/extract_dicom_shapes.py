#!/usr/bin/env python3
"""
Script to extract DICOM image shapes and dimensions for each series.

This script:
1. Samples one DICOM file per series folder
2. Reads pixel data and extracts shape information
3. Determines if the data is 2D, 3D, or 4D
4. Left joins with train_with_dcm_and_segmentation.csv
5. Saves output to data/processed/

Usage:
    python scripts/extract_dicom_shapes.py
"""

import os
import pandas as pd
import pydicom
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def determine_dimension(shape: Tuple[int, ...]) -> str:
    """
    Determine if the pixel data is 2D, 3D, or 4D based on shape.
    
    Args:
        shape: Tuple representing the shape of pixel data
        
    Returns:
        String indicating dimension: '2d', '3d', or '4d'
    """
    if len(shape) == 2:
        return '2d'
    elif len(shape) == 3:
        return '3d'
    elif len(shape) == 4:
        return '4d'
    else:
        return f'{len(shape)}d'

def extract_dicom_shape(dicom_path: str) -> Tuple[Optional[Tuple[int, ...]], Optional[str]]:
    """
    Extract pixel data shape from a DICOM file.
    
    Args:
        dicom_path: Path to the DICOM file
        
    Returns:
        Tuple of (shape, dimension) or (None, None) if extraction fails
    """
    try:
        ds = pydicom.dcmread(dicom_path, force=True)
        
        # Check if pixel data exists
        if not hasattr(ds, 'pixel_array'):
            logger.warning(f"No pixel data found in {dicom_path}")
            return None, None
            
        # Get pixel array shape
        pixel_array = ds.pixel_array
        shape = pixel_array.shape
        dimension = determine_dimension(shape)
        
        logger.debug(f"File: {dicom_path}, Shape: {shape}, Dimension: {dimension}")
        return shape, dimension
        
    except Exception as e:
        logger.error(f"Error reading {dicom_path}: {str(e)}")
        return None, None

def get_sample_dicom_per_series(series_base_path: str) -> pd.DataFrame:
    """
    Sample one DICOM file per series and extract shape information.
    
    Args:
        series_base_path: Base path containing series folders
        
    Returns:
        DataFrame with SeriesInstanceUID, shape, dimension, and num_instances columns
    """
    results = []
    series_folders = [d for d in os.listdir(series_base_path) 
                     if os.path.isdir(os.path.join(series_base_path, d))]
    
    logger.info(f"Found {len(series_folders)} series folders")
    
    for i, series_uid in enumerate(series_folders):
        if i % 100 == 0:
            logger.info(f"Processing series {i+1}/{len(series_folders)}")
            
        series_path = os.path.join(series_base_path, series_uid)
        
        # Get all DICOM files in the series folder
        dicom_files = [f for f in os.listdir(series_path) 
                      if f.endswith('.dcm')]
        
        if not dicom_files:
            logger.warning(f"No DICOM files found in {series_path}")
            continue
            
        # Count the number of instances (DICOM files) in this series
        num_instances = len(dicom_files)
        
        # Sample the first DICOM file
        sample_dicom = dicom_files[0]
        dicom_path = os.path.join(series_path, sample_dicom)
        
        # Extract shape information
        shape, dimension = extract_dicom_shape(dicom_path)
        
        if shape is not None:
            results.append({
                'SeriesInstanceUID': series_uid,
                'pixel_shape': shape,
                'dimension': dimension,
                'num_instances': num_instances
            })
        else:
            # Still add the series with None values for completeness
            results.append({
                'SeriesInstanceUID': series_uid,
                'pixel_shape': None,
                'dimension': None,
                'num_instances': num_instances
            })
    
    return pd.DataFrame(results)

def main():
    """Main function to extract DICOM shapes and create output dataset."""
    
    # Define paths
    base_path = Path('/home/hongrui/work/kaggle_rsna_2025')
    series_path = base_path / 'data' / 'series'
    train_file = base_path / 'data' / 'processed' / 'train_with_dcm_and_segmentation.csv'
    output_file = base_path / 'data' / 'processed' / 'train_with_dicom_shapes.csv'
    
    # Check if paths exist
    if not series_path.exists():
        logger.error(f"Series path does not exist: {series_path}")
        return
        
    if not train_file.exists():
        logger.error(f"Train file does not exist: {train_file}")
        return
    
    logger.info("Starting DICOM shape extraction...")
    
    # Extract DICOM shapes
    logger.info("Extracting pixel data shapes from DICOM files...")
    dicom_shapes_df = get_sample_dicom_per_series(str(series_path))
    
    logger.info(f"Extracted shapes for {len(dicom_shapes_df)} series")
    
    # Load the training data
    logger.info("Loading training data...")
    train_df = pd.read_csv(train_file)
    
    logger.info(f"Training data contains {len(train_df)} series")
    
    # Left join with training data
    logger.info("Performing left join...")
    result_df = train_df.merge(
        dicom_shapes_df, 
        on='SeriesInstanceUID', 
        how='left'
    )
    
    logger.info(f"Final dataset contains {len(result_df)} rows")
    
    # Display some statistics
    logger.info("\nShape extraction statistics:")
    logger.info(f"- Total series in training data: {len(train_df)}")
    logger.info(f"- Series with extracted shapes: {len(dicom_shapes_df[dicom_shapes_df['pixel_shape'].notna()])}")
    logger.info(f"- Series with missing shapes: {len(dicom_shapes_df[dicom_shapes_df['pixel_shape'].isna()])}")
    
    # Show dimension distribution
    if 'dimension' in result_df.columns:
        dim_counts = result_df['dimension'].value_counts(dropna=False)
        logger.info(f"\nDimension distribution:")
        for dim, count in dim_counts.items():
            logger.info(f"- {dim}: {count}")
    
    # Save the result
    logger.info(f"Saving results to {output_file}")
    result_df.to_csv(output_file, index=False)
    
    logger.info("DICOM shape extraction completed successfully!")

if __name__ == "__main__":
    main()