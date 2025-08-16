#!/usr/bin/env python3
"""
Sample version of DICOM shape extraction script for testing.

This script processes only a subset of series for quick validation.
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
    """Determine if the pixel data is 2D, 3D, or 4D based on shape."""
    if len(shape) == 2:
        return '2d'
    elif len(shape) == 3:
        return '3d'
    elif len(shape) == 4:
        return '4d'
    else:
        return f'{len(shape)}d'

def extract_dicom_shape(dicom_path: str) -> Tuple[Optional[Tuple[int, ...]], Optional[str]]:
    """Extract pixel data shape from a DICOM file."""
    try:
        ds = pydicom.dcmread(dicom_path, force=True)
        
        if not hasattr(ds, 'pixel_array'):
            logger.warning(f"No pixel data found in {dicom_path}")
            return None, None
            
        pixel_array = ds.pixel_array
        shape = pixel_array.shape
        dimension = determine_dimension(shape)
        
        logger.debug(f"File: {dicom_path}, Shape: {shape}, Dimension: {dimension}")
        return shape, dimension
        
    except Exception as e:
        logger.error(f"Error reading {dicom_path}: {str(e)}")
        return None, None

def main():
    """Main function to extract DICOM shapes for a sample."""
    
    # Define paths
    base_path = Path('/home/hongrui/work/kaggle_rsna_2025')
    series_path = base_path / 'data' / 'series'
    train_file = base_path / 'data' / 'processed' / 'train_with_dcm_and_segmentation.csv'
    output_file = base_path / 'data' / 'processed' / 'train_with_dicom_shapes_sample.csv'
    
    # Get first 10 series for testing
    series_folders = [d for d in os.listdir(series_path) 
                     if os.path.isdir(os.path.join(series_path, d))][:10]
    
    logger.info(f"Processing {len(series_folders)} series for testing...")
    
    results = []
    for i, series_uid in enumerate(series_folders):
        logger.info(f"Processing series {i+1}/{len(series_folders)}: {series_uid}")
        
        series_dir = series_path / series_uid
        dicom_files = [f for f in os.listdir(series_dir) if f.endswith('.dcm')]
        
        if not dicom_files:
            logger.warning(f"No DICOM files found in {series_dir}")
            continue
            
        # Sample the first DICOM file
        sample_dicom = dicom_files[0]
        dicom_path = series_dir / sample_dicom
        
        # Extract shape information
        shape, dimension = extract_dicom_shape(str(dicom_path))
        
        results.append({
            'SeriesInstanceUID': series_uid,
            'pixel_shape': shape,
            'dimension': dimension,
            'sample_file': sample_dicom,
            'total_dicom_files': len(dicom_files)
        })
        
        logger.info(f"  Shape: {shape}, Dimension: {dimension}, Files: {len(dicom_files)}")
    
    # Create DataFrame
    dicom_shapes_df = pd.DataFrame(results)
    
    # Load the training data
    train_df = pd.read_csv(train_file)
    
    # Left join with training data
    result_df = train_df.merge(
        dicom_shapes_df, 
        on='SeriesInstanceUID', 
        how='left'
    )
    
    # Save the result
    result_df.to_csv(output_file, index=False)
    
    logger.info(f"Sample extraction completed. Results saved to {output_file}")
    logger.info(f"Sample results:")
    print(dicom_shapes_df.to_string(index=False))

if __name__ == "__main__":
    main()