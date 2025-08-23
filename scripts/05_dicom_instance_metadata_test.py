#!/usr/bin/env python3
"""
Test version of comprehensive DICOM instance-level metadata extraction script.

This script tests the functionality on only 3 series to verify everything works correctly.
"""

import os
import pandas as pd
import pydicom
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, Optional, Tuple
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# DICOM tags to extract
DICOM_TAGS = [
    'BitsAllocated', 'BitsStored', 'Columns', 'FrameOfReferenceUID', 'HighBit',
    'ImageOrientationPatient', 'ImagePositionPatient', 'InstanceNumber', 'Modality',
    'PatientID', 'PhotometricInterpretation', 'PixelRepresentation', 'PixelSpacing',
    'PlanarConfiguration', 'RescaleIntercept', 'RescaleSlope', 'RescaleType', 'Rows',
    'SOPClassUID', 'SOPInstanceUID', 'SamplesPerPixel', 'SliceThickness',
    'SpacingBetweenSlices', 'StudyInstanceUID', 'TransferSyntaxUID'
]

def extract_dicom_metadata(dicom_path: str) -> Dict[str, Any]:
    """
    Extract pixel data shape and specified DICOM tags from a single DICOM file.
    
    Args:
        dicom_path: Path to the DICOM file
        
    Returns:
        Dictionary containing extracted metadata and any errors
    """
    result = {
        'file_path': dicom_path,
        'pixel_shape': None,
        'dimension': None,
        'shape_str': None,
        'error': None
    }
    
    # Initialize all DICOM tags as None
    for tag in DICOM_TAGS:
        result[tag] = None
    
    try:
        # Read DICOM file
        ds = pydicom.dcmread(dicom_path, force=True)
        
        # Extract pixel data information
        try:
            if hasattr(ds, 'pixel_array'):
                pixel_array = ds.pixel_array
                shape = pixel_array.shape
                result['pixel_shape'] = shape
                result['dimension'] = len(shape)
                result['shape_str'] = str(shape)
            else:
                result['error'] = 'No pixel data available'
                logger.warning(f"No pixel data in {dicom_path}")
        except Exception as pixel_error:
            result['error'] = f'Pixel data error: {str(pixel_error)}'
            logger.warning(f"Error reading pixel data from {dicom_path}: {pixel_error}")
        
        # Extract DICOM tags
        for tag in DICOM_TAGS:
            try:
                if hasattr(ds, tag):
                    value = getattr(ds, tag)
                    # Convert certain types to strings for CSV compatibility
                    if isinstance(value, (list, tuple, np.ndarray)):
                        result[tag] = str(value)
                    elif isinstance(value, bytes):
                        result[tag] = value.decode('utf-8', errors='ignore')
                    else:
                        result[tag] = value
                else:
                    result[tag] = None
            except Exception as tag_error:
                logger.debug(f"Error extracting tag {tag} from {dicom_path}: {tag_error}")
                result[tag] = None
                
    except Exception as e:
        error_msg = f'DICOM read error: {str(e)}'
        result['error'] = error_msg
        logger.error(f"Error reading DICOM file {dicom_path}: {e}")
        
    return result

def extract_metadata_from_sample_series(series_base_path: str, max_series: int = 3) -> pd.DataFrame:
    """
    Extract metadata from first few series for testing.
    
    Args:
        series_base_path: Base path containing series folders
        max_series: Maximum number of series to process
        
    Returns:
        DataFrame with instance-level metadata
    """
    results = []
    
    # Get first few series folders
    series_folders = [d for d in os.listdir(series_base_path) 
                     if os.path.isdir(os.path.join(series_base_path, d))][:max_series]
    
    logger.info(f"Testing with {len(series_folders)} series")
    
    total_instances = 0
    
    for series_idx, series_uid in enumerate(series_folders):
        series_path = os.path.join(series_base_path, series_uid)
        
        # Get all DICOM files in the series folder
        dicom_files = [f for f in os.listdir(series_path) if f.endswith('.dcm')]
        
        if not dicom_files:
            logger.warning(f"No DICOM files found in {series_path}")
            continue
        
        logger.info(f"Processing series {series_idx+1}/{len(series_folders)}: {series_uid} "
                   f"({len(dicom_files)} instances)")
        
        total_instances += len(dicom_files)
        
        for dicom_file in dicom_files:
            dicom_path = os.path.join(series_path, dicom_file)
            
            try:
                # Extract metadata
                metadata = extract_dicom_metadata(dicom_path)
                metadata['SeriesInstanceUID'] = series_uid
                metadata['instance_filename'] = dicom_file
                
                results.append(metadata)
                
            except Exception as e:
                logger.error(f"Unexpected error processing {dicom_path}: {e}")
                logger.error(traceback.format_exc())
                
                # Still add a record with error information
                error_record = {
                    'file_path': dicom_path,
                    'SeriesInstanceUID': series_uid,
                    'instance_filename': dicom_file,
                    'error': f'Processing error: {str(e)}'
                }
                # Initialize all other fields as None
                for tag in DICOM_TAGS:
                    error_record[tag] = None
                error_record.update({
                    'pixel_shape': None,
                    'dimension': None,
                    'shape_str': None
                })
                
                results.append(error_record)
    
    logger.info(f"Processed {total_instances} total instances from {len(series_folders)} series")
    
    return pd.DataFrame(results)

def analyze_test_results(df: pd.DataFrame):
    """Generate summary statistics of the test extraction."""
    logger.info("\n" + "="*50)
    logger.info("DICOM METADATA EXTRACTION TEST RESULTS")
    logger.info("="*50)
    
    total_instances = len(df)
    logger.info(f"Total instances processed: {total_instances}")
    
    # Error analysis
    errors = df['error'].notna().sum()
    success_rate = (total_instances - errors) / total_instances * 100
    logger.info(f"Successful extractions: {total_instances - errors} ({success_rate:.1f}%)")
    logger.info(f"Failed extractions: {errors} ({errors/total_instances*100:.1f}%)")
    
    if errors > 0:
        logger.info("\nError types:")
        error_types = df[df['error'].notna()]['error'].value_counts()
        for error, count in error_types.items():
            logger.info(f"  {error}: {count}")
    
    # Shape analysis
    logger.info(f"\nShape information:")
    valid_shapes = df['pixel_shape'].notna().sum()
    logger.info(f"Instances with valid pixel shapes: {valid_shapes}")
    
    if valid_shapes > 0:
        # Show some example shapes
        logger.info(f"Example shapes:")
        for i, (idx, row) in enumerate(df[df['pixel_shape'].notna()].head(5).iterrows()):
            logger.info(f"  {row['instance_filename']}: {row['shape_str']}")
        
        # Dimension distribution
        logger.info(f"\nDimension distribution:")
        dim_counts = df['dimension'].value_counts(dropna=False).sort_index()
        for dim, count in dim_counts.items():
            logger.info(f"  {dim}D: {count}")
    
    # Series-level statistics
    series_count = df['SeriesInstanceUID'].nunique()
    logger.info(f"\nSeries statistics:")
    logger.info(f"Total unique series: {series_count}")
    logger.info(f"Average instances per series: {total_instances/series_count:.1f}")
    
    # Tag availability (show all for test)
    logger.info(f"\nDICOM tag availability:")
    for tag in DICOM_TAGS:
        available = df[tag].notna().sum()
        logger.info(f"  {tag}: {available}/{total_instances} ({available/total_instances*100:.1f}%)")
    
    # Show sample data
    logger.info(f"\nSample data (first few columns):")
    sample_cols = ['SeriesInstanceUID', 'instance_filename', 'shape_str', 'dimension', 'Modality', 'SOPInstanceUID']
    available_cols = [col for col in sample_cols if col in df.columns]
    print(df[available_cols].head())
    
    logger.info("="*50)

def main():
    """Main function to test DICOM metadata extraction."""
    
    # Define paths
    base_path = Path('/home/hongrui/work/kaggle_rsna_2025')
    series_path = base_path / 'data' / 'series'
    output_file = base_path / 'data' / 'processed' / 'dicom_instance_metadata_test.csv'
    
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if paths exist
    if not series_path.exists():
        logger.error(f"Series path does not exist: {series_path}")
        return
    
    logger.info("Starting DICOM metadata extraction test (3 series)...")
    logger.info(f"Input path: {series_path}")
    logger.info(f"Output file: {output_file}")
    
    # Extract metadata from sample series
    logger.info("Extracting metadata from sample DICOM instances...")
    df = extract_metadata_from_sample_series(str(series_path), max_series=3)
    
    # Save test results
    logger.info(f"Saving test results to {output_file}")
    df.to_csv(output_file, index=False)
    
    # Analyze results
    analyze_test_results(df)
    
    logger.info("DICOM metadata extraction test completed successfully!")

if __name__ == "__main__":
    main()