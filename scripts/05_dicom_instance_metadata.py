#!/usr/bin/env python3
"""
Comprehensive DICOM instance-level metadata extraction script.

This script:
1. Reads every DICOM instance in the data/series folder
2. Extracts pixel data shapes and specified DICOM tags  
3. Saves data incrementally to handle large datasets and resume on errors
4. Creates an instance-level dataframe with one row per DICOM instance

Usage:
    python scripts/05_dicom_instance_metadata.py
"""

import os
import pandas as pd
import pydicom
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, Optional, Tuple
import traceback
import json

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
    'SpacingBetweenSlices', 'StudyInstanceUID', 'TransferSyntaxUID', 'WindowCenter', 'WindowWidth'
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

def save_checkpoint(data: list, checkpoint_file: str):
    """Save current progress to checkpoint file."""
    try:
        df = pd.DataFrame(data)
        df.to_csv(checkpoint_file, index=False)
        logger.info(f"Checkpoint saved with {len(data)} records")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")

def load_checkpoint(checkpoint_file: str) -> Tuple[list, set]:
    """Load progress from checkpoint file."""
    if os.path.exists(checkpoint_file):
        try:
            df = pd.read_csv(checkpoint_file)
            data = df.to_dict('records')
            processed_files = set(df['file_path'].tolist())
            logger.info(f"Loaded checkpoint with {len(data)} records")
            return data, processed_files
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return [], set()
    return [], set()

def extract_metadata_from_all_instances(series_base_path: str, output_file: str, 
                                      checkpoint_interval: int = 1000) -> pd.DataFrame:
    """
    Extract metadata from all DICOM instances across all series.
    
    Args:
        series_base_path: Base path containing series folders
        output_file: Path to save final output
        checkpoint_interval: Number of instances to process before saving checkpoint
        
    Returns:
        DataFrame with instance-level metadata
    """
    checkpoint_file = output_file.replace('.csv', '_checkpoint.csv')
    
    # Load existing progress
    results, processed_files = load_checkpoint(checkpoint_file)
    
    # Get all series folders
    series_folders = [d for d in os.listdir(series_base_path) 
                     if os.path.isdir(os.path.join(series_base_path, d))]
    
    logger.info(f"Found {len(series_folders)} series folders")
    logger.info(f"Already processed {len(processed_files)} files")
    
    total_instances = 0
    processed_instances = len(processed_files)
    
    # Count total instances for progress tracking
    for series_uid in series_folders:
        series_path = os.path.join(series_base_path, series_uid)
        dicom_files = [f for f in os.listdir(series_path) if f.endswith('.dcm')]
        total_instances += len(dicom_files)
    
    logger.info(f"Total instances to process: {total_instances}")
    logger.info(f"Remaining instances: {total_instances - processed_instances}")
    
    instance_count = processed_instances
    
    for series_idx, series_uid in enumerate(series_folders):
        series_path = os.path.join(series_base_path, series_uid)
        
        # Get all DICOM files in the series folder
        dicom_files = [f for f in os.listdir(series_path) if f.endswith('.dcm')]
        
        if not dicom_files:
            logger.warning(f"No DICOM files found in {series_path}")
            continue
        
        logger.info(f"Processing series {series_idx+1}/{len(series_folders)}: {series_uid} "
                   f"({len(dicom_files)} instances)")
        
        for dicom_file in dicom_files:
            dicom_path = os.path.join(series_path, dicom_file)
            
            # Skip if already processed
            if dicom_path in processed_files:
                continue
            
            try:
                # Extract metadata
                metadata = extract_dicom_metadata(dicom_path)
                metadata['SeriesInstanceUID'] = series_uid
                metadata['instance_filename'] = dicom_file
                
                results.append(metadata)
                processed_files.add(dicom_path)
                instance_count += 1
                
                # Progress logging
                if instance_count % 100 == 0:
                    logger.info(f"Processed {instance_count}/{total_instances} instances "
                               f"({instance_count/total_instances*100:.1f}%)")
                
                # Save checkpoint periodically
                if len(results) % checkpoint_interval == 0:
                    save_checkpoint(results, checkpoint_file)
                    
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
                instance_count += 1
    
    # Final save
    logger.info("Creating final DataFrame...")
    df = pd.DataFrame(results)
    
    # Save final output
    logger.info(f"Saving final results to {output_file}")
    df.to_csv(output_file, index=False)
    
    # Remove checkpoint file after successful completion
    try:
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            logger.info("Checkpoint file removed after successful completion")
    except Exception as e:
        logger.warning(f"Could not remove checkpoint file: {e}")
    
    return df

def analyze_results(df: pd.DataFrame):
    """Generate summary statistics of the extracted metadata."""
    logger.info("\n" + "="*50)
    logger.info("DICOM METADATA EXTRACTION SUMMARY")
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
        error_types = df[df['error'].notna()]['error'].value_counts().head(10)
        for error, count in error_types.items():
            logger.info(f"  {error}: {count}")
    
    # Shape analysis
    logger.info(f"\nShape information:")
    valid_shapes = df['pixel_shape'].notna().sum()
    logger.info(f"Instances with valid pixel shapes: {valid_shapes}")
    
    if valid_shapes > 0:
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
    
    # Tag availability
    logger.info(f"\nDICOM tag availability:")
    for tag in DICOM_TAGS[:10]:  # Show first 10 tags
        available = df[tag].notna().sum()
        logger.info(f"  {tag}: {available}/{total_instances} ({available/total_instances*100:.1f}%)")
    
    logger.info("="*50)

def main():
    """Main function to extract comprehensive DICOM metadata."""
    
    # Define paths
    base_path = Path('/home/hongrui/work/kaggle_rsna_2025')
    series_path = base_path / 'data' / 'series'
    output_file = base_path / 'data' / 'processed' / 'dicom_instance_metadata.csv'
    
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if paths exist
    if not series_path.exists():
        logger.error(f"Series path does not exist: {series_path}")
        return
    
    logger.info("Starting comprehensive DICOM metadata extraction...")
    logger.info(f"Input path: {series_path}")
    logger.info(f"Output file: {output_file}")
    
    # Extract metadata from all instances
    logger.info("Extracting metadata from all DICOM instances...")
    df = extract_metadata_from_all_instances(str(series_path), str(output_file))
    
    # Analyze results
    analyze_results(df)
    
    logger.info("DICOM metadata extraction completed successfully!")

if __name__ == "__main__":
    main()