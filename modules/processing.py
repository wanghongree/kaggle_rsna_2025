"""
Processing utilities for RSNA Intracranial Aneurysm Detection project.

This module provides functions for extracting metadata from DICOM files and processing NIfTI segmentation files.
"""

import pandas as pd
import pydicom
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


def extract_dicom_tags(dicom_path: Path) -> Optional[Dict[str, Any]]:
    """Extract specified DICOM tags from a single DICOM file.
    
    Args:
        dicom_path: Path to the DICOM file
        
    Returns:
        Dictionary containing extracted DICOM metadata, or None if extraction failed
    """
    tags_to_extract = [
        'BitsAllocated', 'BitsStored', 'Columns', 'FrameOfReferenceUID', 
        'HighBit', 'ImageOrientationPatient', 'ImagePositionPatient', 
        'InstanceNumber', 'Modality', 'PatientID', 'PhotometricInterpretation', 
        'PixelRepresentation', 'PixelSpacing', 'PlanarConfiguration', 
        'RescaleIntercept', 'RescaleSlope', 'RescaleType', 'Rows', 
        'SOPClassUID', 'SOPInstanceUID', 'SamplesPerPixel', 'SliceThickness', 
        'SpacingBetweenSlices', 'StudyInstanceUID', 'TransferSyntaxUID'
    ]
    
    try:
        ds = pydicom.dcmread(dicom_path, stop_before_pixels=False)
        
        # Extract DICOM tags
        dicom_data = {}
        for tag in tags_to_extract:
            try:
                value = getattr(ds, tag, None)
                # Convert numpy arrays to strings for storage
                if isinstance(value, np.ndarray):
                    value = str(value.tolist())
                dicom_data[tag] = value
            except Exception as e:
                dicom_data[tag] = None
                logger.warning(f"Could not extract tag {tag} from {dicom_path}: {e}")
        
        # Extract pixel data shape (without loading full pixel data)
        try:
            pixel_shape = None
            if hasattr(ds, 'pixel_array'):
                # Get shape without loading full array into memory
                rows = getattr(ds, 'Rows', None)
                cols = getattr(ds, 'Columns', None) 
                if rows and cols:
                    pixel_shape = f"({rows}, {cols})"
        except Exception as e:
            pixel_shape = None
            logger.warning(f"Could not extract pixel shape from {dicom_path}: {e}")
            
        dicom_data['pixel_shape'] = pixel_shape
        dicom_data['dicom_path'] = str(dicom_path)
        
        return dicom_data
    except Exception as e:
        error_msg = f"Error processing DICOM {dicom_path}: {e}"
        logger.error(error_msg)
        return None


def process_single_series_segmentation(series_uid: str) -> pd.DataFrame:
    """Process segmentation files for a single series.
    
    Args:
        series_uid: SeriesInstanceUID to process
        
    Returns:
        DataFrame containing segmentation metadata for the series
    """
    from .config import get_config
    
    config = get_config()
    seg_dir = config.get_segmentations_dir()
    
    if not seg_dir.exists():
        logger.debug(f"Segmentation directory not found: {seg_dir}")
        return pd.DataFrame()
    
    series_path = seg_dir / series_uid
    
    if not series_path.exists() or not series_path.is_dir():
        logger.debug(f"No segmentation directory found for series: {series_uid}")
        return pd.DataFrame()
    
    logger.info(f"Processing segmentation for series: {series_uid}")
    
    # Process main segmentation file
    nii_file = series_path / f"{series_uid}.nii"
    cowseg_file = series_path / f"{series_uid}_cowseg.nii"
    
    seg_info = {'SeriesInstanceUID': series_uid}
    
    # Process .nii file
    if nii_file.exists():
        try:
            nii_img = nib.load(str(nii_file))
            seg_info['nii_path'] = str(nii_file)
            seg_info['nii_shape'] = str(nii_img.shape)
            seg_info['nii_affine'] = str(nii_img.affine.tolist())
            seg_info['nii_header_pixdim'] = str(nii_img.header['pixdim'].tolist())
            logger.debug(f"Successfully processed {nii_file}")
        except Exception as e:
            error_msg = f"Error processing {nii_file}: {e}"
            logger.error(error_msg)
            seg_info['nii_path'] = str(nii_file)
            seg_info['nii_shape'] = None
            seg_info['nii_affine'] = None
            seg_info['nii_header_pixdim'] = None
    
    # Process _cowseg.nii file
    if cowseg_file.exists():
        try:
            cowseg_img = nib.load(str(cowseg_file))
            seg_info['cowseg_path'] = str(cowseg_file)
            seg_info['cowseg_shape'] = str(cowseg_img.shape)
            seg_info['cowseg_affine'] = str(cowseg_img.affine.tolist())
            seg_info['cowseg_header_pixdim'] = str(cowseg_img.header['pixdim'].tolist())
            logger.debug(f"Successfully processed {cowseg_file}")
        except Exception as e:
            error_msg = f"Error processing {cowseg_file}: {e}"
            logger.error(error_msg)
            seg_info['cowseg_path'] = str(cowseg_file)
            seg_info['cowseg_shape'] = None
            seg_info['cowseg_affine'] = None
            seg_info['cowseg_header_pixdim'] = None
    
    # Only return data if we found at least one segmentation file
    if nii_file.exists() or cowseg_file.exists():
        return pd.DataFrame([seg_info])
    else:
        logger.debug(f"No segmentation files found for series: {series_uid}")
        return pd.DataFrame()


def process_single_series(series_uid: str) -> pd.DataFrame:
    """Process a single series and return complete data with all joins.
    
    Args:
        series_uid: SeriesInstanceUID to process
        
    Returns:
        DataFrame containing all data for the series (DICOM + CSV + segmentation)
    """
    from .config import get_config
    
    config = get_config()
    series_dir = config.get_series_dir()
    series_path = series_dir / series_uid
    
    if not series_path.exists():
        logger.warning(f"Series directory not found: {series_path}")
        return pd.DataFrame()
    
    logger.info(f"Processing series: {series_uid}")
    
    # Process DICOM files for this series
    dicom_files = list(series_path.glob("*.dcm"))
    if not dicom_files:
        logger.warning(f"No DICOM files found in {series_path}")
        return pd.DataFrame()
    
    dicom_data = []
    for dicom_file in dicom_files:
        dicom_info = extract_dicom_tags(dicom_file)
        if dicom_info:
            dicom_info['SeriesInstanceUID'] = series_uid
            dicom_data.append(dicom_info)
    
    if not dicom_data:
        logger.warning(f"No DICOM data extracted for series: {series_uid}")
        return pd.DataFrame()
    
    dicom_df = pd.DataFrame(dicom_data)
    logger.info(f"Extracted {len(dicom_df)} DICOM images from series {series_uid}")
    
    # Join with CSV files directly inline
    try:
        train_csv_path = config.get_train_csv()
        train_localizers_csv_path = config.get_train_localizers_csv()
        
        # Load CSV files
        train_df = pd.read_csv(train_csv_path)
        train_localizers_df = pd.read_csv(train_localizers_csv_path)
        
        # Join with train.csv, using suffixes to avoid column name conflicts
        df_combined = dicom_df.merge(train_df, on='SeriesInstanceUID', how='left', suffixes=('_dicom', '_train'))
        
        # Left join with train_localizers.csv
        df_with_csv = df_combined.merge(train_localizers_df, on=['SeriesInstanceUID', 'SOPInstanceUID'], how='left')
        
    except Exception as e:
        logger.error(f"Error in CSV joins: {e}")
        df_with_csv = dicom_df
    
    # Process segmentation for this series
    seg_df = process_single_series_segmentation(series_uid)
    
    # Join with segmentation data
    if len(seg_df) > 0:
        df_final = df_with_csv.merge(seg_df, on='SeriesInstanceUID', how='left')
    else:
        df_final = df_with_csv
    
    logger.info(f"Completed processing series {series_uid}: {len(df_final)} rows")
    return df_final


def get_already_processed_series(output_file: Path) -> set:
    """Get set of SeriesInstanceUIDs that have already been processed.
    
    Args:
        output_file: Path to the output parquet file (will check CSV fallback)
        
    Returns:
        Set of already processed SeriesInstanceUIDs
    """
    # Try parquet first
    if output_file.exists():
        try:
            df = pd.read_parquet(output_file)
            processed_series = set(df['SeriesInstanceUID'].unique())
            logger.info(f"Found {len(processed_series)} already processed series (parquet)")
            return processed_series
        except Exception as e:
            logger.warning(f"Could not read parquet file {output_file}: {e}")
    
    # Try CSV fallback
    csv_file = output_file.with_suffix('.csv')
    if csv_file.exists():
        try:
            df = pd.read_csv(csv_file)
            processed_series = set(df['SeriesInstanceUID'].unique())
            logger.info(f"Found {len(processed_series)} already processed series (CSV)")
            return processed_series
        except Exception as e:
            logger.warning(f"Could not read CSV file {csv_file}: {e}")
    
    logger.info("No existing output file found")
    return set()


def append_series_data(series_data: pd.DataFrame, output_file: Path) -> None:
    """Append series data to output file.
    
    Args:
        series_data: DataFrame containing data for one series
        output_file: Path to the output parquet file (will fallback to CSV if needed)
    """
    if len(series_data) == 0:
        return
    
    try:
        if output_file.exists():
            # Read existing data and append
            try:
                existing_df = pd.read_parquet(output_file)
            except Exception:
                # Fallback to CSV if parquet fails
                csv_file = output_file.with_suffix('.csv')
                if csv_file.exists():
                    existing_df = pd.read_csv(csv_file)
                    output_file = csv_file
                else:
                    raise
            combined_df = pd.concat([existing_df, series_data], ignore_index=True)
        else:
            # Create new file
            output_file.parent.mkdir(parents=True, exist_ok=True)
            combined_df = series_data
        
        # Save to parquet (with CSV fallback)
        try:
            combined_df.to_parquet(output_file, index=False)
            logger.info(f"Saved {len(series_data)} rows to {output_file} (parquet)")
        except Exception as e:
            # Fallback to CSV
            csv_file = output_file.with_suffix('.csv')
            combined_df.to_csv(csv_file, index=False)
            logger.info(f"Saved {len(series_data)} rows to {csv_file} (CSV fallback)")
        
    except Exception as e:
        logger.error(f"Error saving data to {output_file}: {e}")
        raise