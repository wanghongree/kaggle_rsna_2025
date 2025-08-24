# %%
"""
Scout Detection Test - Process small subset to validate algorithm
"""

import pandas as pd
import numpy as np
import ast
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Tuple, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# %%
def parse_array_string(array_str: Any) -> Optional[np.ndarray]:
    """Parse array string (IOP or IPP) to numpy array"""
    try:
        if pd.isna(array_str) or array_str == '' or array_str == 'nan':
            return None
        # Handle string representation of list
        if isinstance(array_str, str):
            array_list = ast.literal_eval(array_str)
        else:
            return None
        return np.array(array_list, dtype=float)
    except (ValueError, SyntaxError, TypeError) as e:
        logger.debug(f"Failed to parse array: {array_str}, error: {e}")
        return None

# %%
def calculate_slice_normal_vector(iop: np.ndarray) -> Optional[np.ndarray]:
    """Calculate slice normal vector from ImageOrientationPatient"""
    if iop is None or len(iop) != 6:
        return None
    
    # IOP contains row direction cosines (first 3) and column direction cosines (last 3)
    row_cosines = iop[:3]
    col_cosines = iop[3:]
    
    # Normal vector is cross product of row and column direction cosines
    normal = np.cross(row_cosines, col_cosines)
    
    # Normalize
    norm = np.linalg.norm(normal)
    if norm == 0:
        return None
    
    return normal / norm

# %%
def calculate_slice_position(ipp: np.ndarray, normal: np.ndarray) -> float:
    """Calculate slice position by projecting IPP onto normal vector"""
    return np.dot(ipp, normal)

# %%
def detect_scout_images_in_series(series_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect scout images in a single series based on distance outliers
    
    Returns dictionary with scout detection results
    """
    series_uid = series_data['SeriesInstanceUID'].iloc[0]
    modality = series_data['Modality'].iloc[0]
    
    valid_slices = []
    
    # Parse IOP/IPP for each slice
    for _, row in series_data.iterrows():
        iop = parse_array_string(row['ImageOrientationPatient'])
        ipp = parse_array_string(row['ImagePositionPatient'])
        
        if iop is not None and ipp is not None:
            normal = calculate_slice_normal_vector(iop)
            if normal is not None:
                position = calculate_slice_position(ipp, normal)
                valid_slices.append({
                    'SOPInstanceUID': row['SOPInstanceUID'],
                    'InstanceNumber': row.get('InstanceNumber', 0),
                    'ipp': ipp,
                    'normal': normal,
                    'position': position
                })
    
    result = {
        'SeriesInstanceUID': series_uid,
        'Modality': modality,
        'total_instances': len(series_data),
        'valid_instances': len(valid_slices),
        'scout_instances': [],
        'non_scout_instances': [],
        'has_scouts': False,
        'min_distance': None,
        'max_distance': None,
        'mean_distance': None,
        'total_distance': None
    }
    
    if len(valid_slices) < 3:  # Need at least 3 slices for scout detection
        return result
    
    # Sort slices by position
    valid_slices.sort(key=lambda x: x['position'])
    
    # Calculate distances between consecutive slices
    distances = []
    for i in range(1, len(valid_slices)):
        dist = abs(valid_slices[i]['position'] - valid_slices[i-1]['position'])
        distances.append(dist)
    
    if len(distances) == 0:
        return result
    
    distances = np.array(distances)
    
    # Scout detection algorithm
    median_dist = np.median(distances)
    q75_dist = np.percentile(distances, 75)
    q25_dist = np.percentile(distances, 25)
    iqr = q75_dist - q25_dist
    
    # Define outlier threshold (slices with distances much larger than typical)
    outlier_threshold = max(
        median_dist * 5,  # 5x median distance
        q75_dist + 3 * iqr  # 3 IQRs above 75th percentile
    )
    
    # Identify scout positions
    scout_indices = set()
    for i, dist in enumerate(distances):
        if dist > outlier_threshold:
            # Mark both slices involved in the large gap
            scout_indices.add(i)      # slice before the gap
            scout_indices.add(i + 1)  # slice after the gap
    
    # Additional check: if the first or last slice is very far from the main group
    if len(distances) > 2:
        # Check if first slice is a scout
        first_gap = distances[0]
        remaining_distances = distances[1:]
        if len(remaining_distances) > 0 and first_gap > np.median(remaining_distances) * 5:
            scout_indices.add(0)
        
        # Check if last slice is a scout  
        last_gap = distances[-1]
        remaining_distances = distances[:-1]
        if len(remaining_distances) > 0 and last_gap > np.median(remaining_distances) * 5:
            scout_indices.add(len(valid_slices) - 1)
    
    # Separate scout and non-scout instances
    scout_instances = []
    non_scout_instances = []
    
    for i, slice_info in enumerate(valid_slices):
        if i in scout_indices:
            scout_instances.append(slice_info['SOPInstanceUID'])
        else:
            non_scout_instances.append(slice_info['SOPInstanceUID'])
    
    # Calculate distances after removing scouts
    non_scout_slices = [valid_slices[i] for i in range(len(valid_slices)) if i not in scout_indices]
    
    if len(non_scout_slices) >= 2:
        non_scout_slices.sort(key=lambda x: x['position'])
        
        # Calculate total distance (first to last)
        total_distance = abs(non_scout_slices[-1]['position'] - non_scout_slices[0]['position'])
        
        # Calculate distances between consecutive non-scout slices
        clean_distances = []
        for i in range(1, len(non_scout_slices)):
            dist = abs(non_scout_slices[i]['position'] - non_scout_slices[i-1]['position'])
            clean_distances.append(dist)
        
        if len(clean_distances) > 0:
            result.update({
                'min_distance': np.min(clean_distances),
                'max_distance': np.max(clean_distances),
                'mean_distance': np.mean(clean_distances),
                'total_distance': total_distance
            })
    
    result.update({
        'scout_instances': scout_instances,
        'non_scout_instances': non_scout_instances,
        'has_scouts': len(scout_instances) > 0
    })
    
    return result

# %%
def test_scout_detection():
    """Test scout detection on a small subset of data"""
    logger.info("Loading DICOM instance metadata for testing...")
    
    data_path = Path("data/processed/dicom_instance_metadata.csv")
    
    # Load only first 10000 rows for testing
    dtype_dict = {'RescaleType': str}
    df = pd.read_csv(data_path, nrows=10000, dtype=dtype_dict)
    
    logger.info(f"Loaded {len(df)} instances for testing")
    
    results = []
    
    # Process each series
    for series_uid, series_data in df.groupby('SeriesInstanceUID'):
        result = detect_scout_images_in_series(series_data)
        results.append(result)
    
    logger.info(f"Processed {len(results)} series")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Print summary
    print("\n" + "="*50)
    print("TEST RESULTS - SCOUT DETECTION")
    print("="*50)
    
    total_series = len(results_df)
    series_with_scouts = len(results_df[results_df['has_scouts'] == True])
    
    print(f"Series analyzed: {total_series}")
    print(f"Series with scouts: {series_with_scouts}")
    print(f"Scout detection rate: {series_with_scouts/total_series*100:.1f}%")
    
    # Show examples
    print("\nEXAMPLES WITH SCOUTS:")
    scout_examples = results_df[results_df['has_scouts'] == True].head(3)
    for _, row in scout_examples.iterrows():
        print(f"  {row['Modality']} series: {len(row['scout_instances'])} scouts out of {row['valid_instances']} slices")
    
    print("\nEXAMPLES WITHOUT SCOUTS:")
    non_scout_examples = results_df[results_df['has_scouts'] == False].head(3)
    for _, row in non_scout_examples.iterrows():
        if row['total_distance'] is not None:
            print(f"  {row['Modality']} series: {row['valid_instances']} slices, total distance: {row['total_distance']:.1f}mm")
    
    # Distance analysis
    distance_data = results_df[
        (results_df['has_scouts'] == False) & 
        (results_df['total_distance'].notna())
    ]
    
    if len(distance_data) > 0:
        print(f"\nDISTANCE ANALYSIS ({len(distance_data)} non-scout series):")
        print(f"Mean total distance: {distance_data['total_distance'].mean():.1f}mm")
        print(f"Median total distance: {distance_data['total_distance'].median():.1f}mm")
        print(f"Min total distance: {distance_data['total_distance'].min():.1f}mm")
        print(f"Max total distance: {distance_data['total_distance'].max():.1f}mm")
    
    return results_df

# %%
if __name__ == "__main__":
    results_df = test_scout_detection()