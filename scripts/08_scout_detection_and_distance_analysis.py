# %%
"""
Scout Detection and Distance Analysis
Detect scout images and analyze distance distributions using DICOM instance metadata
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
def analyze_scout_detection() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main analysis function to detect scouts and calculate distances
    
    Returns:
        Tuple of (series_results_df, distance_records_df)
    """
    logger.info("Loading DICOM instance metadata...")
    
    data_path = Path("data/processed/dicom_instance_metadata.csv")
    
    # Load data in chunks to handle large file
    chunk_size = 25000  # Smaller chunk size for better performance
    results = []
    processed_series = set()
    
    logger.info("Processing DICOM metadata in chunks for scout detection...")
    chunk_count = 0
    
    # Add dtype specification to handle mixed types warning
    dtype_dict = {'RescaleType': str}  # Handle mixed types in RescaleType column
    
    for chunk in pd.read_csv(data_path, chunksize=chunk_size, dtype=dtype_dict, low_memory=False):
        chunk_count += 1
        logger.info(f"Processing chunk {chunk_count}...")
        
        # Group by series and analyze each series
        for series_uid, series_data in chunk.groupby('SeriesInstanceUID'):
            if series_uid not in processed_series:
                result = detect_scout_images_in_series(series_data)
                results.append(result)
                processed_series.add(series_uid)
                
                if len(results) % 500 == 0:
                    logger.info(f"Processed {len(results)} series...")
    
    logger.info(f"Completed scout detection analysis of {len(results)} series")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Create distance records for histogram analysis
    distance_records = []
    for _, row in results_df.iterrows():
        if not row['has_scouts'] and row['total_distance'] is not None:
            distance_records.append({
                'SeriesInstanceUID': row['SeriesInstanceUID'],
                'Modality': row['Modality'],
                'total_distance': row['total_distance'],
                'mean_slice_distance': row['mean_distance'],
                'valid_instances': row['valid_instances']
            })
    
    distance_df = pd.DataFrame(distance_records)
    
    return results_df, distance_df

# %%
def generate_summary_statistics(results_df: pd.DataFrame, distance_df: pd.DataFrame):
    """Generate and print summary statistics"""
    
    print("\n" + "="*70)
    print("SCOUT DETECTION AND DISTANCE ANALYSIS SUMMARY")
    print("="*70)
    
    total_series = len(results_df)
    series_with_scouts = len(results_df[results_df['has_scouts'] == True])
    series_without_scouts = total_series - series_with_scouts
    series_with_valid_distances = len(distance_df)
    
    print(f"Total series analyzed: {total_series:,}")
    print(f"Series with scout images detected: {series_with_scouts:,} ({series_with_scouts/total_series*100:.1f}%)")
    print(f"Series without scout images: {series_without_scouts:,} ({series_without_scouts/total_series*100:.1f}%)")
    print(f"Series with valid distance calculations: {series_with_valid_distances:,}")
    
    # Scout detection by modality
    print(f"\nSCOUT DETECTION BY MODALITY:")
    print("-" * 50)
    scout_by_modality = results_df.groupby('Modality')['has_scouts'].agg(['count', 'sum']).reset_index()
    scout_by_modality['scout_rate'] = scout_by_modality['sum'] / scout_by_modality['count'] * 100
    scout_by_modality.columns = ['Modality', 'Total_Series', 'Series_with_Scouts', 'Scout_Rate_%']
    
    for _, row in scout_by_modality.iterrows():
        print(f"{row['Modality']:>8}: {row['Series_with_Scouts']:>4}/{row['Total_Series']:>4} ({row['Scout_Rate_%']:>5.1f}%)")
    
    # Distance statistics by modality (non-scout series only)
    if len(distance_df) > 0:
        print(f"\nDISTANCE STATISTICS BY MODALITY (non-scout series):")
        print("-" * 60)
        
        for modality in sorted(distance_df['Modality'].unique()):
            modality_data = distance_df[distance_df['Modality'] == modality]
            
            if len(modality_data) > 0:
                print(f"\n{modality}:")
                print(f"  Series count: {len(modality_data):,}")
                print(f"  Total distance (1st to last slice):")
                print(f"    Mean: {modality_data['total_distance'].mean():.2f} mm")
                print(f"    Median: {modality_data['total_distance'].median():.2f} mm")
                print(f"    Std: {modality_data['total_distance'].std():.2f} mm")
                print(f"    Min: {modality_data['total_distance'].min():.2f} mm")
                print(f"    Max: {modality_data['total_distance'].max():.2f} mm")
                print(f"  Mean slice spacing: {modality_data['mean_slice_distance'].mean():.3f} mm")

# %%
def create_distance_distribution_plots(distance_df: pd.DataFrame):
    """Create histogram plots of distance distributions by modality"""
    
    if len(distance_df) == 0:
        logger.warning("No distance data available for plotting")
        return
    
    logger.info("Creating distance distribution plots...")
    
    # Create reports directory if it doesn't exist
    reports_dir = Path("reports/figs")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Get modalities with sufficient data
    modality_counts = distance_df['Modality'].value_counts()
    modalities_to_plot = modality_counts[modality_counts >= 10].index.tolist()
    
    if len(modalities_to_plot) == 0:
        logger.warning("No modalities with sufficient data for plotting")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Distance Distributions After Scout Image Removal', fontsize=16, y=0.98)
    
    # Plot 1: Total distance histogram
    ax1 = axes[0, 0]
    for i, modality in enumerate(modalities_to_plot[:6]):  # Limit to 6 modalities
        modality_data = distance_df[distance_df['Modality'] == modality]['total_distance']
        ax1.hist(modality_data, bins=30, alpha=0.6, label=modality, density=True)
    
    ax1.set_xlabel('Total Distance (1st to last slice, mm)')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution of Total Distances')
    ax1.legend()
    ax1.set_xlim(0, distance_df['total_distance'].quantile(0.95))
    
    # Plot 2: Box plot of total distances
    ax2 = axes[0, 1]
    modalities_subset = modalities_to_plot[:8]  # Limit for readability
    box_data = [distance_df[distance_df['Modality'] == mod]['total_distance'].values 
                for mod in modalities_subset]
    
    bp = ax2.boxplot(box_data, labels=modalities_subset, patch_artist=True)
    ax2.set_ylabel('Total Distance (mm)')
    ax2.set_title('Box Plot of Total Distances')
    ax2.tick_params(axis='x', rotation=45)
    
    # Color the boxes
    colors = sns.color_palette("husl", len(modalities_subset))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    # Plot 3: Mean slice distance histogram
    ax3 = axes[1, 0]
    for i, modality in enumerate(modalities_to_plot[:6]):
        modality_data = distance_df[distance_df['Modality'] == modality]['mean_slice_distance']
        ax3.hist(modality_data, bins=30, alpha=0.6, label=modality, density=True)
    
    ax3.set_xlabel('Mean Slice Distance (mm)')
    ax3.set_ylabel('Density')
    ax3.set_title('Distribution of Mean Slice Distances')
    ax3.legend()
    ax3.set_xlim(0, distance_df['mean_slice_distance'].quantile(0.95))
    
    # Plot 4: Summary statistics bar plot
    ax4 = axes[1, 1]
    stats_data = []
    for modality in modalities_subset:
        modality_distances = distance_df[distance_df['Modality'] == modality]['total_distance']
        stats_data.append([
            modality_distances.mean(),
            modality_distances.median(),
            modality_distances.std()
        ])
    
    stats_df = pd.DataFrame(stats_data, 
                           columns=['Mean', 'Median', 'Std Dev'],
                           index=modalities_subset)
    
    x_pos = np.arange(len(stats_df))
    width = 0.25
    
    ax4.bar(x_pos - width, stats_df['Mean'], width, label='Mean', alpha=0.8)
    ax4.bar(x_pos, stats_df['Median'], width, label='Median', alpha=0.8)
    ax4.bar(x_pos + width, stats_df['Std Dev'], width, label='Std Dev', alpha=0.8)
    
    ax4.set_xlabel('Modality')
    ax4.set_ylabel('Distance (mm)')
    ax4.set_title('Summary Statistics by Modality')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(stats_df.index, rotation=45)
    ax4.legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_path = reports_dir / "distance_distributions_after_scout_removal.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Distribution plots saved to {plot_path}")
    plt.show()

# %%
if __name__ == "__main__":
    # Run the complete analysis
    results_df, distance_df = analyze_scout_detection()
    
    # Generate summary statistics
    generate_summary_statistics(results_df, distance_df)
    
    # Save results
    output_dir = Path("data/processed")
    
    # Save scout detection results
    scout_results_path = output_dir / "scout_detection_results.csv"
    results_df.to_csv(scout_results_path, index=False)
    logger.info(f"Scout detection results saved to {scout_results_path}")
    
    # Save distance analysis results
    distance_results_path = output_dir / "distance_analysis_after_scout_removal.csv"
    distance_df.to_csv(distance_results_path, index=False)
    logger.info(f"Distance analysis results saved to {distance_results_path}")
    
    # Create visualization plots
    if len(distance_df) > 0:
        create_distance_distribution_plots(distance_df)
    else:
        logger.warning("No distance data available for plotting")