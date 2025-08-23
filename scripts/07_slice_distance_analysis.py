# %%
"""
Slice Distance Analysis
Calculate distances between DICOM slices using IOP and IPP data
"""

import pandas as pd
import numpy as np
import ast
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# %%
def parse_array_string(array_str):
    """Parse array string (IOP or IPP) to numpy array"""
    try:
        if pd.isna(array_str) or array_str == '' or array_str == 'nan':
            return None
        # Handle string representation of list
        array_list = ast.literal_eval(array_str)
        return np.array(array_list, dtype=float)
    except (ValueError, SyntaxError, TypeError) as e:
        logger.debug(f"Failed to parse array: {array_str}, error: {e}")
        return None

# %%
def calculate_slice_normal_vector(iop):
    """Calculate slice normal vector from IOP"""
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
def calculate_slice_distances(series_data):
    """Calculate distances between slices in a series"""
    valid_slices = []
    
    for _, row in series_data.iterrows():
        iop = parse_array_string(row['ImageOrientationPatient'])
        ipp = parse_array_string(row['ImagePositionPatient'])
        
        if iop is not None and ipp is not None:
            normal = calculate_slice_normal_vector(iop)
            if normal is not None:
                valid_slices.append({
                    'SOPInstanceUID': row['SOPInstanceUID'],
                    'InstanceNumber': row['InstanceNumber'],
                    'ipp': ipp,
                    'normal': normal,
                    'slice_position': np.dot(ipp, normal)  # Project IPP onto normal
                })
    
    if len(valid_slices) < 2:
        return {
            'SeriesInstanceUID': series_data['SeriesInstanceUID'].iloc[0],
            'valid_slices': len(valid_slices),
            'total_slices': len(series_data),
            'distances': [],
            'mean_distance': None,
            'std_distance': None,
            'min_distance': None,
            'max_distance': None,
            'is_scout': None
        }
    
    # Sort by slice position
    valid_slices.sort(key=lambda x: x['slice_position'])
    
    # Calculate distances between consecutive slices
    distances = []
    for i in range(1, len(valid_slices)):
        dist = abs(valid_slices[i]['slice_position'] - valid_slices[i-1]['slice_position'])
        distances.append(dist)
    
    distances = np.array(distances)
    
    # Heuristic to detect scout images: large variation in distances or very large gaps
    is_scout = False
    if len(distances) > 0:
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        max_dist = np.max(distances)
        
        # Scout detection criteria:
        # 1. Very large standard deviation relative to mean
        # 2. Maximum distance much larger than mean
        if (std_dist > mean_dist * 2) or (max_dist > mean_dist * 5):
            is_scout = True
    
    return {
        'SeriesInstanceUID': series_data['SeriesInstanceUID'].iloc[0],
        'valid_slices': len(valid_slices),
        'total_slices': len(series_data),
        'distances': distances.tolist(),
        'mean_distance': np.mean(distances) if len(distances) > 0 else None,
        'std_distance': np.std(distances) if len(distances) > 0 else None,
        'min_distance': np.min(distances) if len(distances) > 0 else None,
        'max_distance': np.max(distances) if len(distances) > 0 else None,
        'is_scout': is_scout
    }

# %%
def analyze_slice_distances():
    """Main analysis function"""
    logger.info("Loading DICOM instance metadata...")
    
    data_path = Path("../data/processed/dicom_instance_metadata.csv")
    
    # Process data in chunks
    chunk_size = 50000
    results = []
    processed_series = set()
    
    logger.info("Processing DICOM metadata in chunks...")
    chunk_count = 0
    
    for chunk in pd.read_csv(data_path, chunksize=chunk_size):
        chunk_count += 1
        logger.info(f"Processing chunk {chunk_count}...")
        
        # Group by series and analyze each series
        for series_uid, series_data in chunk.groupby('SeriesInstanceUID'):
            if series_uid not in processed_series:
                result = calculate_slice_distances(series_data)
                result['Modality'] = series_data['Modality'].iloc[0]
                results.append(result)
                processed_series.add(series_uid)
                
                if len(results) % 1000 == 0:
                    logger.info(f"Processed {len(results)} series...")
    
    logger.info(f"Completed analysis of {len(results)} series")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Expand distances for detailed analysis
    distance_records = []
    for _, row in results_df.iterrows():
        if row['distances'] and not row['is_scout']:  # Exclude scouts from distance analysis
            for dist in row['distances']:
                distance_records.append({
                    'SeriesInstanceUID': row['SeriesInstanceUID'],
                    'Modality': row['Modality'],
                    'distance': dist
                })
    
    distance_df = pd.DataFrame(distance_records)
    
    # Generate summary statistics
    logger.info("Generating summary statistics...")
    
    print("\n" + "="*60)
    print("SLICE DISTANCE ANALYSIS SUMMARY")
    print("="*60)
    
    total_series = len(results_df)
    series_with_valid_distances = len(results_df[results_df['mean_distance'].notna()])
    scout_series = len(results_df[results_df['is_scout'] == True])
    
    print(f"Total series analyzed: {total_series:,}")
    print(f"Series with valid distances: {series_with_valid_distances:,}")
    print(f"Scout series detected: {scout_series:,}")
    print(f"Non-scout series: {total_series - scout_series:,}")
    
    # Modality breakdown
    print("\nDISTANCE STATISTICS BY MODALITY (excluding scouts):")
    print("-"*60)
    
    non_scout_series = results_df[results_df['is_scout'] != True]
    
    for modality in sorted(non_scout_series['Modality'].unique()):
        modality_data = non_scout_series[non_scout_series['Modality'] == modality]
        modality_distances = distance_df[distance_df['Modality'] == modality]['distance']
        
        if len(modality_distances) > 0:
            print(f"\n{modality}:")
            print(f"  Series count: {len(modality_data):,}")
            print(f"  Distance measurements: {len(modality_distances):,}")
            print(f"  Mean distance: {modality_distances.mean():.3f} mm")
            print(f"  Median distance: {modality_distances.median():.3f} mm")
            print(f"  Std distance: {modality_distances.std():.3f} mm")
            print(f"  Min distance: {modality_distances.min():.3f} mm")
            print(f"  Max distance: {modality_distances.max():.3f} mm")
            print(f"  25th percentile: {modality_distances.quantile(0.25):.3f} mm")
            print(f"  75th percentile: {modality_distances.quantile(0.75):.3f} mm")
    
    # Save detailed results
    output_path = Path("../data/processed/slice_distances.csv")
    results_df.to_csv(output_path, index=False)
    logger.info(f"Series-level results saved to {output_path}")
    
    distance_output_path = Path("../data/processed/slice_distances_detailed.csv")
    distance_df.to_csv(distance_output_path, index=False)
    logger.info(f"Distance-level results saved to {distance_output_path}")
    
    return results_df, distance_df

# %%
def create_distribution_plots(distance_df):
    """Create distribution plots for slice distances by modality"""
    logger.info("Creating distribution plots...")
    
    # Create reports directory if it doesn't exist
    reports_dir = Path("../reports/figs")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Get modalities with sufficient data
    modality_counts = distance_df['Modality'].value_counts()
    modalities_to_plot = modality_counts[modality_counts >= 100].index.tolist()
    
    if len(modalities_to_plot) == 0:
        logger.warning("No modalities with sufficient data for plotting")
        return
    
    # Create distribution plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Slice Distance Distributions by Modality', fontsize=16)
    
    # Plot 1: Histogram
    ax1 = axes[0, 0]
    for modality in modalities_to_plot[:6]:  # Limit to top 6 modalities
        modality_data = distance_df[distance_df['Modality'] == modality]['distance']
        ax1.hist(modality_data, bins=50, alpha=0.6, label=modality, density=True)
    ax1.set_xlabel('Distance (mm)')
    ax1.set_ylabel('Density')
    ax1.set_title('Histogram of Slice Distances')
    ax1.legend()
    ax1.set_xlim(0, min(20, distance_df['distance'].quantile(0.95)))  # Focus on main distribution
    
    # Plot 2: Box plot
    ax2 = axes[0, 1]
    modalities_subset = modalities_to_plot[:8]  # Limit for readability
    box_data = [distance_df[distance_df['Modality'] == mod]['distance'].values 
                for mod in modalities_subset]
    ax2.boxplot(box_data, labels=modalities_subset)
    ax2.set_ylabel('Distance (mm)')
    ax2.set_title('Box Plot of Slice Distances')
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_ylim(0, min(15, distance_df['distance'].quantile(0.95)))
    
    # Plot 3: Log scale histogram
    ax3 = axes[1, 0]
    for modality in modalities_to_plot[:6]:
        modality_data = distance_df[distance_df['Modality'] == modality]['distance']
        # Filter out zeros for log scale
        modality_data = modality_data[modality_data > 0]
        if len(modality_data) > 0:
            ax3.hist(modality_data, bins=50, alpha=0.6, label=modality, density=True)
    ax3.set_xlabel('Distance (mm)')
    ax3.set_ylabel('Density')
    ax3.set_title('Log Scale Distribution of Slice Distances')
    ax3.set_yscale('log')
    ax3.legend()
    
    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    stats_data = []
    for modality in modalities_to_plot[:8]:
        modality_distances = distance_df[distance_df['Modality'] == modality]['distance']
        stats_data.append([
            modality_distances.mean(),
            modality_distances.median(),
            modality_distances.std()
        ])
    
    stats_df = pd.DataFrame(stats_data, 
                           columns=['Mean', 'Median', 'Std Dev'],
                           index=modalities_to_plot[:8])
    
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
    plot_path = reports_dir / "slice_distance_distributions.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Distribution plots saved to {plot_path}")
    plt.show()

# %%
if __name__ == "__main__":
    results_df, distance_df = analyze_slice_distances()
    
    if len(distance_df) > 0:
        create_distribution_plots(distance_df)
    else:
        logger.warning("No distance data available for plotting")