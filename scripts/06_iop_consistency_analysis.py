# %%
"""
IOP (Image Orientation Patient) Consistency Analysis
Checks if all instances within each series have the same IOP values
"""

import pandas as pd
import numpy as np
import ast
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# %%
def parse_iop_string(iop_str):
    """Parse IOP string to numpy array"""
    try:
        if pd.isna(iop_str) or iop_str == '' or iop_str == 'nan':
            return None
        # Handle string representation of list
        iop_list = ast.literal_eval(iop_str)
        return np.array(iop_list)
    except (ValueError, SyntaxError) as e:
        logger.warning(f"Failed to parse IOP: {iop_str}, error: {e}")
        return None

# %%
def check_iop_consistency(series_data):
    """Check if all IOPs in a series are consistent"""
    iop_arrays = []
    valid_count = 0
    
    for iop_str in series_data['ImageOrientationPatient']:
        iop_array = parse_iop_string(iop_str)
        if iop_array is not None:
            iop_arrays.append(iop_array)
            valid_count += 1
    
    if valid_count == 0:
        return {
            'consistent': None,
            'valid_iop_count': 0,
            'total_instances': len(series_data),
            'reference_iop': None
        }
    
    if valid_count == 1:
        return {
            'consistent': True,
            'valid_iop_count': 1,
            'total_instances': len(series_data),
            'reference_iop': iop_arrays[0].tolist()
        }
    
    # Check consistency by comparing all IOPs to the first valid one
    reference_iop = iop_arrays[0]
    consistent = True
    tolerance = 1e-6  # Small tolerance for floating point comparison
    
    for iop_array in iop_arrays[1:]:
        if not np.allclose(reference_iop, iop_array, atol=tolerance):
            consistent = False
            break
    
    return {
        'consistent': consistent,
        'valid_iop_count': valid_count,
        'total_instances': len(series_data),
        'reference_iop': reference_iop.tolist()
    }

# %%
def analyze_iop_consistency():
    """Main analysis function"""
    logger.info("Loading DICOM instance metadata...")
    
    # Load data in chunks to handle large file
    data_path = Path("../data/processed/dicom_instance_metadata.csv")
    
    # First, get unique series for progress tracking
    logger.info("Counting unique series...")
    unique_series = pd.read_csv(data_path, usecols=['SeriesInstanceUID']).drop_duplicates()
    total_series = len(unique_series)
    logger.info(f"Found {total_series} unique series to analyze")
    
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
                result = check_iop_consistency(series_data)
                result['SeriesInstanceUID'] = series_uid
                result['Modality'] = series_data['Modality'].iloc[0]
                results.append(result)
                processed_series.add(series_uid)
                
                if len(results) % 1000 == 0:
                    logger.info(f"Processed {len(results)} series...")
    
    logger.info(f"Completed analysis of {len(results)} series")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Generate summary statistics
    logger.info("Generating summary statistics...")
    
    total_series_analyzed = len(results_df)
    series_with_valid_iop = len(results_df[results_df['valid_iop_count'] > 0])
    consistent_series = len(results_df[results_df['consistent'] == True])
    inconsistent_series = len(results_df[results_df['consistent'] == False])
    no_iop_series = len(results_df[results_df['consistent'].isna()])
    
    print("\n" + "="*60)
    print("IOP CONSISTENCY ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total series analyzed: {total_series_analyzed:,}")
    print(f"Series with valid IOP data: {series_with_valid_iop:,} ({series_with_valid_iop/total_series_analyzed*100:.1f}%)")
    print(f"Series with consistent IOP: {consistent_series:,} ({consistent_series/series_with_valid_iop*100:.1f}% of valid)")
    print(f"Series with inconsistent IOP: {inconsistent_series:,} ({inconsistent_series/series_with_valid_iop*100:.1f}% of valid)")
    print(f"Series without IOP data: {no_iop_series:,}")
    
    # Modality breakdown
    print("\nBREAKDOWN BY MODALITY:")
    print("-"*40)
    modality_summary = results_df.groupby('Modality').agg({
        'SeriesInstanceUID': 'count',
        'consistent': lambda x: (x == True).sum(),
        'valid_iop_count': lambda x: (x > 0).sum()
    }).rename(columns={
        'SeriesInstanceUID': 'total_series',
        'consistent': 'consistent_series',
        'valid_iop_count': 'series_with_iop'
    })
    
    modality_summary['consistency_rate'] = (modality_summary['consistent_series'] / 
                                          modality_summary['series_with_iop'] * 100).round(1)
    
    for modality, row in modality_summary.iterrows():
        print(f"{modality}: {row['consistent_series']}/{row['series_with_iop']} consistent ({row['consistency_rate']:.1f}%)")
    
    # Save detailed results
    output_path = Path("../data/processed/iop_consistency_summary.csv")
    results_df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")
    
    return results_df

# %%
if __name__ == "__main__":
    results = analyze_iop_consistency()