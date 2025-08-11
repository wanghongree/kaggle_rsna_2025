# %% house keeping
import pandas as pd
from pathlib import Path
import warnings
import logging
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Import project modules
import sys
sys.path.append('/home/hongrui/work/kaggle_rsna_2025')
from modules import (
    process_single_series, 
    get_already_processed_series, 
    append_series_data, 
    get_config
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# %% task 1: Create DICOM image level data table with incremental processing

def get_series_to_process(config) -> list:
    """Get list of SeriesInstanceUIDs that need to be processed from train.csv."""
    train_csv_path = config.get_train_csv()
    
    if not train_csv_path.exists():
        logger.error(f"Train CSV file not found: {train_csv_path}")
        return []
    
    train_df = pd.read_csv(train_csv_path)
    all_series = train_df['SeriesInstanceUID'].unique().tolist()
    logger.info(f"Found {len(all_series)} series in train.csv")
    return all_series


def main():
    """Main processing function with series-by-series incremental processing."""
    logger.info("Starting DICOM metadata extraction with incremental processing")
    
    # Load configuration
    config = get_config()
    output_path = config.get_dicom_metadata_output()
    
    # Get list of all series that need to be processed
    all_series = get_series_to_process(config)
    if not all_series:
        logger.error("No series found to process")
        return
    
    # Get already processed series
    processed_series = get_already_processed_series(output_path)
    remaining_series = [s for s in all_series if s not in processed_series]
    
    logger.info(f"Total series to process: {len(all_series)}")
    logger.info(f"Already processed series: {len(processed_series)}")
    logger.info(f"Remaining series to process: {len(remaining_series)}")
    
    if not remaining_series:
        logger.info("All series have already been processed!")
        
        # Display summary of existing data
        if output_path.exists():
            df_final = pd.read_parquet(output_path)
            print("\n=== SUMMARY ===" )
            print(f"Total DICOM images: {len(df_final)}")
            print(f"Unique series: {df_final['SeriesInstanceUID'].nunique()}")
            print(f"Images with localizer data: {df_final['coordinates'].notna().sum()}")
            if 'nii_path' in df_final.columns:
                print(f"Series with segmentation data: {df_final['nii_path'].notna().sum()}")
        return
    
    # Process remaining series one by one
    successful_count = 0
    failed_series = []
    
    logger.info(f"Processing {len(remaining_series)} remaining series...")
    
    for series_uid in tqdm(remaining_series, desc="Processing series"):
        try:
            logger.info(f"Processing series: {series_uid}")
            
            # Process this series
            series_data = process_single_series(series_uid)
            
            if len(series_data) > 0:
                # Save this series data
                append_series_data(series_data, output_path)
                successful_count += 1
                logger.info(f"Successfully processed series {series_uid}: {len(series_data)} images")
            else:
                logger.warning(f"No data extracted for series: {series_uid}")
                failed_series.append(series_uid)
                
        except Exception as e:
            logger.error(f"Error processing series {series_uid}: {e}")
            failed_series.append(series_uid)
            continue
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("PROCESSING COMPLETE")
    logger.info("="*60)
    logger.info(f"Successfully processed: {successful_count} series")
    logger.info(f"Failed to process: {len(failed_series)} series")
    
    if failed_series:
        logger.warning(f"Failed series: {failed_series[:10]}...")  # Show first 10
    
    # Display final summary
    if output_path.exists():
        df_final = pd.read_parquet(output_path)
        print("\n=== FINAL SUMMARY ===")
        print(f"Total DICOM images processed: {len(df_final)}")
        print(f"Unique series: {df_final['SeriesInstanceUID'].nunique()}")
        print(f"Images with localizer data: {df_final['coordinates'].notna().sum()}")
        if 'nii_path' in df_final.columns:
            print(f"Series with segmentation data: {df_final['nii_path'].notna().sum()}")
        
        # Display sample of the data
        print("\n=== SAMPLE DATA ===")
        print(df_final.head())
        print("\n=== COLUMNS ===")
        print(df_final.columns.tolist())
        
        print(f"\nData saved to: {output_path}")
    
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()