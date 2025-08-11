"""
Test processing of specific SeriesInstanceUIDs that were previously problematic.

This test validates the processing pipeline for 5 specific series that had issues
during initial development and debugging.
"""

import pytest
import pandas as pd
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.append('/home/hongrui/work/kaggle_rsna_2025')

from modules import process_single_series, get_config

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# These are the 5 specific SeriesInstanceUIDs that were previously problematic
PROBLEMATIC_SERIES = [
    "1.2.826.0.1.3680043.8.498.24059937312846701557229931292132131003",
    "1.2.826.0.1.3680043.8.498.75712554178574230484227682423862727306", 
    "1.2.826.0.1.3680043.8.498.82768897201281605198635077495114055892",
    "1.2.826.0.1.3680043.8.498.10004044428023505108375152878107656647",
    "1.2.826.0.1.3680043.8.498.10035643165968342618460849823699311381"
]

class TestSpecificSeries:
    """Test class for validating processing of specific series."""
    
    def setup_method(self):
        """Set up test configuration."""
        self.config = get_config()
        self.series_dir = self.config.get_series_dir()
        logger.info(f"Testing with series directory: {self.series_dir}")
    
    def test_series_directories_exist(self):
        """Test that all problematic series directories exist."""
        missing_series = []
        
        for series_uid in PROBLEMATIC_SERIES:
            series_path = self.series_dir / series_uid
            if not series_path.exists():
                missing_series.append(series_uid)
                logger.warning(f"Series directory not found: {series_path}")
        
        if missing_series:
            pytest.skip(f"Missing series directories: {missing_series}")
    
    @pytest.mark.parametrize("series_uid", PROBLEMATIC_SERIES)
    def test_process_single_series(self, series_uid):
        """Test processing of each problematic series individually."""
        logger.info(f"Testing processing of series: {series_uid}")
        
        # Skip if series directory doesn't exist
        series_path = self.series_dir / series_uid
        if not series_path.exists():
            pytest.skip(f"Series directory not found: {series_path}")
        
        # Process the series
        result_df = process_single_series(series_uid)
        
        # Validate results
        assert isinstance(result_df, pd.DataFrame), f"Expected DataFrame, got {type(result_df)}"
        
        if len(result_df) > 0:
            # Validate that SeriesInstanceUID is consistent
            assert result_df['SeriesInstanceUID'].nunique() == 1, "Multiple SeriesInstanceUIDs in result"
            assert result_df['SeriesInstanceUID'].iloc[0] == series_uid, "SeriesInstanceUID mismatch"
            
            # Validate required DICOM columns are present
            required_dicom_columns = [
                'SOPInstanceUID', 'dicom_path', 'pixel_shape', 'Rows', 'Columns'
            ]
            for col in required_dicom_columns:
                assert col in result_df.columns, f"Missing required column: {col}"
            
            # Validate that we have actual data
            assert result_df['SOPInstanceUID'].notna().any(), "No valid SOPInstanceUIDs found"
            assert result_df['dicom_path'].notna().any(), "No valid DICOM paths found"
            
            logger.info(f"Successfully processed series {series_uid}: {len(result_df)} images")
            
        else:
            logger.warning(f"No data extracted for series: {series_uid}")
    
    def test_all_series_processing(self):
        """Test processing all problematic series together."""
        results = {}
        total_images = 0
        
        for series_uid in PROBLEMATIC_SERIES:
            series_path = self.series_dir / series_uid
            if not series_path.exists():
                logger.warning(f"Skipping non-existent series: {series_uid}")
                continue
                
            try:
                result_df = process_single_series(series_uid)
                results[series_uid] = len(result_df)
                total_images += len(result_df)
                logger.info(f"Series {series_uid}: {len(result_df)} images")
            except Exception as e:
                logger.error(f"Failed to process series {series_uid}: {e}")
                results[series_uid] = 0
        
        # Print summary
        logger.info("=== PROCESSING SUMMARY ===")
        logger.info(f"Total images processed: {total_images}")
        for series_uid, count in results.items():
            logger.info(f"{series_uid}: {count} images")
        
        # Validate that at least some series were processed successfully
        successful_series = [s for s, count in results.items() if count > 0]
        assert len(successful_series) > 0, "No series were processed successfully"
    
    def test_data_structure_consistency(self):
        """Test that all processed series have consistent data structure."""
        all_columns = set()
        series_columns = {}
        
        for series_uid in PROBLEMATIC_SERIES:
            series_path = self.series_dir / series_uid
            if not series_path.exists():
                continue
                
            try:
                result_df = process_single_series(series_uid)
                if len(result_df) > 0:
                    series_columns[series_uid] = set(result_df.columns)
                    all_columns.update(result_df.columns)
                    logger.info(f"Series {series_uid}: {len(result_df.columns)} columns")
            except Exception as e:
                logger.error(f"Failed to process series {series_uid}: {e}")
        
        # Check that we have some common structure
        if series_columns:
            # Find columns that are common across all processed series
            common_columns = set.intersection(*series_columns.values()) if series_columns else set()
            logger.info(f"Common columns across all series: {len(common_columns)}")
            logger.info(f"Total unique columns: {len(all_columns)}")
            
            # Validate that essential columns are always present
            essential_columns = {'SeriesInstanceUID', 'SOPInstanceUID', 'dicom_path'}
            missing_essential = essential_columns - common_columns
            assert not missing_essential, f"Missing essential columns: {missing_essential}"

if __name__ == "__main__":
    """Run tests directly for debugging."""
    test_instance = TestSpecificSeries()
    test_instance.setup_method()
    
    print("Testing specific problematic series...")
    print(f"Series to test: {PROBLEMATIC_SERIES}")
    
    # Run the comprehensive test
    test_instance.test_all_series_processing()