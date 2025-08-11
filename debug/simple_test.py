#!/usr/bin/env python3
"""
Simple test script to debug the 5 specific series processing.
"""

import sys
import logging
sys.path.append('/home/hongrui/work/kaggle_rsna_2025')

from modules import process_single_series, get_config

# Configure logging to file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/hongrui/work/kaggle_rsna_2025/debug/processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# The 5 specific SeriesInstanceUIDs
TEST_SERIES = [
    "1.2.826.0.1.3680043.8.498.24059937312846701557229931292132131003",
    "1.2.826.0.1.3680043.8.498.75712554178574230484227682423862727306", 
    "1.2.826.0.1.3680043.8.498.82768897201281605198635077495114055892",
    "1.2.826.0.1.3680043.8.498.10004044428023505108375152878107656647",
    "1.2.826.0.1.3680043.8.498.10035643165968342618460849823699311381"
]

def test_series_processing():
    """Test processing each series individually."""
    print("=" * 60)
    print("TESTING 5 SPECIFIC SERIES")
    print("=" * 60)
    
    config = get_config()
    series_dir = config.get_series_dir()
    print(f"Series directory: {series_dir}")
    
    results = {}
    total_images = 0
    
    for i, series_uid in enumerate(TEST_SERIES, 1):
        print(f"\n[{i}/5] Testing series: {series_uid}")
        
        # Check if series directory exists
        series_path = series_dir / series_uid
        if not series_path.exists():
            print(f"❌ Series directory not found: {series_path}")
            results[series_uid] = {"status": "missing", "images": 0, "error": "Directory not found"}
            continue
        
        try:
            # Process the series
            result_df = process_single_series(series_uid)
            
            if len(result_df) > 0:
                results[series_uid] = {"status": "success", "images": len(result_df), "error": None}
                total_images += len(result_df)
                print(f"✅ Success: {len(result_df)} images processed")
                
                # Show basic info about the data
                print(f"   Columns: {len(result_df.columns)}")
                print(f"   SeriesInstanceUID unique: {result_df['SeriesInstanceUID'].nunique()}")
                if 'coordinates' in result_df.columns:
                    localizer_count = result_df['coordinates'].notna().sum()
                    print(f"   Images with localizer data: {localizer_count}")
                if 'nii_path' in result_df.columns:
                    seg_count = result_df['nii_path'].notna().sum()
                    print(f"   Images with segmentation data: {seg_count}")
                    
            else:
                results[series_uid] = {"status": "empty", "images": 0, "error": "No data extracted"}
                print("⚠️  Warning: No data extracted")
                
        except Exception as e:
            results[series_uid] = {"status": "error", "images": 0, "error": str(e)}
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    successful = [s for s, r in results.items() if r["status"] == "success"]
    failed = [s for s, r in results.items() if r["status"] in ["error", "missing"]]
    empty = [s for s, r in results.items() if r["status"] == "empty"]
    
    print(f"✅ Successful: {len(successful)} series ({sum(r['images'] for r in results.values() if r['status'] == 'success')} total images)")
    print(f"❌ Failed: {len(failed)} series")
    print(f"⚠️  Empty: {len(empty)} series")
    
    # Detailed results
    print(f"\nDetailed Results:")
    for series_uid, result in results.items():
        status_icon = {"success": "✅", "error": "❌", "missing": "❌", "empty": "⚠️"}[result["status"]]
        print(f"{status_icon} {series_uid}: {result['images']} images - {result.get('error', 'OK')}")
    
    # Summary for debug file
    with open('/home/hongrui/work/kaggle_rsna_2025/debug/test_summary.txt', 'w') as f:
        f.write("SERIES PROCESSING TEST SUMMARY\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total series tested: {len(TEST_SERIES)}\n")
        f.write(f"Successful: {len(successful)}\n")
        f.write(f"Failed: {len(failed)}\n") 
        f.write(f"Empty: {len(empty)}\n")
        f.write(f"Total images processed: {total_images}\n\n")
        
        f.write("DETAILED RESULTS:\n")
        for series_uid, result in results.items():
            f.write(f"{series_uid}: {result['status']} - {result['images']} images")
            if result['error']:
                f.write(f" - {result['error']}")
            f.write("\n")

if __name__ == "__main__":
    test_series_processing()