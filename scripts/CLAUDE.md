# 01_create_meta_data.py

## Overview

The `01_create_meta_data.py` script is the primary data processing pipeline for the RSNA Intracranial Aneurysm Detection project. It extracts comprehensive metadata from DICOM medical imaging files and combines them with training labels and segmentation data to create a unified dataset for machine learning model training.

## What the script does

### Core functionality:
1. **DICOM Metadata Extraction**: Processes thousands of DICOM files to extract medical imaging metadata including pixel dimensions, spacing, orientation, and technical parameters
2. **Incremental Processing**: Processes data series-by-series with automatic resume capability to handle long-running operations safely
3. **Data Integration**: Combines DICOM metadata with:
   - Training labels from `train.csv` (aneurysm presence/absence by anatomical location)
   - Spatial localization data from `train_localizers.csv` (aneurysm coordinates)
   - Vessel segmentation metadata from NIfTI files
4. **Robust Data Persistence**: Saves processed data incrementally with parquet/CSV fallback mechanisms

### Processing workflow:
1. Load configuration and determine output paths
2. Read `train.csv` to get list of all SeriesInstanceUIDs to process  
3. Check existing output files to identify already-processed series (resume capability)
4. For each remaining series:
   - Extract DICOM tags from all `.dcm` files in the series directory
   - Join with training labels from `train.csv` 
   - Join with localization data from `train_localizers.csv`
   - Process corresponding NIfTI segmentation files (if available)
   - Append results to output file immediately (incremental save)
5. Provide comprehensive processing summary and statistics

## Key features:

### Incremental Processing
- **Series-by-series processing**: Processes one SeriesInstanceUID at a time instead of batch processing
- **Automatic resume**: Skips already-processed series on restart
- **Immediate persistence**: Saves data after each series to prevent data loss
- **Progress tracking**: Detailed logging and progress bars

### Data Integration
- **Multi-source joins**: Combines DICOM, CSV, and segmentation data
- **Column conflict handling**: Uses suffixes to resolve naming conflicts (e.g., 'Modality_dicom' vs 'Modality_train')
- **Flexible file format support**: Handles both parquet and CSV with automatic fallback

### Error Handling
- **Graceful degradation**: Continues processing if individual series fail
- **Comprehensive logging**: Detailed error messages and warnings
- **File format resilience**: Falls back to CSV if parquet operations fail

## Input data sources:
- `data/raw/series/{SeriesInstanceUID}/*.dcm` - DICOM imaging files
- `data/raw/train.csv` - Primary training labels  
- `data/raw/train_localizers.csv` - Aneurysm spatial coordinates
- `data/raw/segmentations/{SeriesInstanceUID}/*.nii` - Vessel segmentation files

## Output:
- `data/interim/dicom_metadata_with_csv.parquet` (or `.csv` fallback) - Unified dataset containing:
  - DICOM technical metadata (25+ fields including pixel spacing, orientation, modality)
  - Training labels (13 binary aneurysm location columns + demographics)
  - Spatial localization coordinates (when available)
  - Segmentation file metadata (when available)

## Usage:
```bash
python scripts/01_create_meta_data.py
```

The script is designed to be run multiple times safely - it will automatically resume from where it left off if interrupted.

## Architecture notes:
- Uses the unified `modules.processing` module for all DICOM and segmentation processing
- Implements proper logging with timestamps and level-based filtering
- Follows pandas best practices with vectorized operations and efficient joins
- Memory-efficient processing by handling one series at a time
- Configurable paths through `modules.config` for environment flexibility