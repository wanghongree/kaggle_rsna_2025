# Scripts Documentation

## 01_eda_dcm_number.py

**Purpose**: Count the number of DICOM files for each series in the RSNA Intracranial Aneurysm dataset.

**What it does**:
- Scans the `data/series/` directory to find all series subdirectories
- Counts the number of `.dcm` files in each series directory
- Creates a summary CSV with SeriesInstanceUID and corresponding DICOM file count
- Saves results to `data/processed/dcm_file_counts_per_series.csv`
- Prints basic statistics about the file count distribution

**Output**: 
- CSV file: `data/processed/dcm_file_counts_per_series.csv`
- Columns: `SeriesInstanceUID`, `dcm_file_count`

**Usage**: Run from project root directory with `python scripts/01_eda_dcm_number.py`

## 02_eda_dcm_by_modality.py

**Purpose**: Analyze DICOM file count distribution across different imaging modalities.

**What it does**:
- Reads `data/train.csv` and merges it with `data/processed/dcm_file_counts_per_series.csv`
- Saves the merged dataset to `data/processed/train_with_dcm_counts.csv`
- Generates individual histograms for each modality (CTA, MRA, MRI, etc.)
- Creates a combined histogram with subplots showing all modalities
- Generates a boxplot comparing DICOM file counts across modalities
- Prints summary statistics by modality
- Saves all visualizations to `reports/` folder

**Output**: 
- Merged dataset: `data/processed/train_with_dcm_counts.csv`
- Individual histogram images: `reports/dcm_count_histogram_{modality}.png`
- Combined histogram: `reports/dcm_count_histograms_combined.png`
- Boxplot comparison: `reports/dcm_count_boxplot_by_modality.png`

**Dependencies**: Requires `01_eda_dcm_number.py` to be run first to generate DICOM counts.

**Usage**: Run from project root directory with `python scripts/02_eda_dcm_by_modality.py`

## 03_add_segmentation_info.py

**Purpose**: Add segmentation data availability information to the training dataset.

**What it does**:
- Scans `data/segmentations/` directory to identify which series have segmentation data
- Extracts SeriesInstanceUID from directory names and identifies `.nii` and `_cowseg.nii` files
- Left joins segmentation information with `data/processed/train_with_dcm_counts.csv`
- Adds segmentation indicators and file names to the dataset
- Provides summary statistics about segmentation availability by modality and aneurysm presence
- Saves enhanced dataset to `data/processed/train_with_dcm_and_segmentation.csv`

**Output**: 
- Enhanced dataset: `data/processed/train_with_dcm_and_segmentation.csv`
- New columns: `has_segmentation`, `main_nii_file`, `cowseg_nii_file`, `segmentation_file_count`

**Dependencies**: Requires `02_eda_dcm_by_modality.py` to be run first to generate train data with DICOM counts.

**Usage**: Run from project root directory with `python scripts/03_add_segmentation_info.py`

## 04_segmentation_summary.py

**Purpose**: Generate comprehensive summary of segmentation data availability by modality.

**What it does**:
- Reads `data/processed/train_with_dcm_and_segmentation.csv`
- Creates detailed summary statistics of segmentation availability by modality
- Generates breakdown by modality and aneurysm presence
- Creates multiple visualizations showing segmentation distribution patterns
- Provides insights on which modalities have the best segmentation coverage
- Saves summary tables and charts for further analysis

**Output**: 
- Summary table: `data/processed/segmentation_summary_by_modality.csv`
- Detailed breakdown: `data/processed/segmentation_detailed_breakdown.csv`
- Charts: `reports/segmentation_availability_by_modality.png`
- Overall distribution: `reports/overall_segmentation_distribution.png`
- Aneurysm breakdown: `reports/segmentation_by_aneurysm_presence.png`

**Dependencies**: Requires `03_add_segmentation_info.py` to be run first to generate the enhanced dataset.

**Usage**: Run from project root directory with `python scripts/04_segmentation_summary.py`

## 05_dicom_instance_metadata.py

**Purpose**: Extract comprehensive instance-level metadata from all DICOM files in the dataset.

**What it does**:
- Reads every DICOM instance across all series in `data/series/`
- Extracts pixel data shapes (2D/3D/4D) and dimensions for each instance
- Extracts 25 specified DICOM tags per instance including technical parameters, positioning, and identifiers
- Implements incremental saving with checkpoint/resume capability for large datasets
- Handles errors gracefully and continues processing, recording error details
- Creates instance-level dataset with one row per DICOM file
- Provides comprehensive statistics and error analysis

**Key Features**:
- **Incremental Processing**: Saves progress periodically and can resume from interruptions
- **Error Handling**: Continues processing on errors, logs details, and includes error records
- **Memory Efficient**: Processes files individually to handle large datasets
- **Comprehensive Extraction**: 25+ metadata fields plus pixel data information

**DICOM Tags Extracted**:
`BitsAllocated`, `BitsStored`, `Columns`, `FrameOfReferenceUID`, `HighBit`, `ImageOrientationPatient`, `ImagePositionPatient`, `InstanceNumber`, `Modality`, `PatientID`, `PhotometricInterpretation`, `PixelRepresentation`, `PixelSpacing`, `PlanarConfiguration`, `RescaleIntercept`, `RescaleSlope`, `RescaleType`, `Rows`, `SOPClassUID`, `SOPInstanceUID`, `SamplesPerPixel`, `SliceThickness`, `SpacingBetweenSlices`, `StudyInstanceUID`, `TransferSyntaxUID`

**Output**: 
- Main dataset: `data/processed/dicom_instance_metadata.csv` (instance-level)
- Columns: 32 total including metadata fields, pixel shape info, and error tracking
- Key columns: `SeriesInstanceUID`, `instance_filename`, `pixel_shape`, `dimension`, `shape_str`, `error`, plus all DICOM tags

**Test Version**: `05_dicom_instance_metadata_test.py` processes only 3 series for testing

**Dependencies**: None - can be run independently

**Usage**: 
- Full processing: `python scripts/05_dicom_instance_metadata.py`
- Test version: `python scripts/05_dicom_instance_metadata_test.py`