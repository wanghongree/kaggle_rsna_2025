######################
# Refactored Code
######################
import os
import warnings
from typing import Tuple, Optional, List, Dict, Any

import cv2
import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import Dataset
from pydicom.multival import MultiValue

# --- Constants ---
# Moved to module level as they are true constants.
TAGS_TO_EXTRACT = [
    'BitsAllocated', 'BitsStored', 'Columns', 'FrameOfReferenceUID', 'HighBit',
    'ImageOrientationPatient', 'ImagePositionPatient', 'InstanceNumber', 'Modality',
    'PatientID', 'PhotometricInterpretation', 'PixelRepresentation', 'PixelSpacing',
    'PlanarConfiguration', 'RescaleIntercept', 'RescaleSlope', 'RescaleType',
    'Rows', 'SOPClassUID', 'SOPInstanceUID', 'SamplesPerPixel', 'SliceThickness',
    'SpacingBetweenSlices', 'StudyInstanceUID', 'TransferSyntaxUID'
]
REJECT_IMAGE_TYPES = {'LOCALIZER', 'SCOUT', 'SECONDARY', 'MPR'}

# --- Helper Functions for Loading and Validation ---

def _validate_parameters(series_path: str, mode: str):
    """Validates the initial input parameters."""
    if not os.path.isdir(series_path):
        raise FileNotFoundError(f"Directory not found: {series_path}")
    if mode.lower() not in ['prod', 'dev']:
        raise ValueError(f"Unknown mode: '{mode}'. Options are 'prod' or 'dev'.")

def _is_slice_valid(ds: Dataset, filter_invalid: bool) -> bool:
    """Checks if a single DICOM slice is valid for inclusion in the volume."""
    if not hasattr(ds, 'PixelData'):
        raise ValueError("File has no pixel data.")

    if ds.get('SamplesPerPixel') == 3 or 'RGB' in ds.get('PhotometricInterpretation', ''):
        raise ValueError("Color DICOM images are not supported.")
    
    if filter_invalid:
        image_type = set(ds.get('ImageType', []))
        if REJECT_IMAGE_TYPES.intersection(image_type):
            raise ValueError(f"Filtered as invalid ImageType: {image_type}")

        pixel_array_check = ds.pixel_array
        if pixel_array_check.min() == pixel_array_check.max():
            raise ValueError("Filtered as blank/constant pixel data.")
            
    return True


def _extract_tags(ds: Dataset, filename: str) -> Dict[str, Any]:
    """Extracts a predefined list of tags from a DICOM dataset."""
    slice_tags = {'FileName': filename}

    def try_float(val):
        try:
            return float(val)
        except (ValueError, TypeError):
            return val  # leave as-is if not numeric

    for tag in TAGS_TO_EXTRACT:
        value = ds.get(tag, None)

        if isinstance(value, MultiValue):
            slice_tags[tag] = [try_float(v) for v in value]
        else:
            slice_tags[tag] = try_float(value)

    return slice_tags


def _load_and_validate_slices(
    series_path: str, mode: str, filter_invalid: bool
) -> List[Tuple[Dataset, Dict[str, Any]]]:
    """Walks the directory, reads, validates, and extracts tags from DICOM files."""
    validated_slices = []
    run_mode = mode.lower()

    for root, _, files in os.walk(series_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                ds = pydicom.dcmread(file_path)
                if _is_slice_valid(ds, filter_invalid):
                    tags = _extract_tags(ds, file)
                    validated_slices.append((ds, tags))
            except Exception as e:
                if run_mode == 'prod':
                    warnings.warn(f"Skipping file '{file}' due to error: {e}", UserWarning)
                    continue
                else:  # mode == 'dev'
                    raise type(e)(f"Error processing file '{file_path}': {e}") from e
    
    return validated_slices

# --- Helper Functions for Sorting ---

def _sort_by_ipp(items: List[Tuple[Dataset, Dict]]):
    """Sorts a list of (dataset, tags) tuples by spatial position (IPP)."""
    first_ds = items[0][0]
    row_vec = np.array(first_ds.ImageOrientationPatient[:3])
    col_vec = np.array(first_ds.ImageOrientationPatient[3:])
    normal_vec = np.cross(row_vec, col_vec)
    items.sort(key=lambda item: np.dot(np.array(item[0].ImagePositionPatient), normal_vec))

def _sort_by_instance_number(items: List[Tuple[Dataset, Dict]]):
    """Sorts a list of (dataset, tags) tuples by InstanceNumber."""
    items.sort(key=lambda item: int(item[0].InstanceNumber))

def _sort_dicom_series(
    validated_slices: List[Tuple[Dataset, Dict]], sort_by: Optional[str]
):
    """Orchestrates sorting of DICOM slices based on the specified method."""
    if not sort_by:
        return

    sort_mode = sort_by.lower()
    try:
        if sort_mode == 'ipp':
            _sort_by_ipp(validated_slices)
        elif sort_mode == 'instance_number':
            _sort_by_instance_number(validated_slices)
        elif sort_mode == 'fallback':
            try:
                _sort_by_ipp(validated_slices)
            except (AttributeError, IndexError):
                warnings.warn("Could not sort by IPP/IOP. Falling back to InstanceNumber.", UserWarning)
                _sort_by_instance_number(validated_slices)
        else:
            raise ValueError(f"Unknown sort_by mode: '{sort_by}'.")
    except Exception as e:
        warnings.warn(f"Sorting failed with error: '{e}'. Proceeding with unsorted slices.", UserWarning)

# --- Helper Functions for Volume Creation and Post-Processing ---

def _create_volume_from_datasets(
    dicom_datasets: List[Dataset], resize_to: Optional[Tuple[int, int]]
) -> np.ndarray:
    """Extracts pixel data from datasets, resizes, and stacks them into a 3D volume."""
    all_frames = []
    target_dims = (resize_to[1], resize_to[0]) if resize_to else None

    for ds in dicom_datasets:
        pixel_array = ds.pixel_array.astype(np.float32)
        frames = [pixel_array] if pixel_array.ndim == 2 else list(pixel_array)
        
        for frame in frames:
            if target_dims:
                frame = cv2.resize(frame, target_dims, interpolation=cv2.INTER_LINEAR)
            all_frames.append(frame)

    return np.stack(all_frames, axis=0)

def _apply_rescale(volume: np.ndarray, meta: Dataset) -> np.ndarray:
    """Applies RescaleSlope and RescaleIntercept if present."""
    slope = float(meta.get('RescaleSlope', 1.0))
    intercept = float(meta.get('RescaleIntercept', 0.0))
    if slope != 1.0 or intercept != 0.0:
        return volume * slope + intercept
    return volume

def _apply_percentile_clip(
    volume: np.ndarray,
    clip_range: Tuple[float, float],
    sampling_size: Optional[int]
) -> np.ndarray:
    """Clips the volume's intensity values to the given percentile range."""
    low_p, high_p = clip_range
    if sampling_size and volume.size > sampling_size:
        indices = np.random.choice(volume.size, size=sampling_size, replace=False)
        p_low, p_high = np.percentile(volume.ravel()[indices], [low_p, high_p])
    else:
        p_low, p_high = np.percentile(volume, [low_p, high_p])
    
    return np.clip(volume, p_low, p_high)

def _get_window_from_tags(ds_meta: Dataset) -> Optional[Tuple[float, float]]:
    """Extracts window center and width from DICOM tags."""
    if 'WindowCenter' in ds_meta and 'WindowWidth' in ds_meta:
        wc_val, ww_val = ds_meta.WindowCenter, ds_meta.WindowWidth
        center = float(wc_val[0] if isinstance(wc_val, MultiValue) else wc_val)
        width = float(ww_val[0] if isinstance(ww_val, MultiValue) else ww_val)
        return center, width
    return None

def _apply_windowing(
    volume: np.ndarray,
    meta: Dataset,
    mode: str,
    custom_window: Optional[Tuple[float, float]]
) -> np.ndarray:
    """Applies windowing (contrast/brightness adjustment) to the volume."""
    wc, ww = None, None
    mode = mode.lower()

    if mode == 'tags':
        wc, ww = _get_window_from_tags(meta)
        if wc is None:
            warnings.warn("windowing_mode='tags' but tags not found.", UserWarning)
    elif mode == 'custom':
        if custom_window and len(custom_window) == 2:
            wc, ww = custom_window
        else:
            raise ValueError("windowing_mode='custom' requires a valid `custom_window` tuple.")
    elif mode == 'fallback':
        wc, ww = _get_window_from_tags(meta)
        if wc is None and custom_window and len(custom_window) == 2:
            wc, ww = custom_window
    elif mode not in [None, 'none']:
        raise ValueError(f"Unknown windowing_mode: '{mode}'.")

    if wc is not None and ww is not None and ww > 0:
        img_min = wc - ww / 2
        img_max = wc + ww / 2
        return np.clip(volume, img_min, img_max)
    
    return volume

def _normalize_to_uint8(volume: np.ndarray) -> np.ndarray:
    """Normalizes a volume to the 0-255 range and converts to uint8."""
    min_val, max_val = np.min(volume), np.max(volume)
    if max_val > min_val:
        volume = (volume - min_val) / (max_val - min_val)
    else:
        volume = np.zeros_like(volume)
    return (volume * 255).astype(np.uint8)

# --- Main Public Function ---



def convert_columns_to_float_lists(df, columns):
    """
    Convert specified columns in df to lists of floats.
    - Handles both real lists of strings and stringified lists.
    - Ignores empty values (None, NaN, '', []).
    - Modifies df in place and also returns it.
    """
    def to_float_list(val):
        if val in (None, "", []):
            return val
        if isinstance(val, float) and np.isnan(val):
            return val
        if isinstance(val, list):
            return [float(i) for i in val]
        if isinstance(val, str):
            try:
                parsed = ast.literal_eval(val)  # safely parse
                return [float(i) for i in parsed]
            except Exception:
                return val  # leave as is if it can't be parsed
        return val  # leave other types untouched

    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(to_float_list)

    return df

def read_dicom_series(
    series_path: str,
    mode: str = 'prod',
    filter_invalid_slices: bool = True,
    sort_by: Optional[str] = 'fallback',
    process_by_modality: bool = False, 
    windowing_mode: Optional[str] = 'fallback',
    custom_window: Optional[Tuple[float, float]] = None,
    resize_to: Optional[Tuple[int, int]] = (512, 512),
    percentile_clip: Optional[Tuple[float, float]] = None,
    percentile_clip_sampling: Optional[int] = 2**20,
    to_uint8: bool = False,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Reads a DICOM series, processes it, and returns a 3D NumPy volume
    along with a DataFrame of corresponding DICOM tags.

    Args:
        series_path (str): Path to the folder containing the DICOM series.
        mode (str, optional): Execution mode ('prod' or 'dev'). Defaults to 'prod'.
        filter_invalid_slices (bool, optional): Filters out localizers, scouts, etc. Defaults to True.
        sort_by (Optional[str], optional): Sorting method ('fallback', 'ipp', 'instance_number', None).
        process_by_modality (bool, optional): If True, applies windowing for CT and percentile
            clipping for MR. Overrides default sequential processing. Defaults to False.
        windowing_mode (Optional[str], optional): Windowing method ('fallback', 'tags', 'custom', None).
        custom_window (Optional[Tuple[float, float]], optional): (window_center, window_width).
        resize_to (Optional[Tuple[int, int]], optional): Resizes each slice to (height, width).
        percentile_clip (Optional[Tuple[float, float]], optional): Clips intensities to (low_percentile, high_percentile).
        percentile_clip_sampling (Optional[int]): Number of pixels for percentile calculation.
        to_uint8 (bool, optional): If True, converts the final volume to uint8 (0-255).

    Returns:
        Tuple[np.ndarray, pd.DataFrame]: A tuple containing the 3D volume and a DataFrame of tags.
    """
    # 1. Validate initial parameters
    _validate_parameters(series_path, mode)

    # 2. Discover, read, and validate individual DICOM slices
    validated_slices = _load_and_validate_slices(series_path, mode, filter_invalid_slices)
    
    if not validated_slices:
        warnings.warn(f"No valid DICOM files could be processed in {series_path}", UserWarning)
        return np.array([]), pd.DataFrame()

    # 3. Sort the validated slices
    _sort_dicom_series(validated_slices, sort_by)
    
    # Separate datasets and tags post-sorting to maintain sync
    dicom_datasets, tags_data = zip(*validated_slices)
    df_tags = pd.DataFrame(tags_data)

    # 4. Create the 3D volume from pixel data
    volume = _create_volume_from_datasets(list(dicom_datasets), resize_to)

    # 5. Apply post-processing steps to the entire volume
    first_slice_meta = dicom_datasets[0]
    volume = _apply_rescale(volume, first_slice_meta)

    if process_by_modality:
        modality = first_slice_meta.get('Modality', '').upper()
        if modality == 'CT':
            if windowing_mode:
                volume = _apply_windowing(volume, first_slice_meta, windowing_mode, custom_window)
        elif modality == 'MR':
            if percentile_clip:
                volume = _apply_percentile_clip(volume, percentile_clip, percentile_clip_sampling)
        # For other modalities, no clipping or windowing is applied in this mode.
    else:
        # Original sequential processing logic
        if percentile_clip:
            volume = _apply_percentile_clip(volume, percentile_clip, percentile_clip_sampling)
        if windowing_mode:
            volume = _apply_windowing(volume, first_slice_meta, windowing_mode, custom_window)

    
    if to_uint8:
        volume = _normalize_to_uint8(volume)

    return volume, df_tags