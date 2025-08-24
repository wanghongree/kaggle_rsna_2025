import os
import warnings
from typing import Tuple, Optional, List, Dict, Any
from collections import defaultdict
from types import SimpleNamespace # Import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import Dataset
from pydicom.multival import MultiValue
from pydicom.errors import InvalidDicomError

# --- Constants (Unchanged) ---
TAGS_TO_EXTRACT = [
    'BitsAllocated', 'BitsStored', 'Columns', 'FrameOfReferenceUID', 'HighBit',
    'ImageOrientationPatient', 'ImagePositionPatient', 'InstanceNumber', 'Modality',
    'PatientID', 'PhotometricInterpretation', 'PixelRepresentation', 'PixelSpacing',
    'PlanarConfiguration', 'RescaleIntercept', 'RescaleSlope', 'RescaleType',
    'Rows', 'SOPClassUID', 'SOPInstanceUID', 'SamplesPerPixel', 'SeriesDescription',
    'SliceThickness', 'SpacingBetweenSlices', 'StudyInstanceUID', 'TransferSyntaxUID', 'ImageType'
]
SCOUT_KEYWORDS = {'SCOUT', 'LOCALIZER', 'TOPOGRAM', 'SCANOGRAM'}
REJECT_IMAGE_TYPES = {'LOCALIZER', 'SECONDARY', 'MPR'}

# --- Helper Functions for Loading and Validation ---

def _validate_parameters(series_path: str, mode: str):
    """Validates the initial input parameters."""
    if not os.path.isdir(series_path):
        raise FileNotFoundError(f"Directory not found: {series_path}")
    if mode.lower() not in ['prod', 'dev']:
        raise ValueError(f"Unknown mode: '{mode}'. Options are 'prod' or 'dev'.")

def _is_slice_valid(ds: Dataset, pixel_array: np.ndarray, filter_invalid: bool) -> bool:
    """
    Checks if a single DICOM slice is fundamentally valid for inclusion.
    Now takes pixel_array directly to support unpacked frames.
    """
    if ds.get('SamplesPerPixel') == 3 or 'RGB' in ds.get('PhotometricInterpretation', ''):
        raise ValueError("Color DICOM images are not supported.")

    if filter_invalid:
        if pixel_array.min() == pixel_array.max():
            raise ValueError("Filtered as blank/constant pixel data.")
            
    return True

# --- NEW: Function to unpack both classic and multi-frame DICOMs ---

def _unpack_frames(
    series_path: str, mode: str, filter_invalid: bool
) -> List[Tuple[Any, Dict[str, Any]]]:
    """
    Walks a directory, reads DICOM files, and unpacks them into a list of
    individual frames. Handles both classic (one file/frame) and Enhanced
    multi-frame (one file/many frames) DICOMs.

    Returns:
        A list of tuples, where each tuple contains:
        - A proxy object (or original dataset) for a single frame.
        - A dictionary of tags for that frame.
    """
    unpacked_frames = []
    run_mode = mode.lower()

    for root, _, files in os.walk(series_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                ds = pydicom.dcmread(file_path)

                # Determine if the file is a multi-frame Enhanced DICOM
                is_multi_frame = ds.get('NumberOfFrames', 1) > 1 and \
                                 hasattr(ds, 'PerFrameFunctionalGroupsSequence')
                
                if is_multi_frame:
                    # --- Handle Enhanced Multi-frame DICOM ---
                    num_frames = ds.NumberOfFrames
                    per_frame_groups = ds.PerFrameFunctionalGroupsSequence
                    shared_groups = ds.get('SharedFunctionalGroupsSequence', [{}])[0]

                    # Check for potential mismatch in metadata length
                    if len(per_frame_groups) != num_frames:
                        warnings.warn(
                            f"File '{file}' has {num_frames} frames but {len(per_frame_groups)} "
                            f"PerFrameFunctionalGroups. Metadata may be incomplete.", UserWarning
                        )

                    # Extract shared tags once
                    shared_iop = shared_groups.get('PlaneOrientationSequence', [{}])[0].get('ImageOrientationPatient', None)
                    pixel_measures = shared_groups.get('PixelMeasuresSequence', [{}])[0]
                    shared_pixel_spacing = pixel_measures.get('PixelSpacing', None)
                    shared_slice_thickness = pixel_measures.get('SliceThickness', None)
                    
                    full_pixel_array = ds.pixel_array

                    for i in range(num_frames):
                        frame_pixel_array = full_pixel_array[i]
                        
                        if not _is_slice_valid(ds, frame_pixel_array, filter_invalid):
                            continue
                            
                        # Create a proxy object to hold frame-specific data
                        frame_proxy = SimpleNamespace()
                        frame_proxy.pixel_array = frame_pixel_array
                        
                        # Copy necessary attributes for post-processing and sorting
                        for attr in ['RescaleSlope', 'RescaleIntercept', 'WindowCenter', 'WindowWidth', 'Modality']:
                            setattr(frame_proxy, attr, ds.get(attr, None))
                        
                        # Extract per-frame tags
                        frame_tags = _extract_tags(ds, file, is_multi_frame=True)
                        frame_tags['InstanceNumber'] = i + 1 # Use frame index as instance number
                        frame_proxy.InstanceNumber = frame_tags['InstanceNumber']

                        # Override shared tags with per-frame info if available
                        frame_group = per_frame_groups[i] if i < len(per_frame_groups) else {}
                        
                        # IPP (Image Position Patient) from Per-Frame Sequence
                        ipp = frame_group.get('PlanePositionSequence', [{}])[0].get('ImagePositionPatient', None)
                        frame_tags['ImagePositionPatient'] = [float(v) for v in ipp] if ipp else None
                        frame_proxy.ImagePositionPatient = frame_tags['ImagePositionPatient']
                        
                        # IOP (Image Orientation Patient) from Shared Sequence
                        frame_tags['ImageOrientationPatient'] = [float(v) for v in shared_iop] if shared_iop else None
                        frame_proxy.ImageOrientationPatient = frame_tags['ImageOrientationPatient']

                        # Other useful tags might be in shared sequence
                        frame_tags['PixelSpacing'] = [float(v) for v in shared_pixel_spacing] if shared_pixel_spacing else frame_tags.get('PixelSpacing')
                        frame_tags['SliceThickness'] = float(shared_slice_thickness) if shared_slice_thickness else frame_tags.get('SliceThickness')

                        unpacked_frames.append((frame_proxy, frame_tags))
                
                else:
                    # --- Handle Classic Single-frame DICOM ---
                    if not _is_slice_valid(ds, ds.pixel_array, filter_invalid):
                        continue
                    tags = _extract_tags(ds, file)
                    unpacked_frames.append((ds, tags))

            except (InvalidDicomError, AttributeError, Exception) as e:
                if run_mode == 'prod':
                    warnings.warn(f"Skipping file '{file}' due to error: {e}", UserWarning)
                    continue
                else:
                    raise type(e)(f"Error processing file '{file_path}': {e}") from e
    
    return unpacked_frames

# --- Modified Helper Functions ---

def _extract_tags(ds: Dataset, filename: str, is_multi_frame: bool = False) -> Dict[str, Any]:
    """
    Extracts a predefined list of tags from a DICOM dataset.
    For multi-frame, it avoids extracting frame-specific tags that will be
    overridden later.
    """
    slice_tags = {'FileName': filename}

    def try_float(val):
        try:
            return float(val)
        except (ValueError, TypeError):
            return val

    for tag in TAGS_TO_EXTRACT:
        # For multi-frame, these will be extracted from functional groups, so we skip them here
        if is_multi_frame and tag in ['ImageOrientationPatient', 'ImagePositionPatient', 
                                      'PixelSpacing', 'SliceThickness']:
            slice_tags[tag] = None
            continue

        value = ds.get(tag, None)

        if isinstance(value, MultiValue):
            slice_tags[tag] = list(value) if tag == 'ImageType' else [try_float(v) for v in value]
        else:
            slice_tags[tag] = try_float(value)

    return slice_tags

# --- Scout Filtering Function (Unchanged from previous version) ---
def _filter_scout_images(
    all_slices: List[Tuple[Any, Dict[str, Any]]]
) -> List[Tuple[Any, Dict[str, Any]]]:
    """
    Applies a multi-layered strategy to filter out scout/localizer images.
    Now accepts proxy objects or pydicom datasets.
    """
    if len(all_slices) <= 2:
        return all_slices

    indices_to_keep = []
    preliminary_filtered_slices = []

    for i, (item, tags) in enumerate(all_slices):
        image_type = set(tags.get('ImageType', []))
        series_desc = tags.get('SeriesDescription', '').upper()
        
        if REJECT_IMAGE_TYPES.intersection(image_type):
            continue
        if any(keyword in series_desc for keyword in SCOUT_KEYWORDS):
            continue
            
        indices_to_keep.append(i)
        preliminary_filtered_slices.append((item, tags))

    if len(preliminary_filtered_slices) <= 2:
        return preliminary_filtered_slices

    orientation_groups = defaultdict(list)
    for i, (item, tags) in enumerate(preliminary_filtered_slices):
        iop = tags.get('ImageOrientationPatient')
        if iop:
            iop_tuple = tuple(np.round(iop, 4))
            orientation_groups[iop_tuple].append(i)

    if not orientation_groups:
        return preliminary_filtered_slices

    main_iop_group_indices = max(orientation_groups.values(), key=len)
    final_indices_to_keep = set(main_iop_group_indices)
    
    # Spatial outlier check
    main_stack_items = [preliminary_filtered_slices[i] for i in main_iop_group_indices if preliminary_filtered_slices[i][1].get('ImagePositionPatient')]
    
    if len(main_stack_items) > 1:
        main_iop = main_stack_items[0][1]['ImageOrientationPatient']
        row_vec, col_vec = np.array(main_iop[:3]), np.array(main_iop[3:])
        normal_vec = np.cross(row_vec, col_vec)

        distances = [np.dot(normal_vec, item[1]['ImagePositionPatient']) for item in main_stack_items]
        distances.sort()
        min_dist, max_dist = distances[0], distances[-1]
        
        spacings = np.diff(distances)
        avg_spacing = np.median(spacings) if len(spacings) > 0 else 0
        threshold = max(avg_spacing * 5, 10.0)

        outlier_indices = set()
        for i in main_iop_group_indices:
            ipp = preliminary_filtered_slices[i][1].get('ImagePositionPatient')
            if ipp:
                dist = np.dot(normal_vec, ipp)
                if not (min_dist - threshold <= dist <= max_dist + threshold):
                    outlier_indices.add(i)
        
        final_indices_to_keep -= outlier_indices

    return [preliminary_filtered_slices[i] for i in sorted(list(final_indices_to_keep))]


# --- Sorting Helpers (Updated to handle proxy objects) ---

def _sort_by_ipp(items: List[Tuple[Any, Dict]]):
    """Sorts a list of items by spatial position (IPP)."""
    # Use the first item that has complete geometric info
    ref_item = next((item for item in items if item[0].ImageOrientationPatient and item[0].ImagePositionPatient), None)
    if not ref_item: raise AttributeError("No items with complete IPP/IOP found for sorting.")
    
    row_vec = np.array(ref_item[0].ImageOrientationPatient[:3])
    col_vec = np.array(ref_item[0].ImageOrientationPatient[3:])
    normal_vec = np.cross(row_vec, col_vec)
    items.sort(key=lambda item: np.dot(np.array(item[0].ImagePositionPatient), normal_vec))

def _sort_by_instance_number(items: List[Tuple[Any, Dict]]):
    """Sorts a list of items by InstanceNumber."""
    items.sort(key=lambda item: int(item[0].InstanceNumber))
    
def _sort_dicom_series(validated_slices: List[Tuple[Any, Dict]], sort_by: Optional[str]):
    """Orchestrates sorting. Now handles proxy objects."""
    if not sort_by or len(validated_slices) < 2:
        return
    # The rest of the logic is the same...
    sort_mode = sort_by.lower()
    try:
        if sort_mode == 'ipp':
            _sort_by_ipp(validated_slices)
        elif sort_mode == 'instance_number':
            _sort_by_instance_number(validated_slices)
        elif sort_mode == 'fallback':
            try:
                _sort_by_ipp(validated_slices)
            except (AttributeError, IndexError, TypeError):
                warnings.warn("Could not sort by IPP/IOP. Falling back to InstanceNumber.", UserWarning)
                _sort_by_instance_number(validated_slices)
        else:
            raise ValueError(f"Unknown sort_by mode: '{sort_by}'.")
    except Exception as e:
        warnings.warn(f"Sorting failed with error: '{e}'. Proceeding with unsorted slices.", UserWarning)

# --- Volume Creation & Post-processing (Updated to handle proxy objects) ---

def _create_volume_from_datasets(
    items: List[Any], resize_to: Optional[Tuple[int, int]]
) -> np.ndarray:
    """Extracts pixel data, resizes, and stacks them into a 3D volume."""
    all_frames = []
    target_dims = (resize_to[1], resize_to[0]) if resize_to else None

    for item in items:
        # This now works for both pydicom datasets and our proxy objects
        pixel_array = item.pixel_array.astype(np.float32)
        
        # The pixel_array from our proxy is always 2D, simplifying this
        if target_dims:
            frame = cv2.resize(pixel_array, target_dims, interpolation=cv2.INTER_LINEAR)
        else:
            frame = pixel_array
        all_frames.append(frame)

    return np.stack(all_frames, axis=0)

# Other post-processing functions (_apply_rescale, etc.) are compatible as they
# use the proxy object's attributes (e.g., meta.get('RescaleSlope')) or direct
# attribute access which we have populated on the proxy.
# (The functions are omitted here for brevity but should be kept in your file)
def _apply_rescale(volume: np.ndarray, meta: Any) -> np.ndarray:
    slope = float(getattr(meta, 'RescaleSlope', 1.0) or 1.0)
    intercept = float(getattr(meta, 'RescaleIntercept', 0.0) or 0.0)
    if slope != 1.0 or intercept != 0.0:
        return volume * slope + intercept
    return volume

def _get_window_from_tags(meta: Any) -> Optional[Tuple[float, float]]:
    wc_val = getattr(meta, 'WindowCenter', None)
    ww_val = getattr(meta, 'WindowWidth', None)
    if wc_val is not None and ww_val is not None:
        center = float(wc_val[0] if isinstance(wc_val, MultiValue) else wc_val)
        width = float(ww_val[0] if isinstance(ww_val, MultiValue) else ww_val)
        return center, width
    return None

def _apply_windowing(volume: np.ndarray, meta: Any, mode: str, custom_window: Optional[Tuple[float, float]]) -> np.ndarray:
    # This function should work correctly as long as _get_window_from_tags is adapted.
    # The logic inside remains the same.
    # (Omitted for brevity)
    wc, ww = None, None
    mode = mode.lower()
    if mode == 'tags':
        wc, ww = _get_window_from_tags(meta)
        if wc is None: warnings.warn("windowing_mode='tags' but tags not found.", UserWarning)
    elif mode == 'custom':
        if custom_window and len(custom_window) == 2: wc, ww = custom_window
        else: raise ValueError("windowing_mode='custom' requires a valid `custom_window` tuple.")
    elif mode == 'fallback':
        wc, ww = _get_window_from_tags(meta)
        if wc is None and custom_window and len(custom_window) == 2: wc, ww = custom_window
    elif mode not in [None, 'none']:
        raise ValueError(f"Unknown windowing_mode: '{mode}'.")
    if wc is not None and ww is not None and ww > 0:
        img_min = wc - ww / 2
        img_max = wc + ww / 2
        return np.clip(volume, img_min, img_max)
    return volume
# _apply_percentile_clip and _normalize_to_uint8 do not depend on metadata and are unchanged.
# (Omitted for brevity)
def _apply_percentile_clip(volume: np.ndarray, clip_range: Tuple[float, float], sampling_size: Optional[int]) -> np.ndarray:
    low_p, high_p = clip_range
    if sampling_size and volume.size > sampling_size:
        indices = np.random.choice(volume.size, size=sampling_size, replace=False)
        p_low, p_high = np.percentile(volume.ravel()[indices], [low_p, high_p])
    else:
        p_low, p_high = np.percentile(volume, [low_p, high_p])
    return np.clip(volume, p_low, p_high)
def _normalize_to_uint8(volume: np.ndarray) -> np.ndarray:
    min_val, max_val = np.min(volume), np.max(volume)
    if max_val > min_val: volume = (volume - min_val) / (max_val - min_val)
    else: volume = np.zeros_like(volume)
    return (volume * 255).astype(np.uint8)

# --- Main Public Function (Updated Workflow) ---

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
    (Docstring remains the same)
    """
    # 1. Validate initial parameters
    _validate_parameters(series_path, mode)

    # 2. Discover, read, and UNPACK all frames from classic and multi-frame DICOMs
    unpacked_frames = _unpack_frames(series_path, mode, filter_invalid_slices)
    
    # 2a. Apply advanced filtering for scouts/localizers
    if filter_invalid_slices and unpacked_frames:
        filtered_frames = _filter_scout_images(unpacked_frames)
    else:
        filtered_frames = unpacked_frames
    
    if not filtered_frames:
        warnings.warn(f"No valid DICOM frames could be processed in {series_path} after filtering.", UserWarning)
        return np.array([]), pd.DataFrame()

    # 3. Sort the final list of frames
    _sort_dicom_series(filtered_frames, sort_by)
    
    # Separate proxy objects/datasets and tags post-sorting to maintain sync
    items, tags_data = zip(*filtered_frames)
    df_tags = pd.DataFrame(list(tags_data))
    
    # 4. Create the 3D volume from pixel data
    volume = _create_volume_from_datasets(list(items), resize_to)

    # 5. Apply post-processing steps to the entire volume
    first_item_meta = items[0]
    volume = _apply_rescale(volume, first_item_meta)

    # (The rest of the post-processing logic is unchanged)
    modality = getattr(first_item_meta, 'Modality', '').upper()
    if process_by_modality:
        if modality == 'CT':
            if windowing_mode:
                volume = _apply_windowing(volume, first_item_meta, windowing_mode, custom_window)
        elif modality == 'MR':
            if percentile_clip:
                volume = _apply_percentile_clip(volume, percentile_clip, percentile_clip_sampling)
    else:
        if percentile_clip:
            volume = _apply_percentile_clip(volume, percentile_clip, percentile_clip_sampling)
        if windowing_mode:
            volume = _apply_windowing(volume, first_item_meta, windowing_mode, custom_window)

    if to_uint8:
        volume = _normalize_to_uint8(volume)

    return volume, df_tags