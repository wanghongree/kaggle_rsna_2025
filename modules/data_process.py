# !pip install --upgrade \
#     pylibjpeg \
#     pylibjpeg-libjpeg \
#     pylibjpeg-openjpeg \
#     imagecodecs

# !pip install --upgrade "numpy<2.0"


def read_dicom_series(
    series_path: str,
    sort_by: Optional[str] = 'fallback',
    windowing_mode: Optional[str] = 'fallback',
    custom_window: Optional[Tuple[float, float]] = None,
    resize_to: Optional[Tuple[int, int]] = (512, 512),
    percentile_clip: Optional[Tuple[float, float]] = None,
    percentile_clip_sampling: Optional[int] = 2**20,
    to_uint8: bool = False,
    # --- NEW PARAMETER ---
    filter_invalid_slices: bool = True,
) -> np.ndarray:
    """
    Reads a grayscale DICOM series, processes it, and returns a 3D NumPy volume.

    Args:
        series_path (str): Path to the folder containing the DICOM series.
        sort_by (Optional[str], optional): Sorting method for slices. Options:
            - 'fallback' (default): Tries to sort by spatial position (IPP/IOP), falls back to InstanceNumber.
            - 'ipp': Strictly sorts by spatial position. Warns if tags are missing.
            - 'instance_number': Strictly sorts by InstanceNumber. Warns if tag is missing.
            - None: No sorting is performed.
        windowing_mode (Optional[str], optional): Controls how windowing is applied. Options:
            - 'fallback' (default): Prioritizes window parameters from DICOM tags. If not found, uses `custom_window`.
            - 'tags': Strictly uses window parameters from DICOM tags. Warns if not found.
            - 'custom': Strictly uses the window provided in `custom_window`. Raises error if not provided.
            - None: No windowing is applied.
        custom_window (Optional[Tuple[float, float]], optional): A tuple of (window_center, window_width)
            for use with `windowing_mode` 'custom' or 'fallback'. Defaults to None.
        resize_to (Optional[Tuple[int, int]], optional): If provided, resizes each slice to (height, width).
            Defaults to (512, 512). Set to None to disable resizing.
        percentile_clip (Optional[Tuple[float, float]], optional): If provided as (low, high), clips pixel
            intensities to the specified percentiles. Defaults to None.
        percentile_clip_sampling (Optional[int]): The number of pixels to sample for calculating percentiles,
            for efficiency on large volumes. Defaults to 2**20 (~1 million).
        to_uint8 (bool, optional): If True, converts the final volume to uint8 (0-255).
            Otherwise, returns float32 array (e.g., in Hounsfield Units). Defaults to False.
        # --- NEW DOCSTRING ---
        filter_invalid_slices (bool, optional): If True (default), filters out DICOM files that are not
            part of the main volume, such as localizers, scout images, or secondary captures. It also
            removes images with empty or constant (blank) pixel data.

    Returns:
        np.ndarray: A 3D NumPy array representing the DICOM volume (slices, height, width).
    """
    if not os.path.isdir(series_path):
        raise FileNotFoundError(f"Directory not found: {series_path}")

    # --- MODIFIED SECTION: File reading and filtering ---
    
    # Define reject types outside the loop for efficiency
    # REJECT_IMAGE_TYPES = {'LOCALIZER', 'SCOUT', 'SECONDARY', 'MPR'}
    REJECT_IMAGE_TYPES = {'SCOUT'}

    dicom_files: List[Dataset] = []
    for root, _, files in os.walk(series_path):
        for file in files:
            try:
                ds = pydicom.dcmread(os.path.join(root, file))
                
                # Basic check for pixel data, which is essential
                if not hasattr(ds, 'PixelData'):
                    print("get here!!")
                    continue
                
                # --- NEW FILTERING LOGIC ---
                if filter_invalid_slices:
                    # 1. Check ImageType for non-volumetric slices
                    image_type = set(ds.get('ImageType', []))
                    print(image_type)
                    if REJECT_IMAGE_TYPES.intersection(image_type):
                        continue # Skip this file as it's a type we want to reject
                        
                    # 2. Check for empty or constant pixel data
                    # This requires accessing the pixel array, which can decompress it.
                    pixel_array = ds.pixel_array
                    if pixel_array.min() == pixel_array.max():
                        continue # Skip this file as it's blank/constant
                
                dicom_files.append(ds)

            except InvalidDicomError:
                continue
            except Exception as e:
                # Catch potential errors from ds.pixel_array if data is corrupt
                warnings.warn(f"Skipping file {file} due to read error: {e}", UserWarning)
                continue
    # --- END OF MODIFIED SECTION ---

    if not dicom_files:
        raise ValueError(f"No valid DICOM files with pixel data found in {series_path}")

    # 2. Sort slices based on the specified mode
    if sort_by:
        sort_mode = sort_by.lower()
        
        def sort_by_ipp(files):
            # Check for required tags on the first file before proceeding
            if not all(hasattr(files[0], tag) for tag in ['ImageOrientationPatient', 'ImagePositionPatient']):
                 raise AttributeError("Missing IPP/IOP tags for sorting.")
            row_vec = np.array(files[0].ImageOrientationPatient[:3])
            col_vec = np.array(files[0].ImageOrientationPatient[3:])
            normal_vec = np.cross(row_vec, col_vec)
            files.sort(key=lambda ds: np.dot(np.array(ds.ImagePositionPatient), normal_vec))
            return True
        
        def sort_by_instance(files):
            if not hasattr(files[0], 'InstanceNumber'):
                raise AttributeError("Missing InstanceNumber tag for sorting.")
            files.sort(key=lambda ds: int(ds.InstanceNumber))
            return True

        if sort_mode == 'ipp':
            try:
                sort_by_ipp(dicom_files)
            except AttributeError:
                warnings.warn("sort_by='ipp' failed: ImagePositionPatient/ImageOrientationPatient tags missing.", UserWarning)
        elif sort_mode == 'instance_number':
            try:
                sort_by_instance(dicom_files)
            except AttributeError:
                warnings.warn("sort_by='instance_number' failed: InstanceNumber tag missing.", UserWarning)
        elif sort_mode == 'fallback':
            try:
                sort_by_ipp(dicom_files)
            except AttributeError:
                warnings.warn("Could not sort by ImagePositionPatient. Falling back to InstanceNumber.", UserWarning)
                try:
                    sort_by_instance(dicom_files)
                except AttributeError:
                    warnings.warn("Fallback sort by InstanceNumber also failed. Proceeding with unsorted slices.", UserWarning)
        else:
            raise ValueError(f"Unknown sort_by mode: '{sort_by}'. Options are 'ipp', 'instance_number', 'fallback', or None.")

    # 3. Extract, resize, and collect frames
    all_frames = []
    multiframe_count = 0
    
    target_dims = None
    if resize_to:
        if not (isinstance(resize_to, tuple) and len(resize_to) == 2):
            raise TypeError("resize_to must be a tuple of two integers (height, width).")
        target_dims = (resize_to[1], resize_to[0])

    for ds in dicom_files:
        is_color = ds.get('SamplesPerPixel') == 3 or ('PhotometricInterpretation' in ds and 'RGB' in ds.PhotometricInterpretation)
        if is_color:
            warnings.warn(f"Skipping a color DICOM image as they are not supported.", UserWarning)
            continue

        pixel_array = ds.pixel_array.astype(np.float32)
        
        frames_to_process = []
        if pixel_array.ndim == 2:
            frames_to_process.append(pixel_array)
        elif pixel_array.ndim == 3:
            multiframe_count += 1
            frames_to_process.extend(pixel_array[i] for i in range(pixel_array.shape[0]))
        else:
            warnings.warn(f"Skipping image with unsupported pixel array dimension: {pixel_array.ndim}", UserWarning)
            continue
            
        for frame in frames_to_process:
            if target_dims:
                frame = cv2.resize(frame, target_dims, interpolation=cv2.INTER_LINEAR)
            all_frames.append(frame)

    if multiframe_count > 1:
        warnings.warn(f"Series contains {multiframe_count} multi-frame DICOM files. All frames have been stacked together.", UserWarning)

    if not all_frames:
        raise ValueError("Could not extract any pixel frames from the filtered DICOM series.")

    # 4. Stack all collected frames into a volume
    volume = np.stack(all_frames, axis=0)

    # 5. Apply Rescale Slope and Intercept (HU conversion)
    first_slice_meta = dicom_files[0]
    if 'RescaleSlope' in first_slice_meta and 'RescaleIntercept' in first_slice_meta:
        slope = float(first_slice_meta.RescaleSlope)
        intercept = float(first_slice_meta.RescaleIntercept)
        volume = volume * slope + intercept

    # 6. Apply percentile clipping with efficient sampling
    if percentile_clip:
        if not (isinstance(percentile_clip, tuple) and len(percentile_clip) == 2):
            raise TypeError("percentile_clip must be a tuple of two floats (low, high).")
        low, high = percentile_clip
        if percentile_clip_sampling and volume.size > percentile_clip_sampling:
            flat_volume = volume.ravel()
            indices = np.random.choice(flat_volume.size, size=percentile_clip_sampling, replace=False)
            sample = flat_volume[indices]
            p_low, p_high = np.percentile(sample, [low, high])
        else:
            p_low, p_high = np.percentile(volume, [low, high])
        volume = np.clip(volume, p_low, p_high)

    # 7. Apply windowing based on the specified mode
    if windowing_mode:
        window_center, window_width = None, None
        mode = windowing_mode.lower()
        
        def get_window_from_tags(ds_meta):
            if 'WindowCenter' in ds_meta and 'WindowWidth' in ds_meta:
                wc_val = ds_meta.WindowCenter
                ww_val = ds_meta.WindowWidth
                center = float(wc_val[0]) if isinstance(wc_val, pydicom.multival.MultiValue) else float(wc_val)
                width = float(ww_val[0]) if isinstance(ww_val, pydicom.multival.MultiValue) else float(ww_val)
                return center, width
            return None, None

        if mode == 'tags':
            window_center, window_width = get_window_from_tags(first_slice_meta)
            if window_center is None:
                warnings.warn("windowing_mode='tags' but tags not found. Skipping windowing.", UserWarning)
        elif mode == 'custom':
            if custom_window and isinstance(custom_window, tuple) and len(custom_window) == 2:
                window_center, window_width = custom_window
            else:
                raise ValueError("windowing_mode='custom' requires a valid `custom_window` tuple of (center, width).")
        elif mode == 'fallback':
            window_center, window_width = get_window_from_tags(first_slice_meta)
            if window_center is None and custom_window:
                if isinstance(custom_window, tuple) and len(custom_window) == 2:
                    window_center, window_width = custom_window
                else:
                    warnings.warn("DICOM tags for windowing not found, and provided `custom_window` is invalid. Skipping windowing.", UserWarning)
        elif mode is not None:
             raise ValueError(f"Unknown windowing_mode: '{windowing_mode}'. Options are 'tags', 'custom', 'fallback', or None.")

        if window_center is not None and window_width is not None:
            img_min = window_center - window_width / 2
            img_max = window_center + window_width / 2
            volume = np.clip(volume, img_min, img_max)

    # 8. Convert to uint8 if requested
    if to_uint8:
        min_val, max_val = np.min(volume), np.max(volume)
        if max_val > min_val:
            volume = (volume - min_val) / (max_val - min_val)
        else:
            volume = np.zeros_like(volume)
        volume = (volume * 255).astype(np.uint8)

    return volume