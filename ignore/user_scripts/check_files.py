# %%
import pandas as pd
import os

# %% read pixel shape
series_id = "1.2.826.0.1.3680043.8.498.75712554178574230484227682423862727306" 
data_path = "../data/"
series_path = os.path.join(data_path, "series/", series_id)
dcm_instance_name = "1.2.826.0.1.3680043.8.498.13275984078749145470856597218259380669.dcm"
dcm_instance_path = os.path.join(series_path, dcm_instance_name)
#dcm_instance_path = "../data/series/1.2.826.0.1.3680043.8.498.10004044428023505108375152878107656647/1.2.826.0.1.3680043.8.498.10124807242473374136099471315028464450.dcm"


dcm_instance_path

# %% check nii
seg_list = os.listdir("/home/hongrui/work/kaggle_rsna_2025/data/segmentations")

dcm_instance_name in seg_list




# %%
import pydicom

# Load the DICOM file
ds = pydicom.dcmread(dcm_instance_path)

# Iterate over all elements and print their tag, name, and value
for elem in ds.iterall():
    print(f"{elem.tag} : {elem.keyword} : {elem.value}")

# %%
import pydicom
import numpy as np
from pydicom.pixel_data_handlers.util import apply_voi_lut
import matplotlib.pyplot as plt



ds = pydicom.dcmread(dcm_instance_path)

arr = ds.pixel_array

if "BitsStored" in ds:
    mask = (1 << int(ds.BitsStored)) - 1
    arr = arr.astype(np.uint16) & mask

try:
    arr_disp = apply_voi_lut(arr, ds)
except Exception:
    arr_disp = arr

if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
    arr_disp = np.max(arr_disp) - arr_disp

# Normalize to 0–255 (NumPy 2.0 fix: use np.ptp)
arr_norm = arr_disp.astype(np.float32)
arr_norm -= arr_norm.min()
ptp_val = np.ptp(arr_norm) or 1.0
arr_norm = (arr_norm / ptp_val) * 255.0
arr_norm = arr_norm.astype(np.uint8)

if arr_norm.ndim == 3:
    img = arr_norm[0]  # first frame
else:
    img = arr_norm

plt.imshow(img, cmap="gray")
plt.axis("off")
plt.title("DICOM image")
plt.show()

# %%
