# src/data/dataset.py

import torch
import torchvision
from torch.utils.data import Dataset
from PIL import Image
import os
import sys
import pickle
import hashlib

from hest import iter_hest # Still needed potentially for metadata or other info if desired
    print("Imported iter_hest from HEST library.")
except ImportError:
    print("Warning: HEST library not found. Some fallback logic might be affected if metadata CSV is not used.")
    iter_hest = None
# Import the helper function from the same data module
from .utils import _get_sample_label_from_metadata

# Added h5py for loading patch data
try:
    import h5py
    print("Imported h5py.")
except ImportError:
    print("Error: h5py library not found. Please install it ('pip install h5py').")
    h5py = None

class HistoPatchDataset(Dataset):
    """Loads HEST histopathology patch data from .h5 files."""
    def __init__(self, data_dir, id_list, img_size=256, augment=False,
                 global_label_map=None, global_num_classes=None,
                 label_source_column=None, loaded_metadata=None,
                 patch_dir_name='patches', patch_file_suffix='_patches.h5',
                 h5_image_key='data', cache_dir=None):
        if h5py is None: raise ImportError("h5py required for this dataset.")
        if global_label_map is None or global_num_classes is None:
             raise ValueError("Requires global_label_map and global_num_classes.")

        self.data_dir = data_dir
        self.id_list = sorted(list(id_list)) if id_list else []
        self.img_size = img_size
        self.augment = augment
        self.patch_dir_name = patch_dir_name
        self.patch_file_suffix = patch_file_suffix
        self.h5_image_key = h5_image_key
        self.cache_dir = cache_dir

        self.label_map = global_label_map
        self.num_classes = global_num_classes
        self.unknown_label_idx = self.label_map.get('unknown', 0)
        self.metadata = loaded_metadata
        self.label_source_column = label_source_column
        self.load_errors = 0

        print(f"Initializing Dataset for {len(self.id_list)} IDs using global map.")
        if not self.id_list: print("Warning: HistoPatchDataset initialized with empty id_list.")

        self.base_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((self.img_size, self.img_size), antialias=True),
            torchvision.transforms.ToTensor()
        ])
        self.aug_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.08, hue=0.04)
        ])

        self.data = self._load_or_build_manifest()
        print(f"Dataset initialized with {len(self.data)} patches.")
        if not self.data and self.id_list:
            print("\nWARNING: No patch data indexed. Check data paths and patch file contents.")

    def _get_cache_key(self):
        hasher = hashlib.md5()
        hasher.update(self.data_dir.encode())
        hasher.update("||".join(self.id_list).encode())
        hasher.update(self.patch_dir_name.encode())
        hasher.update(self.patch_file_suffix.encode())
        hasher.update(self.h5_image_key.encode())
        return hasher.hexdigest()

    def _load_or_build_manifest(self):
        manifest = []
        cache_valid = False
        cache_path = None
        if self.cache_dir and self.id_list:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_key = self._get_cache_key()
            cache_path = os.path.join(self.cache_dir, f"manifest_{cache_key}.pkl")
            print(f"Checking for manifest cache: {cache_path}")
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'rb') as f: manifest = pickle.load(f)
                    print(f"Loaded manifest from cache ({len(manifest)} entries).")
                    cache_valid = True
                except Exception as e:
                    print(f"Warning: Could not load cache file '{cache_path}': {e}. Rebuilding.")
        if not cache_valid:
            print("Building patch manifest...")
            manifest = self._build_manifest_from_files()
            if manifest and cache_path:
                try:
                    print(f"Saving manifest cache to: {cache_path}")
                    with open(cache_path, 'wb') as f: pickle.dump(manifest, f)
                except Exception as e:
                    print(f"Warning: Could not save manifest cache '{cache_path}': {e}")
        return manifest

    def _build_manifest_from_files(self):
        manifest = []
        print(f"  Scanning for patch files for {len(self.id_list)} assigned IDs...")
        loaded, skipped, total_patches = 0, 0, 0
        for sample_id in self.id_list:
            path = os.path.join(self.data_dir, sample_id, self.patch_dir_name, f"{sample_id}{self.patch_file_suffix}")
            if not os.path.exists(path):
                path_alt = os.path.join(self.data_dir, self.patch_dir_name, f"{sample_id}{self.patch_file_suffix}")
                if os.path.exists(path_alt): path = path_alt
                else: skipped += 1; continue
            try:
                label_str = _get_sample_label_from_metadata(self.metadata, self.label_source_column)
                with h5py.File(path, 'r') as f:
                    if self.h5_image_key not in f: skipped += 1; continue
                    num_patches = f[self.h5_image_key].shape[0]
                if num_patches > 0:
                    for i in range(num_patches): manifest.append((path, i, label_str))
                    total_patches += num_patches
                    loaded += 1
                else: skipped += 1
            except Exception as e: print(f"  Error processing {path}: {e}"); skipped += 1
        print(f"  Manifest build complete. Loaded: {loaded}, Skipped: {skipped}, Total Patches: {total_patches}")
        return manifest

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, patch_idx, label_str = self.data[idx]
        try:
            with h5py.File(path, 'r') as f:
                patch_np = f[self.h5_image_key][patch_idx]
            img = Image.fromarray(patch_np)
            tensor = self.base_transform(img)
            if self.augment: tensor = self.aug_transform(tensor)
            label = torch.tensor(self.label_map.get(label_str, self.unknown_label_idx)).long()
            return tensor, label
        except Exception as e:
            self.load_errors += 1
            print(f"\nERROR loading patch {path}, index {patch_idx}: {e}", file=sys.stderr)
            return torch.zeros((3, self.img_size, self.img_size)), torch.tensor(self.unknown_label_idx).long()

    def get_labels(self): return [item[2] for item in self.data]
    def get_load_error_count(self): return self.load_errors
    def reset_load_error_count(self): self.load_errors = 0