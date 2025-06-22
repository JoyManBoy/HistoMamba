# src/data/utils.py

import os
import pandas as pd

def _get_sample_label_from_metadata(sample_id, metadata_df, label_source_column):
    """Helper to extract a single sample's label from the metadata."""
    label_str = 'unknown'
    if metadata_df is not None and label_source_column is not None:
        if sample_id in metadata_df.index:
            meta_label = metadata_df.loc[sample_id, label_source_column]
            if pd.notna(meta_label):
                label_str = str(meta_label).lower().strip()
    if label_str == 'nan': label_str = 'unknown'
    return label_str

def build_global_label_map(all_sample_ids, metadata_csv):
    """Builds a global map from label string to integer index."""
    print("Building global label map...")
    found_labels = set()
    metadata_df = None
    label_source_column = None
    if metadata_csv and os.path.exists(metadata_csv):
        try:
            print(f"Loading metadata for global map from: {metadata_csv}")
            meta_df_temp = pd.read_csv(metadata_csv)
            if 'id' in meta_df_temp.columns:
                metadata_df = meta_df_temp.set_index('id')
                print(f"Global metadata loaded for {len(metadata_df)} IDs.")
                preferred_cols = ['tissue_type', 'oncotree_code', 'organ']
                for col in preferred_cols:
                    if col in metadata_df.columns:
                        label_source_column = col
                        break
                if label_source_column:
                    print(f"Using '{label_source_column}' from metadata for global labels.")
                else:
                    print("Warning: Metadata loaded, but no suitable label column found in:", preferred_cols)
            else:
                print("Warning: Metadata CSV lacks 'id' column.")
                metadata_df = None
        except Exception as e:
            print(f"Warning: Could not load metadata CSV '{metadata_csv}': {e}")
            metadata_df = None
    elif metadata_csv:
         print(f"Warning: Metadata CSV not found: '{metadata_csv}'.")
    
    if not all_sample_ids:
        all_sample_ids = []
    
    for sample_id in all_sample_ids:
        label_str = _get_sample_label_from_metadata(sample_id, metadata_df, label_source_column)
        found_labels.add(label_str)
    
    sorted_labels = sorted(list(found_labels))
    label_map = {}
    if not sorted_labels or (len(sorted_labels) == 1 and 'unknown' in sorted_labels):
        print("\nWARNING: No meaningful labels found. Using a single 'unknown' class.")
        label_map = {'unknown': 0}
    else:
        if 'unknown' in sorted_labels:
            sorted_labels.remove('unknown')
            sorted_labels.append('unknown') # Ensure 'unknown' is last
        else:
            if sorted_labels: sorted_labels.append('unknown')
            else: sorted_labels = ['unknown']
        label_map = {label: i for i, label in enumerate(sorted_labels)}
    
    num_classes = len(label_map)
    print(f"Global label map created with {num_classes} classes: {list(label_map.keys())}")
    print(f"Global map: {label_map}")
    return label_map, label_source_column, metadata_df