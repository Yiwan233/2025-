# -*- coding: utf-8 -*-
"""
Math Modeling - Forensic DNA Analysis - Problem 1: Number of Contributors

Version: 1.8 (Corrected Regex for NoC Extraction)
Date: 2025-05-04
"""
# ... (Imports, Warnings, Config remain the same) ...
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

DATA_DIR = './'
file_path_prob1 = os.path.join(DATA_DIR, '附件1：不同人数的STR图谱数据.csv')
# feature_filename = os.path.join(DATA_DIR, 'prob1_features_v1.8_wide.csv')

# --- Function Definition First ---
def extract_true_noc(filename):
    """ Extracts NoC using a specific regex targeting '-IDs-Ratio-' pattern """
    filename = str(filename)
    # Use the corrected regex that requires the ratio part after the IDs
    match = re.search(r'-(\d+(?:_\d+)*)-[\d;]+-', filename)
    if match:
        contributor_ids_str = match.group(1)
        # DEBUG: (Optional) Keep debug prints if needed
        # print(f"DEBUG (func): Processing '{filename}'")
        # print(f"DEBUG (func):   Regex '-(\\d+(?:_\\d+)*)-[\\d;]+-' matched! Captured IDs: '{contributor_ids_str}'")
        ids_list = contributor_ids_str.split('_')
        ids_list = [id_val for id_val in ids_list if id_val and id_val.isdigit()] # Filter empty/non-digit parts
        num_contributors = len(ids_list)
        # print(f"DEBUG (func):   Split IDs (digits only): {ids_list}")
        # print(f"DEBUG (func):   Calculated NoC: {num_contributors}")
        if num_contributors > 0:
             return int(num_contributors)
        else:
             return np.nan # Should not happen if regex matches digits
    else:
        # DEBUG: (Optional) Keep debug prints if needed
        # print(f"DEBUG (func): Processing '{filename}'")
        # print(f"DEBUG (func):   Regex '-(\\d+(?:_\\d+)*)-[\\d;]+-' pattern not found.")
        return np.nan

# --- Step 1: Load Data ... ---
# ... (Loading code as before) ...
df_prob1 = None
load_successful = False
try:
    print(f"Attempting to load with encoding: 'utf-8' and delimiter: ','...")
    df_prob1 = pd.read_csv(
        file_path_prob1,
        encoding='utf-8',
        sep=',',
        on_bad_lines='skip'
    )
    print(f"Successfully loaded '{file_path_prob1}' using encoding 'utf-8' and delimiter ','.")
    print(f"Note: Rows with incorrect field counts might have been skipped.")
    load_successful = True
except FileNotFoundError:
    print(f"Error: File not found at '{file_path_prob1}'.")
    load_successful = False
except Exception as e:
    print(f"An unexpected error occurred during loading: {e}")
    load_successful = False

# --- Proceed only if loading was successful ---
if load_successful and df_prob1 is not None:
    if df_prob1.empty:
        print("\nWarning: Data loaded, but the DataFrame is empty. Exiting.")
        exit()

    print(f"\nData loaded. Dataset shape: {df_prob1.shape}")

    # --- Step 1 Cont.: NoC Extraction ---
    print("\nAttempting NoC extraction with corrected regex...")
    try:
        if 'Sample File' not in df_prob1.columns:
             print(f"\nError: Essential column 'Sample File' not found.")
             exit()

        # Create the NoC map using the corrected function
        print("\nDEBUG: Creating NoC map...")
        unique_files = df_prob1['Sample File'].dropna().unique()
        noc_map = {filename: extract_true_noc(filename) for filename in unique_files}

        # DEBUG: Print first few items of the created map
        print("\nDEBUG: First 10 items in created noc_map:")
        map_items = list(noc_map.items())
        for i, item in enumerate(map_items[:10]):
             print(f"  {i}: {item}")
        print(f"DEBUG: Total items in noc_map: {len(noc_map)}")


        # Apply the map to the DataFrame column
        df_prob1['NoC_True_Mapped'] = df_prob1['Sample File'].map(noc_map)

        # DEBUG: Check value counts right after mapping
        print("\nDEBUG: Value counts for 'NoC_True_Mapped' column (BEFORE dropna/astype):")
        print(df_prob1['NoC_True_Mapped'].value_counts(dropna=False))

        # Handle potential NaNs (where regex failed completely)
        if df_prob1['NoC_True_Mapped'].isnull().any():
            num_failed = df_prob1['NoC_True_Mapped'].isnull().sum()
            print(f"\nWarning: Mapping resulted in {num_failed} NaN values for NoC (Regex likely failed on these).")
            print("Rows with NaN NoC (showing first 5 unique Sample Files):")
            print(df_prob1[df_prob1['NoC_True_Mapped'].isnull()]['Sample File'].unique()[:5])
            print("Proceeding by dropping rows with NaN NoC...")
            df_prob1.dropna(subset=['NoC_True_Mapped'], inplace=True)
            if df_prob1.empty:
                 print("Error: No valid samples remaining after dropping NaN NoC.")
                 exit()

        # Assign to the final column name and convert type if not empty
        if not df_prob1.empty:
             df_prob1['NoC_True'] = df_prob1['NoC_True_Mapped'].astype(int)
             df_prob1.drop(columns=['NoC_True_Mapped'], inplace=True)
             print("\nSuccessfully processed True NoC for valid samples.")
             print(f"Shape after potentially dropping rows: {df_prob1.shape}")
        else:
             print("\nDataFrame became empty after handling NaNs. Cannot proceed.")
             exit()


        # CRITICAL CHECK: Display NoC distribution again
        noc_distribution = df_prob1['NoC_True'].value_counts().sort_index()
        print("\nCRITICAL CHECK: Distribution of samples per True NoC (after cleaning):")
        print(noc_distribution)
        if len(noc_distribution) == 0:
             print("\nERROR: No samples remaining after NoC processing!")
        # Check if the distribution now looks reasonable (contains values > 1)
        elif noc_distribution.index.max() <= 1 :
             print(f"\nERROR: Maximum NoC extracted is <= 1. Extraction likely failed. Check map values.")

    except Exception as e_noc:
         print(f"Error during NoC extraction phase: {e_noc}")
         import traceback
         traceback.print_exc()
         exit()

    # --- Step 2: Feature Engineering ... ---
    # ... (Rest of the script as in V1.4/V1.5, unchanged) ...
    if df_prob1.empty:
        print("\nDataFrame is empty after NoC processing. Cannot proceed to Feature Engineering.")
        exit()

    print("\n--- Starting Step 2: Feature Engineering for Wide Format (Version 1.1) ---")
    # Function to count valid alleles in a row
    def count_valid_alleles_in_row(row):
        count = 0
        for i in range(1, 101): # Check Allele 1 to Allele 100
            allele_col = f'Allele {i}'
            if allele_col in row.index and pd.notna(row[allele_col]):
                 count += 1
        return count
    df_prob1['allele_count_per_marker'] = df_prob1.apply(count_valid_alleles_in_row, axis=1)
    grouped_by_sample_wide = df_prob1.groupby('Sample File')
    mac_per_sample = grouped_by_sample_wide['allele_count_per_marker'].max()
    total_alleles_per_sample = grouped_by_sample_wide['allele_count_per_marker'].sum()
    avg_alleles_per_marker = grouped_by_sample_wide['allele_count_per_marker'].mean()
    markers_gt2 = grouped_by_sample_wide['allele_count_per_marker'].apply(lambda x: (x > 2).sum())
    markers_gt3 = grouped_by_sample_wide['allele_count_per_marker'].apply(lambda x: (x > 3).sum())
    markers_gt4 = grouped_by_sample_wide['allele_count_per_marker'].apply(lambda x: (x > 4).sum())
    def get_sample_height_stats(group):
        heights = []
        for i in range(1, 101): # Check Height 1 to Height 100
            height_col = f'Height {i}'
            if height_col in group:
                numeric_heights = pd.to_numeric(group[height_col], errors='coerce')
                heights.extend(numeric_heights.dropna().tolist())
        if heights:
            return pd.Series({'avg_peak_height': np.mean(heights), 'std_peak_height': np.std(heights)})
        else:
            return pd.Series({'avg_peak_height': 0, 'std_peak_height': 0})
    height_stats = grouped_by_sample_wide.apply(get_sample_height_stats)
    print("Combining features into final DataFrame...")
    unique_noc_map = df_prob1[['Sample File', 'NoC_True']].drop_duplicates().set_index('Sample File')['NoC_True']
    df_features = pd.DataFrame(index=mac_per_sample.index)
    df_features['NoC_True'] = df_features.index.map(unique_noc_map)
    if df_features['NoC_True'].isnull().any():
        print("Warning: Found NaN in NoC_True after mapping. Dropping these samples.")
        df_features.dropna(subset=['NoC_True'], inplace=True)
    if not df_features.empty:
         df_features['NoC_True'] = df_features['NoC_True'].astype(int)
    else:
         print("Error: Feature DataFrame became empty after handling NoC mapping. Exiting.")
         exit()
    df_features['max_allele_per_sample'] = mac_per_sample
    df_features['total_alleles_per_sample'] = total_alleles_per_sample
    df_features['avg_alleles_per_marker'] = avg_alleles_per_marker
    df_features['markers_gt2_alleles'] = markers_gt2
    df_features['markers_gt3_alleles'] = markers_gt3
    df_features['markers_gt4_alleles'] = markers_gt4
    df_features['avg_peak_height'] = height_stats['avg_peak_height']
    df_features['std_peak_height'] = height_stats['std_peak_height']
    df_features = df_features.dropna(subset=['max_allele_per_sample']) # Re-align after potential NoC drops
    df_features.fillna(0, inplace=True)
    df_features.reset_index(inplace=True)
    print("\n--- Feature Calculation Completed (Wide Format) ---")
    print(f"Feature DataFrame shape: {df_features.shape}")
    print("\nFirst 5 rows of the feature DataFrame:")
    print(df_features.head())
    print("\nSummary statistics of the features:")
    with pd.option_context('display.float_format', '{:.2f}'.format):
        print(df_features.describe())
    print("\nStep 2 (Feature Engineering - Wide Format) completed.")
    # ... (Optional saving) ...

else:
    print("\nExiting script as data could not be loaded successfully in Step 1.")
    exit()

# --- End of Script ---
print("\nScript finished executing Steps 1 and 2 for Problem 1 (Wide Format).")