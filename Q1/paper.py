#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DNA Profile Number of Contributors (NoC) Decision Tree Implementation

This script implements a decision tree approach for determining the number of contributors
to a DNA profile, based on the research paper "Estimating the number of contributors to a DNA 
profile using decision trees" by Kruijver et al.

The implementation includes:
1. Feature extraction from STR profile data
2. Decision tree training for NoC classification
3. Evaluation and comparison of different stutter filtering methods
4. Visualization of decision trees and performance metrics
"""

from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import re
from scipy import stats
from scipy.signal import find_peaks
import os
import warnings
warnings.filterwarnings('ignore')

# Set plotting parameters
plt.rcParams['font.sans-serif'] = ['SimHei']  # For Chinese characters
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')

# Define paths and constants
DATA_DIR = './'
PLOTS_DIR = './noc_plots'
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)
    
def calculate_entropy(probabilities):
    """
    Calculate the entropy of a probability distribution.

    Args:
        probabilities (array-like): Probabilities of the distribution.

    Returns:
        float: Entropy value.
    """
    probabilities = np.array(probabilities)
    probabilities = probabilities[probabilities > 0]  # Remove zero probabilities
    return -np.sum(probabilities * np.log2(probabilities))

def calculate_ols_slope(x, y):
        """
        Calculate the slope of the ordinary least squares (OLS) regression line.
    
        Args:
            x (array-like): Independent variable values.
            y (array-like): Dependent variable values.
    
        Returns:
            float: Slope of the OLS regression line, or NaN if calculation fails.
        """
        if len(x) > 1 and len(y) > 1 and len(x) == len(y):
            x = np.array(x)
            y = np.array(y)
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            numerator = np.sum((x - x_mean) * (y - y_mean))
            denominator = np.sum((x - x_mean) ** 2)
            if denominator != 0:
                return numerator / denominator
        return np.nan

def calculate_wls_slope(x, y, weights):
        """
        Calculate the slope of the weighted least squares (WLS) regression line.
    
        Args:
            x (array-like): Independent variable values.
            y (array-like): Dependent variable values.
            weights (array-like): Weights for each data point.
    
        Returns:
            float: Slope of the WLS regression line, or NaN if calculation fails.
        """
        if len(x) > 1 and len(y) > 1 and len(weights) > 1 and len(x) == len(y) == len(weights):
            x = np.array(x)
            y = np.array(y)
            weights = np.array(weights)
            x_mean = np.average(x, weights=weights)
            y_mean = np.average(y, weights=weights)
            numerator = np.sum(weights * (x - x_mean) * (y - y_mean))
            denominator = np.sum(weights * (x - x_mean) ** 2)
            if denominator != 0:
                return numerator / denominator
        return np.nan

TRUE_ALLELE_CONFIDENCE_THRESHOLD = 0.5  # CTA threshold for true alleles

def extract_features_for_sample(sample_id, sample_data_loci, marker_params, H_sat=30000,
                                phr_imbalance_threshold=0.6, epsilon=1e-9, hp_bins=15,
                                modality_kde_bw='scott', modality_prominence=None):
    """
    Extracts all features for a single sample.

    Args:
        sample_id (str): Identifier for the sample.
        sample_data_loci (list): List of dicts for each locus.
                                Each locus dict: {'locus_name': str,
                                                'alleles': [{'value': str/float, 'height': float, 'size': float}, ...]}
        marker_params (dict): Maps locus_name to {'avg_size_bp': float, 'is_autosomal': bool, ...}
        H_sat (float): Saturation threshold for peak heights.
        phr_imbalance_threshold (float): Threshold for severe PHR imbalance.
        epsilon (float): Small constant to avoid division by zero.
        hp_bins (int): Number of bins for Hp calculation.
        modality_kde_bw: Bandwidth for KDE in modality calculation.
        modality_prominence: Prominence for peak finding in modality.
    
    Returns:
        Dict: A dictionary of calculated features, including sample_id.
    """
    features = {'sample_id': sample_id}
    L_all_expected_names = [name for name, params in marker_params.items()] # All loci names
    L_expected_autosomal_names = [name for name, params in marker_params.items() if params.get('is_autosomal', True)]
    L_expected_autosomal_count = len(L_expected_autosomal_names)
    features['L_expected_autosomal_count_kit'] = L_expected_autosomal_count
    
    all_effective_alleles_list = []
    for locus_info in sample_data_loci:
        for allele in locus_info['alleles']:
            # Ensure 'value' exists, provide a default if not
            if 'value' not in allele:
                allele['value'] = f"Unknown_{allele['size']}"
            all_effective_alleles_list.append(allele)
    
    # Process height and size values
    H_values_raw = np.array([allele['height'] for allele in all_effective_alleles_list])
    S_values_raw = np.array([allele['size'] for allele in all_effective_alleles_list])
    
    # Ensure H_values and S_values have the same length after filtering
    valid_indices_for_H_S = [i for i, allele in enumerate(all_effective_alleles_list) if 'height' in allele and 'size' in allele]
    H_values = np.array([all_effective_alleles_list[i]['height'] for i in valid_indices_for_H_S])
    S_values = np.array([all_effective_alleles_list[i]['size'] for i in valid_indices_for_H_S])
    
    M = len(H_values)
    features['M_total_effective_alleles'] = M
    
    # Process locus data
    locus_summary_map = {} # For observed loci
    for locus_info in sample_data_loci:
        locus_name = locus_info['locus_name']
        # Filter for effective alleles (height > 0)
        locus_alleles_effective = [a for a in locus_info['alleles'] if a.get('height', 0) > 0]
        
        H_l = np.sum([a.get('height', 0) for a in locus_alleles_effective])
        A_l_count = len(locus_alleles_effective)
        A_l_values = [a.get('value', '') for a in locus_alleles_effective] # Allele designations
        
        locus_summary_map[locus_name] = {
            'H_l': H_l,
            'A_l_count': A_l_count,
            'A_l_values': A_l_values,
            'alleles_details': locus_alleles_effective, # For PHR, max height
            'avg_size_bp': marker_params.get(locus_name, {}).get('avg_size_bp', np.nan)
        }
    
    # Le: Effective Loci (at least one effective allele observed)
    L_e_names = [name for name, data in locus_summary_map.items() if data['A_l_count'] > 0]
    num_loci_with_effective_alleles = len(L_e_names)
    features['num_loci_with_effective_alleles'] = num_loci_with_effective_alleles
    
    # List of allele counts for *all expected autosomal loci* (0 for dropouts)
    alleles_per_exp_locus_counts = np.array([locus_summary_map.get(name, {}).get('A_l_count', 0) for name in L_expected_autosomal_names])
    
    # A. Count-based features
    if num_loci_with_effective_alleles > 0: # Max over observed loci
        features['mac_profile'] = np.max([data['A_l_count'] for data in locus_summary_map.values()])
    else:
        features['mac_profile'] = 0
    
    all_distinct_allele_values = set()
    for data in locus_summary_map.values():
        all_distinct_allele_values.update(data['A_l_values'])
    features['total_distinct_alleles'] = len(all_distinct_allele_values)
    
    if L_expected_autosomal_count > 0:
        features['avg_alleles_per_locus'] = np.mean(alleles_per_exp_locus_counts)
        features['std_alleles_per_locus'] = np.std(alleles_per_exp_locus_counts)
    else:
        features['avg_alleles_per_locus'] = np.nan
        features['std_alleles_per_locus'] = np.nan
    
    # MGTN series
    for N_val in [2, 3, 4, 5, 6]:
        features[f'loci_gt{N_val}_alleles'] = np.sum(alleles_per_exp_locus_counts >= N_val)
    
    # Allele count distribution entropy
    if L_expected_autosomal_count > 0:
        counts_of_allele_counts = Counter(alleles_per_exp_locus_counts)
        probs_allele_counts = np.array([count / L_expected_autosomal_count for count in counts_of_allele_counts.values()])
        features['allele_count_dist_entropy'] = calculate_entropy(probs_allele_counts)
    else:
        features['allele_count_dist_entropy'] = 0.0
    
    # B. Peak height and balance features
    features['avg_peak_height'] = np.mean(H_values) if M > 0 else np.nan
    features['std_peak_height'] = np.std(H_values) if M > 1 else np.nan
    
    # PHR calculations
    phr_values = []
    for locus_name in L_e_names:
        locus_data = locus_summary_map[locus_name]
        if locus_data['A_l_count'] == 2:
            h1 = locus_data['alleles_details'][0]['height']
            h2 = locus_data['alleles_details'][1]['height']
            if max(h1, h2) > 0:
                phr = min(h1, h2) / max(h1, h2)
                if not np.isnan(phr):
                    phr_values.append(phr)
    
    features['num_loci_with_phr'] = len(phr_values)
    if len(phr_values) > 0:
        phr_values_np = np.array(phr_values)
        features['avg_phr'] = np.mean(phr_values_np)
        features['std_phr'] = np.std(phr_values_np) if len(phr_values_np) > 1 else np.nan
        features['min_phr'] = np.min(phr_values_np)
        features['median_phr'] = np.median(phr_values_np)
        num_severe_imbalance = np.sum(phr_values_np <= phr_imbalance_threshold)
        features['num_severe_imbalance_loci'] = num_severe_imbalance
        features['ratio_severe_imbalance_loci'] = num_severe_imbalance / len(phr_values)
    else:
        features['avg_phr'] = np.nan
        features['std_phr'] = np.nan
        features['min_phr'] = np.nan
        features['median_phr'] = np.nan
        features['num_severe_imbalance_loci'] = 0
        features['ratio_severe_imbalance_loci'] = np.nan
    
    # Peak height distribution moments
    if M > 2:
        features['skewness_peak_height'] = stats.skew(H_values, bias=False)
        features['kurtosis_peak_height'] = stats.kurtosis(H_values, fisher=False, bias=False)
    else:
        features['skewness_peak_height'] = np.nan
        features['kurtosis_peak_height'] = np.nan
    
    # Modality
    if M > 0:
        log_H_values = np.log(H_values + 1)
        if len(np.unique(log_H_values)) > 1:
            hist_counts, bin_edges = np.histogram(log_H_values, bins='auto')
            peaks, _ = find_peaks(hist_counts, prominence=modality_prominence)
            features['modality_peak_height'] = len(peaks)
        else:
            features['modality_peak_height'] = 1 if M > 0 else 0
    else:
        features['modality_peak_height'] = 0
    
    # Saturation indicators
    features['num_saturated_peaks'] = np.sum(H_values >= H_sat) if M > 0 else 0
    features['ratio_saturated_peaks'] = features['num_saturated_peaks'] / M if M > 0 else 0
    
    # C. Information theory and complexity features
    # Entropy of inter-locus balance
    H_l_values_for_entropy = np.array([locus_summary_map.get(name, {}).get('H_l', 0) for name in L_expected_autosomal_names])
    H_total_profile = np.sum(H_l_values_for_entropy)
    
    if H_total_profile > 0:
        P_l_dist = H_l_values_for_entropy[H_l_values_for_entropy > 0] / H_total_profile
        features['inter_locus_balance_entropy'] = calculate_entropy(P_l_dist)
    else:
        features['inter_locus_balance_entropy'] = 0.0
    
    # Average locus allele entropy
    locus_allele_entropies = []
    for locus_name in L_e_names:
        locus_data = locus_summary_map[locus_name]
        if locus_data['H_l'] > 0:
            allele_heights = np.array([a['height'] for a in locus_data['alleles_details']])
            probs = allele_heights / locus_data['H_l']
            locus_allele_entropies.append(calculate_entropy(probs))
    
    features['avg_locus_allele_entropy'] = np.mean(locus_allele_entropies) if len(locus_allele_entropies) > 0 else np.nan
    
    # Peak height distribution entropy
    if M > 0:
        log_H_plus_1 = np.log(H_values + 1)
        hist_counts, _ = np.histogram(log_H_plus_1, bins=hp_bins)
        probs_hp = hist_counts[hist_counts > 0] / M
        features['peak_height_entropy'] = calculate_entropy(probs_hp)
    else:
        features['peak_height_entropy'] = 0.0
    
    # Profile completeness indicator
    features['num_loci_no_effective_alleles'] = L_expected_autosomal_count - num_loci_with_effective_alleles
    
    # D. DNA degradation indicators
    # Height-size correlation
    if M > 1 and len(np.unique(H_values)) > 1 and len(np.unique(S_values)) > 1:
        correlation, _ = stats.pearsonr(S_values, H_values)
        features['height_size_correlation'] = correlation
    else:
        features['height_size_correlation'] = np.nan
    
    # Linear regression slopes
    features['height_size_slope'] = calculate_ols_slope(S_values, H_values)
    features['weighted_height_size_slope'] = calculate_wls_slope(S_values, H_values, H_values)
    
    # PHR-size slope
    phr_locus_sizes = []
    phr_locus_values_for_slope = []
    
    for locus_name in L_e_names:
        locus_data = locus_summary_map[locus_name]
        if locus_data['A_l_count'] == 2:
            h1 = locus_data['alleles_details'][0]['height']
            h2 = locus_data['alleles_details'][1]['height']
            if max(h1, h2) > 0:
                phr = min(h1, h2) / max(h1, h2)
                avg_size_bp = marker_params.get(locus_name, {}).get('avg_size_bp', np.nan)
                if not np.isnan(phr) and not np.isnan(avg_size_bp):
                    phr_locus_values_for_slope.append(phr)
                    phr_locus_sizes.append(avg_size_bp)
    
    features['phr_size_slope'] = calculate_ols_slope(phr_locus_sizes, phr_locus_values_for_slope)
    
    # Locus dropout weighted by size
    S_l_all_expected = np.array([marker_params.get(name, {}).get('avg_size_bp', np.nan) for name in L_expected_autosomal_names])
    P_l_dropout_flags = np.array([1 if locus_summary_map.get(name, {}).get('A_l_count', 0) == 0 else 0 for name in L_expected_autosomal_names])
    
    valid_dropout_indices = ~np.isnan(S_l_all_expected)
    S_l_valid = S_l_all_expected[valid_dropout_indices]
    P_l_valid = P_l_dropout_flags[valid_dropout_indices]
    
    if len(S_l_valid) > 0 and np.sum(S_l_valid) > 0:
        features['locus_dropout_score_weighted_by_size'] = np.sum(P_l_valid * S_l_valid) / np.sum(S_l_valid)
    else:
        features['locus_dropout_score_weighted_by_size'] = np.nan
    
    # RFU per base pair degradation index
    locus_max_heights = []
    locus_avg_sizes_for_degr = []
    
    for locus_name in L_e_names:
        locus_data = locus_summary_map[locus_name]
        if locus_data['A_l_count'] > 0:
            max_h = np.max([a['height'] for a in locus_data['alleles_details']])
            avg_size_bp = marker_params.get(locus_name, {}).get('avg_size_bp', np.nan)
            if not np.isnan(max_h) and not np.isnan(avg_size_bp):
                locus_max_heights.append(max_h)
                locus_avg_sizes_for_degr.append(avg_size_bp)
    
    features['degradation_index_rfu_per_bp'] = calculate_ols_slope(locus_avg_sizes_for_degr, locus_max_heights)
    
    # Small/large fragment completeness ratio
    V_l_completeness = np.array([1 if locus_summary_map.get(name, {}).get('A_l_count', 0) > 0 else 0 for name in L_expected_autosomal_names])
    
    L_small_vl = []
    L_large_vl = []
    num_small_loci_kit = 0
    num_large_loci_kit = 0
    
    for i, locus_name in enumerate(L_expected_autosomal_names):
        size_cat = marker_params.get(locus_name, {}).get('size_category', 'unknown')
        if size_cat == 'small':
            L_small_vl.append(V_l_completeness[i])
            num_small_loci_kit += 1
        elif size_cat == 'large':
            L_large_vl.append(V_l_completeness[i])
            num_large_loci_kit += 1
    
    C_S = np.sum(L_small_vl) / num_small_loci_kit if num_small_loci_kit > 0 else np.nan
    C_L = np.sum(L_large_vl) / num_large_loci_kit if num_large_loci_kit > 0 else np.nan
    
    if not np.isnan(C_S) and not np.isnan(C_L):
        features['info_completeness_ratio_small_large'] = C_S / max(C_L, epsilon)
    else:
        features['info_completeness_ratio_small_large'] = np.nan
    
    # E. Allele frequency features (placeholder values)
    features['avg_allele_freq'] = np.nan
    features['num_rare_alleles'] = np.nan
    features['min_allele_freq'] = np.nan
    features['avg_locus_sum_neg_log_freq'] = np.nan
    
    return features

def extract_noc_from_filename(filename_str):
    """
    Extract the true NoC from a sample filename.
    This function attempts to parse PROVEDIt naming convention to extract contributor number.
    
    Args:
        filename_str: Sample filename string
    
    Returns:
        int: Number of contributors, or nan if parsing fails
    """
    import re
    filename_str = str(filename_str)
    
    # Try to match PROVEDIt pattern with N contributors (1;1;1;1 or similar pattern)
    match = re.search(r'-(\d+(?:;\d+)*)[-_]', filename_str)
    if match:
        contributors = match.group(1).split(';')
        return len(contributors)
    
    # Try to match pattern with N_N_N_N (e.g., 1_2_3_4)
    match = re.search(r'-(\d+(?:_\d+)*)-', filename_str)
    if match:
        contributors = match.group(1).split('_')
        return len(contributors)
    
    return np.nan

def train_noc_decision_tree(df_features, max_depth=4):
    """
    Train a decision tree classifier for NoC assignment.
    
    Args:
        df_features: DataFrame with features for NoC prediction
        max_depth: Maximum depth of the decision tree
    
    Returns:
        trained decision tree model
    """
    # Prepare features and target
    X = df_features.drop(['Sample File', 'NoC_True'], axis=1, errors='ignore')
    y = df_features['NoC_True']
    
    # Create decision tree classifier
    dt_model = DecisionTreeClassifier(
        max_depth=max_depth,
        class_weight='balanced',
        criterion='gini',
        random_state=42
    )
    
    # Train the model
    dt_model.fit(X, y)
    
    return dt_model, X.columns

def visualize_tree(model, feature_names, output_file, class_names=None):
    """
    Visualize the decision tree model.
    
    Args:
        model: Trained decision tree model
        feature_names: List of feature names
        output_file: Path to save the visualization
        class_names: List of class names
    """
    plt.figure(figsize=(20, 10))
    plot_tree(model, 
              feature_names=feature_names, 
              class_names=class_names if class_names else [str(i) for i in range(1, 6)],
              filled=True, 
              rounded=True, 
              fontsize=10)
    plt.title('Decision Tree for NoC Assignment', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Decision tree visualization saved to {output_file}")

def evaluate_model(model, X_test, y_test, output_file_prefix):
    """
    Evaluate the model performance and generate visualizations.
    
    Args:
        model: Trained decision tree model
        X_test: Test features
        y_test: True labels
        output_file_prefix: Prefix for output files
    
    Returns:
        accuracy score
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, 
                         index=[f'{i}' for i in range(1, len(cm)+1)], 
                         columns=[f'{i}' for i in range(1, len(cm)+1)])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted NoC', fontsize=12)
    plt.ylabel('True NoC', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_file_prefix}_confusion_matrix.png", dpi=300)
    plt.close()
    
    # Generate normalized confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm_df = pd.DataFrame(cm_norm, 
                             index=[f'{i}' for i in range(1, len(cm)+1)], 
                             columns=[f'{i}' for i in range(1, len(cm)+1)])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm_df, annot=True, fmt='.2f', cmap='YlGnBu')
    plt.title('Normalized Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted NoC', fontsize=12)
    plt.ylabel('True NoC', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_file_prefix}_confusion_matrix_norm.png", dpi=300)
    plt.close()
    
    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print("Classification Report:")
    print(report_df)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_test.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
    plt.title('Feature Importance', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_file_prefix}_feature_importance.png", dpi=300)
    plt.close()
    
    print(f"Visualization files saved with prefix: {output_file_prefix}")
    
    return accuracy

def main():
    """Main function to run the NoC decision tree implementation"""
    print("DNA Profile Number of Contributors (NoC) Decision Tree Implementation")
    print("-------------------------------------------------------------------")
    
    # Example files - in a real implementation, these would be the actual files
    # from different filtering methods
    filter_methods = [
        {
            'name': '3SD',
            'file': 'prob1_processed_peaks_3SD.csv',
            'description': 'Peaks filtered using 3 standard deviations threshold'
        },
        {
            'name': 'tree',
            'file': 'prob1_processed_peaks_tree.csv',
            'description': 'Peaks filtered using decision tree approach'
        },
        {
            'name': 'oracle',
            'file': 'prob1_processed_peaks_oracle.csv',
            'description': 'Perfect filtering (only true contributor alleles)'
        }
    ]
    
    results = {}
    
    for method in filter_methods:
        print(f"\nProcessing {method['name']} filtered data ({method['description']})")
        
        # Load processed peak data
        peaks_file = os.path.join(DATA_DIR, method['file'])
        
        # In a real implementation, you would load the actual file
        # For demo purposes, let's create a synthetic dataset
        # df_peaks = load_processed_peaks(peaks_file)
        
        # For demo, create synthetic data to illustrate the workflow
        print(f"Note: Using synthetic data for demonstration (file {method['file']} not loaded)")
        n_samples = 500
        np.random.seed(42)
        
        # Create synthetic peaks data
        samples = [f"Sample_{i}" for i in range(n_samples)]
        noc_true = np.random.choice([1, 2, 3, 4, 5], size=n_samples)
        
        # Create features directly (in real implementation, would extract from peaks)
        df_features = pd.DataFrame({
            'Sample File': samples,
            'NoC_True': noc_true
        })
        
        # Add synthetic features that correlate with NoC
        for sample_idx, noc in enumerate(noc_true):
            # MACP roughly correlates with 2*NoC - random dropout
            df_features.loc[sample_idx, 'mac_profile'] = max(1, min(1 + 2*noc - np.random.randint(0, 3), 10))
            
            # Total distinct alleles increases with NoC
            df_features.loc[sample_idx, 'total_distinct_alleles'] = 10 * noc + np.random.randint(-5, 10)
            
            # Average alleles per locus increases with NoC
            df_features.loc[sample_idx, 'avg_alleles_per_locus'] = 1 + 0.8 * noc + np.random.normal(0, 0.5)
            
            # Standard deviation of alleles
            df_features.loc[sample_idx, 'std_alleles_per_locus'] = 0.2 * noc + np.random.normal(0, 0.2)
            
            # MGTN series
            for n in range(2, 7):
                threshold = 7 - n  # Higher thresholds for lower N
                df_features.loc[sample_idx, f'loci_gt{n}_alleles'] = max(0, min(noc - threshold, 20) + np.random.randint(-1, 2))
            
            # Peak height statistics
            df_features.loc[sample_idx, 'avg_peak_height'] = 1000 - 100 * noc + np.random.normal(0, 200)
            df_features.loc[sample_idx, 'std_peak_height'] = 100 * noc + np.random.normal(0, 50)
            
            # PHR statistics (lower for higher NoC)
            df_features.loc[sample_idx, 'avg_phr'] = max(0.1, min(0.95, 1.0 - 0.1 * noc + np.random.normal(0, 0.1)))
            df_features.loc[sample_idx, 'min_phr'] = max(0.05, min(0.9, 0.95 - 0.15 * noc + np.random.normal(0, 0.05)))
            
            # Other features
            df_features.loc[sample_idx, 'skewness_peak_height'] = 0.2 * noc + np.random.normal(0, 0.5)
            df_features.loc[sample_idx, 'kurtosis_peak_height'] = 0.1 * noc + np.random.normal(0, 0.3)
        
        print(f"Generated features for {len(df_features)} samples")
        
        # Split into training and test sets
        train_size = 300  # As used in the paper
        df_train = df_features.sample(n=min(train_size, len(df_features)), random_state=42)
        df_test = df_features[~df_features['Sample File'].isin(df_train['Sample File'])]
        
        print(f"Training set: {len(df_train)} samples")
        print(f"Test set: {len(df_test)} samples")
        
        # Train the decision tree
        X_train = df_train.drop(['Sample File', 'NoC_True'], axis=1)
        y_train = df_train['NoC_True']
        X_test = df_test.drop(['Sample File', 'NoC_True'], axis=1)
        y_test = df_test['NoC_True']
        
        dt_model, feature_names = train_noc_decision_tree(df_train)
        
        # Visualize the tree
        class_names = [f"{i}人" for i in range(1, 6)]  # Using Chinese characters as in the paper
        tree_file = os.path.join(PLOTS_DIR, f"decision_tree_{method['name']}.png")
        visualize_tree(dt_model, feature_names, tree_file, class_names)
        
        # Evaluate the model
        output_prefix = os.path.join(PLOTS_DIR, f"noc_{method['name']}")
        accuracy = evaluate_model(dt_model, X_test, y_test, output_prefix)
        
        # Store results
        results[method['name']] = accuracy
    
    # Compare results
    print("\nComparison of filtering methods:")
    for method, accuracy in results.items():
        print(f"{method}: {accuracy:.4f}")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    methods = list(results.keys())
    accuracies = [results[m] for m in methods]
    
    # Create a colored bar plot where higher values are in a different color
    colors = ['#1f77b4' if acc < max(accuracies) else '#d62728' for acc in accuracies]
    
    bars = plt.bar(methods, accuracies, color=colors)
    plt.ylim(0, 1.1)
    plt.ylabel('Accuracy')
    plt.xlabel('Filtering Method')
    plt.title('NoC Assignment Accuracy by Filtering Method')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add accuracy values on top of bars
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.03, f"{acc:.4f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'method_comparison.png'), dpi=300)
    
    print("\nImplementation completed successfully.")

if __name__ == "__main__":
    main()