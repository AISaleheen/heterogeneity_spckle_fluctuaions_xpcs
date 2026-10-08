#===========================================================
#Step 1: Feature Engineering & Speckle Segmentation
#==========================================================


import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.ndimage import uniform_filter, gaussian_filter
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings('ignore')



def extract_speckle_features(binned_series_list, pixel_ids, image_width=512):
    """
    Extract physically motivated features from each pixel's 
    binned photon count time series.
    
    Features:
        1. mean_count     : mean photon count rate (intensity)
        2. variance       : variance of count time series
        3. fano_factor    : variance/mean (super-Poissonian for speckles)
        4. max_count      : peak count in any bin
        5. zero_fraction  : fraction of bins with zero counts
        6. cv             : coefficient of variation (std/mean)
    
    Returns:
        feature_df : DataFrame with features and pixel coordinates
    """
    print("\n--- Extracting pixel features ---")
    
    features = []
    for i, counts in enumerate(binned_series_list):
        counts = np.array(counts, dtype=float)
        mean   = np.mean(counts)
        var    = np.var(counts, ddof=1)
        
        # Avoid division by zero for empty pixels
        fano        = var / mean if mean > 0 else 0.0
        cv          = np.std(counts, ddof=1) / mean if mean > 0 else 0.0
        max_count   = np.max(counts)
        zero_frac   = np.sum(counts == 0) / len(counts)
        
        # Decode pixel coordinates from 1did
        pid = pixel_ids[i]
        x   = pid % image_width
        y   = pid // image_width
        
        features.append({
            '1did'         : pid,
            'x'            : x,
            'y'            : y,
            'mean_count'   : mean,
            'variance'     : var,
            'fano_factor'  : fano,
            'max_count'    : max_count,
            'zero_fraction': zero_frac,
            'cv'           : cv
        })
    
    feature_df = pd.DataFrame(features)
    print(f"  Feature matrix shape: {feature_df.shape}")
    print(f"  Feature summary:")
    print(feature_df[['mean_count','variance','fano_factor',
                       'max_count','zero_fraction','cv']].describe().round(3))
    return feature_df


def build_intensity_image(feature_df, image_height=512, image_width=512):
    """
    Build a 2D mean intensity image from the feature DataFrame.
    This is the time-averaged detector image reconstructed from 
    per-pixel mean photon counts.
    """
    intensity_map = np.zeros((image_height, image_width))
    for _, row in feature_df.iterrows():
        intensity_map[int(row['y']), int(row['x'])] = row['mean_count']
    return intensity_map


def extract_enhanced_features(feature_df, binned_series_list,
                               image_height=512, image_width=512,
                               local_radius=5):
    """
    Extract enhanced, physically motivated features that are sensitive
    to local speckle contrast within the q-ring rather than global intensity.

    New features added:
        local_mean        : mean count of neighboring pixels (local background)
        local_contrast    : mean_count / local_mean (speckles > 1, bg ~ 1)
        local_excess      : mean_count - local_mean (absolute local excess)
        acf_short_lag     : mean ACF value at short lag times (coherence signal)
        intensity_rank    : percentile rank of pixel intensity within q-ring
        local_std         : std of neighboring pixel intensities
        peak_to_local_bg  : max_count / local_mean
    """
    print("--- Building intensity image for local contrast features ---")

    # Build 2D intensity image
    intensity_map = build_intensity_image(feature_df, image_height, image_width)

    # Compute local background using a uniform filter
    # This smooths the intensity image over a local neighborhood
    # giving the local diffuse background level at each pixel
    local_bg_map  = uniform_filter(intensity_map,
                                   size=local_radius * 2 + 1,
                                   mode='constant', cval=0.0)
    local_std_map = np.sqrt(uniform_filter(intensity_map ** 2,
                                           size=local_radius * 2 + 1,
                                           mode='constant', cval=0.0)
                            - local_bg_map ** 2)

    print("--- Extracting enhanced features ---")

    # Compute ACF at short lag times for each pixel
    # Use first 10 lag bins as short-lag estimate
    # This captures the coherence/speckle signal directly
    n_short = 10
    acf_short_list = []
    for counts in binned_series_list:
        counts = np.array(counts, dtype=float)
        mean   = np.mean(counts)
        if mean > 0 and len(counts) > n_short + 1:
            # Normalized ACF at lag=1 to lag=n_short
            # Simple estimate: correlation of count series with itself
            c0 = np.var(counts, ddof=1)
            if c0 > 0:
                acf_vals = []
                for lag in range(1, n_short + 1):
                    c_lag = np.mean((counts[:-lag] - mean) *
                                    (counts[lag:]  - mean))
                    acf_vals.append(c_lag / c0)
                acf_short_list.append(np.mean(acf_vals))
            else:
                acf_short_list.append(0.0)
        else:
            acf_short_list.append(0.0)

    # Percentile rank of mean_count within q-ring pixels
    from scipy.stats import rankdata
    intensity_ranks = rankdata(feature_df['mean_count'].values) / len(feature_df)

    # Build enhanced feature DataFrame
    enhanced_rows = []
    for i, (_, row) in enumerate(feature_df.iterrows()):
        x, y = int(row['x']), int(row['y'])

        local_bg  = local_bg_map[y, x]
        local_std = local_std_map[y, x]

        # Local contrast: how much brighter is this pixel vs its neighborhood
        local_contrast = row['mean_count'] / local_bg if local_bg > 0 else 1.0
        local_excess   = row['mean_count'] - local_bg
        peak_to_bg     = row['max_count']  / local_bg if local_bg > 0 else 1.0

        enhanced_rows.append({
            '1did'            : row['1did'],
            'x'               : x,
            'y'               : y,
            # Original features
            'mean_count'      : row['mean_count'],
            'variance'        : row['variance'],
            'fano_factor'     : row['fano_factor'],
            'max_count'       : row['max_count'],
            'zero_fraction'   : row['zero_fraction'],
            'cv'              : row['cv'],
            # New local contrast features
            'local_bg'        : local_bg,
            'local_contrast'  : local_contrast,
            'local_excess'    : local_excess,
            'local_std'       : local_std,
            'peak_to_local_bg': peak_to_bg,
            # ACF and rank features
            'acf_short_lag'   : acf_short_list[i],
            'intensity_rank'  : intensity_ranks[i],
        })

    enhanced_df = pd.DataFrame(enhanced_rows)

    print(f"  Enhanced feature matrix shape: {enhanced_df.shape}")
    print(f"\n  Local contrast statistics (key discriminating feature):")
    print(f"  Mean  : {enhanced_df['local_contrast'].mean():.3f}")
    print(f"  Std   : {enhanced_df['local_contrast'].std():.3f}")
    print(f"  Min   : {enhanced_df['local_contrast'].min():.3f}")
    print(f"  Max   : {enhanced_df['local_contrast'].max():.3f}")
    print(f"  95th percentile: "
          f"{np.percentile(enhanced_df['local_contrast'], 95):.3f}")
    print(f"  99th percentile: "
          f"{np.percentile(enhanced_df['local_contrast'], 99):.3f}")

    return enhanced_df



def plot_feature_diagnostics(enhanced_df):
    """
    Plot the distributions of the most important new features
    to help choose the right clustering threshold.
    This is a diagnostic figure to examine before running clustering.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Feature Diagnostics — Revised Features for Speckle Finding',
                 fontsize=14, fontweight='bold')

    features_to_plot = [
        ('local_contrast',   'Local Contrast (mean/local_bg)',    'blue'),
        ('local_excess',     'Local Excess (mean - local_bg)',     'green'),
        ('acf_short_lag',    'Short-Lag ACF',                     'red'),
        ('intensity_rank',   'Intensity Percentile Rank',          'purple'),
        ('fano_factor',      'Fano Factor',                        'orange'),
        ('peak_to_local_bg', 'Peak Count / Local Background',      'brown'),
    ]

    for ax, (feat, label, color) in zip(axes.flat, features_to_plot):
        vals = enhanced_df[feat].values
        ax.hist(vals, bins=80, color=color, alpha=0.7, edgecolor='none')
        ax.axvline(np.percentile(vals, 95), color='red',
                   linestyle='--', linewidth=1.5,
                   label=f'95th pct: {np.percentile(vals, 95):.3f}')
        ax.axvline(np.percentile(vals, 99), color='darkred',
                   linestyle='--', linewidth=1.5,
                   label=f'99th pct: {np.percentile(vals, 99):.3f}')
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(f'Distribution of {label}', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    #plt.savefig('step1_feature_diagnostics.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Diagnostic figure saved as 'step1_feature_diagnostics.png'")
    print("\nExamine the 'local_contrast' distribution carefully.")
    print("If speckles are present, there should be a tail above ~1.3 to 1.5.")
    print("Use this to guide the threshold or clustering approach.")




def run_streamlined_speckle_finding_loose(enhanced_df,
                                     image_height=512,
                                     image_width=512,
                                     local_contrast_threshold=1.5):
    """
    Streamlined speckle finding based on diagnostic insights:

    Key findings from diagnostics:
        - Local contrast is the only strongly discriminating feature
        - Short-lag ACF, Fano factor, intensity rank are not discriminating
        - No clean bimodal separation exists, so we use local_contrast
          threshold to define candidates, then apply ML within that set

    Parameters:
        local_contrast_threshold : pixels above this are speckle candidates
                                   (default 1.5, i.e. 50% brighter than
                                    local background). Adjust based on
                                    how many speckles you expect.
    """

    print("=" * 60)
    print("STREAMLINED SPECKLE FINDING")
    print("=" * 60)
    print(f"\nLocal contrast threshold : {local_contrast_threshold}")

    # ------------------------------------------------------------------ #
    # Use only the two most informative features based on diagnostics     #
    # ------------------------------------------------------------------ #
    useful_features = ['local_contrast', 'peak_to_local_bg', 'local_excess']

    # ------------------------------------------------------------------ #
    # Approach 1: Hard threshold on local contrast                        #
    # This is the most transparent and physically motivated approach      #
    # ------------------------------------------------------------------ #
    threshold_labels = (enhanced_df['local_contrast'] >=
                        local_contrast_threshold).astype(int)
    enhanced_df['threshold_label'] = threshold_labels
    n_threshold = int(threshold_labels.sum())
    print(f"\n[Approach 1] Hard threshold (local_contrast >= "
          f"{local_contrast_threshold})")
    print(f"  Speckles found : {n_threshold}")

    # ------------------------------------------------------------------ #
    # Approach 2: GMM on local_contrast only (1D)                        #
    # 1D GMM is much more interpretable and robust here                   #
    # ------------------------------------------------------------------ #
    print(f"\n[Approach 2] 1D GMM on local_contrast...")
    X_1d = enhanced_df['local_contrast'].values.reshape(-1, 1)
    
    # Try 2 and 3 components, pick lower BIC
    bic_scores = {}
    gmm_models = {}
    for n in [2, 3]:
        g = GaussianMixture(n_components=n, random_state=42, n_init=20)
        g.fit(X_1d)
        bic_scores[n] = g.bic(X_1d)
        gmm_models[n] = g
        print(f"  n_components={n} : BIC = {g.bic(X_1d):.1f}")

    best_n   = min(bic_scores, key=bic_scores.get)
    best_gmm = gmm_models[best_n]
    print(f"  Best n_components : {best_n} (lower BIC wins)")

    gmm1d_labels = best_gmm.predict(X_1d)
    gmm1d_probs  = best_gmm.predict_proba(X_1d)

    # Component means — speckle component has highest mean
    component_means = best_gmm.means_.flatten()
    print(f"  Component means : {np.sort(component_means).round(3)}")

    # Identify speckle component as the one(s) with mean > 1.0
    # (locally brighter than background)
    speckle_components = np.where(component_means >
                                   component_means.min() + 0.1)[0]
    print(f"  Speckle components : {speckle_components} "
          f"(means: {component_means[speckle_components].round(3)})")

    # Assign speckle label: 1 if assigned to any speckle component
    gmm1d_speckle = np.isin(gmm1d_labels, speckle_components).astype(int)
    
    # Speckle probability = sum of probabilities for speckle components
    gmm1d_speckle_prob = gmm1d_probs[:, speckle_components].sum(axis=1)

    enhanced_df['gmm1d_label']      = gmm1d_speckle
    enhanced_df['gmm1d_prob']       = gmm1d_speckle_prob
    n_gmm1d = int(gmm1d_speckle.sum())
    print(f"  Speckles found : {n_gmm1d}")

    # ------------------------------------------------------------------ #
    # Approach 3: K-Means on [local_contrast, peak_to_local_bg]          #
    # Only two features — more robust than 6-feature version             #
    # ------------------------------------------------------------------ #
    print(f"\n[Approach 3] K-Means (k=2) on "
          f"[local_contrast, peak_to_local_bg]...")
    X_2d     = enhanced_df[['local_contrast',
                              'peak_to_local_bg']].values
    scaler   = StandardScaler()
    X_2d_sc  = scaler.fit_transform(X_2d)

    kmeans   = KMeans(n_clusters=2, random_state=42, n_init=20)
    km_labels = kmeans.fit_predict(X_2d_sc)

    # Speckle cluster = higher local contrast
    if (enhanced_df['local_contrast'][km_labels == 0].mean() >
        enhanced_df['local_contrast'][km_labels == 1].mean()):
        km_labels = 1 - km_labels

    enhanced_df['kmeans_label'] = km_labels
    km_sil = silhouette_score(X_2d_sc, km_labels)
    n_km   = int(km_labels.sum())
    print(f"  Silhouette score : {km_sil:.3f}")
    print(f"  Speckles found   : {n_km}")

    # ------------------------------------------------------------------ #
    # Approach 4: DBSCAN on [local_contrast, peak_to_local_bg]           #
    # ------------------------------------------------------------------ #
    print(f"\n[Approach 4] DBSCAN on [local_contrast, peak_to_local_bg]...")
    best_result = {'eps': None, 'n_speckles': 0, 'labels': None}

    for eps_val in [0.2, 0.3, 0.5, 0.7, 1.0]:
        db      = DBSCAN(eps=eps_val, min_samples=5)
        db_raw  = db.fit_predict(X_2d_sc)
        unique  = np.unique(db_raw[db_raw != -1])
        n_noise = np.sum(db_raw == -1)

        if len(unique) >= 1:
            # Find cluster with highest mean local contrast
            cluster_means = {
                cl: enhanced_df['local_contrast'].values[db_raw == cl].mean()
                for cl in unique
            }
            speckle_cl = max(cluster_means, key=cluster_means.get)
            n_sp       = int(np.sum(db_raw == speckle_cl))
            print(f"  eps={eps_val:.1f} → {len(unique)} clusters, "
                  f"{n_sp} speckle candidates, {n_noise} noise points")

            # Pick the eps that gives a speckle count closest to
            # the threshold-based estimate as a reference
            if abs(n_sp - n_threshold) < abs(
                    best_result['n_speckles'] - n_threshold):
                db_labels_full = np.zeros(len(enhanced_df), dtype=int)
                for i in range(len(enhanced_df)):
                    if db_raw[i] == speckle_cl:
                        db_labels_full[i] = 1
                best_result = {
                    'eps'      : eps_val,
                    'n_speckles': n_sp,
                    'labels'   : db_labels_full
                }

    if best_result['labels'] is not None:
        enhanced_df['dbscan_label'] = best_result['labels']
        n_db = best_result['n_speckles']
        print(f"  Best eps : {best_result['eps']}")
        print(f"  Speckles found : {n_db}")
    else:
        enhanced_df['dbscan_label'] = np.zeros(len(enhanced_df), dtype=int)
        n_db = 0

    # ------------------------------------------------------------------ #
    # Consensus: threshold AND GMM1D AND KMeans all agree                 #
    # This is the most conservative, highest confidence speckle list      #
    # ------------------------------------------------------------------ #
    consensus = ((enhanced_df['threshold_label'] == 1) &
                 (enhanced_df['gmm1d_label']     == 1) &
                 (enhanced_df['kmeans_label']    == 1))
    enhanced_df['consensus_label'] = consensus.astype(int)
    n_consensus = int(consensus.sum())

    # ------------------------------------------------------------------ #
    # Summary                                                             #
    # ------------------------------------------------------------------ #
    results = {
        'threshold': {'n_speckles': n_threshold,
                      'silhouette': np.nan},
        'gmm1d'    : {'n_speckles': n_gmm1d,
                      'silhouette': silhouette_score(
                          X_1d, gmm1d_speckle) if n_gmm1d > 0 else np.nan,
                      'bic'       : bic_scores[best_n]},
        'kmeans'   : {'n_speckles': n_km,
                      'silhouette': km_sil},
        'dbscan'   : {'n_speckles': n_db,
                      'silhouette': np.nan},
        'consensus': {'n_speckles': n_consensus,
                      'silhouette': np.nan}
    }

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"  {'Method':<20} {'N Speckles':>12}")
    print(f"  {'-'*34}")
    for method, res in results.items():
        print(f"  {method:<20} {res['n_speckles']:>12}")
    print(f"\n  Note: Threshold uses local_contrast >= {local_contrast_threshold}")
    print(f"  Recommendation: Use consensus or GMM1D for Step 2.")

    return enhanced_df, results



def _build_single_map(enhanced_df, label_col,
                       image_height=512, image_width=512):
    """Helper to build a single 2D cluster map."""
    cmap_2d = np.full((image_height, image_width), np.nan)
    for _, row in enhanced_df.iterrows():
        cmap_2d[int(row['y']), int(row['x'])] = row[label_col]
    return cmap_2d



def plot_streamlined_results(enhanced_df, results,
                              full_image=None,
                              image_height=512, image_width=512,
                              local_contrast_threshold=1.5):
    """
    Results figure for streamlined speckle finding.

    Top row   : Threshold, GMM1D, K-Means, Consensus maps
    Bottom row : GMM1D probability map, local contrast histogram
                 with GMM component overlay, 2D feature scatter,
                 summary table
    """
    fig = plt.figure(figsize=(24, 12))
    fig.suptitle('Step 1 (Streamlined): ML-Based Speckle Finding',
                 fontsize=15, fontweight='bold', y=0.99)

    cmap_binary  = mcolors.ListedColormap(['#2c2c54', '#f9ca24'])
    method_names = ['Threshold\n(local_contrast)',
                    'GMM 1D\n(local_contrast)',
                    'K-Means\n(contrast + peak)',
                    'Consensus\n(Threshold+GMM+KM)']
    method_keys  = ['threshold_label', 'gmm1d_label',
                    'kmeans_label',    'consensus_label']

    # ------------------------------------------------------------------ #
    # TOP ROW: Four cluster maps                                          #
    # ------------------------------------------------------------------ #
    for i, (name, key) in enumerate(zip(method_names, method_keys)):
        ax = fig.add_subplot(2, 4, i + 1)

        if full_image is not None:
            ax.imshow(full_image, cmap='gray', alpha=0.4,
                      vmin=0, vmax=np.percentile(full_image, 99))

        im = ax.imshow(
            _build_single_map(enhanced_df, key, image_height, image_width),
            cmap=cmap_binary, alpha=0.85, vmin=0, vmax=1,
            interpolation='nearest')

        method_short = key.replace('_label', '')
        n_sp = results.get(method_short, {}).get(
               'n_speckles', int(enhanced_df[key].sum()))

        ax.set_title(f'{name}\nSpeckles: {n_sp}', fontsize=10)
        ax.set_xlabel('Pixel X', fontsize=9)
        ax.set_ylabel('Pixel Y', fontsize=9)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['BG', 'Speckle'], fontsize=8)

    # ------------------------------------------------------------------ #
    # BOTTOM LEFT: GMM1D probability map                                  #
    # ------------------------------------------------------------------ #
    ax5 = fig.add_subplot(2, 4, 5)

    if full_image is not None:
        ax5.imshow(full_image, cmap='gray', alpha=0.4,
                   vmin=0, vmax=np.percentile(full_image, 99))

    prob_map = _build_single_map(enhanced_df, 'gmm1d_prob',
                                  image_height, image_width)
    im5 = ax5.imshow(prob_map, cmap='plasma', alpha=0.85,
                     vmin=0, vmax=1, interpolation='nearest')
    ax5.set_title('GMM 1D Speckle Probability\n(Soft Assignment)', fontsize=10)
    ax5.set_xlabel('Pixel X', fontsize=9)
    ax5.set_ylabel('Pixel Y', fontsize=9)
    fig.colorbar(im5, ax=ax5, fraction=0.046,
                 pad=0.04).set_label('P(Speckle)', fontsize=8)

    # ------------------------------------------------------------------ #
    # BOTTOM 2nd: Local contrast histogram with GMM component overlay     #
    # ------------------------------------------------------------------ #
    ax6 = fig.add_subplot(2, 4, 6)

    lc_vals = enhanced_df['local_contrast'].values
    ax6.hist(lc_vals, bins=100, density=True,
             color='steelblue', alpha=0.5,
             label='All pixels', edgecolor='none')

    # Overlay GMM component distributions
    x_range = np.linspace(lc_vals.min(), lc_vals.max(), 500)
    
    # Re-fit 1D GMM to get component parameters for plotting
    from scipy.stats import norm as scipy_norm
    X_1d = lc_vals.reshape(-1, 1)
    best_n = 2
    best_bic = np.inf
    for n in [2, 3]:
        g_tmp = GaussianMixture(n_components=n, random_state=42, n_init=20)
        g_tmp.fit(X_1d)
        if g_tmp.bic(X_1d) < best_bic:
            best_bic = g_tmp.bic(X_1d)
            best_n   = n
            best_gmm_plot = g_tmp

    component_colors = ['#e74c3c', '#2ecc71', '#9b59b6']
    for k in range(best_n):
        mean_k = best_gmm_plot.means_[k, 0]
        std_k  = np.sqrt(best_gmm_plot.covariances_[k, 0, 0])
        weight = best_gmm_plot.weights_[k]
        y_k    = weight * scipy_norm.pdf(x_range, mean_k, std_k)
        label  = (f'Component {k+1}: '
                  f'μ={mean_k:.3f}, σ={std_k:.3f}, w={weight:.2f}')
        ax6.plot(x_range, y_k, color=component_colors[k],
                 linewidth=2, label=label)

    # Total GMM density
    y_total = np.zeros_like(x_range)
    for k in range(best_n):
        mean_k = best_gmm_plot.means_[k, 0]
        std_k  = np.sqrt(best_gmm_plot.covariances_[k, 0, 0])
        weight = best_gmm_plot.weights_[k]
        y_total += weight * scipy_norm.pdf(x_range, mean_k, std_k)
    ax6.plot(x_range, y_total, 'k--', linewidth=1.5,
             label='Total GMM', alpha=0.8)

    # Threshold line
    ax6.axvline(local_contrast_threshold, color='orange',
                linewidth=2, linestyle='--',
                label=f'Threshold: {local_contrast_threshold}')

    ax6.set_xlabel('Local Contrast (mean / local_bg)', fontsize=10)
    ax6.set_ylabel('Density', fontsize=10)
    ax6.set_title('Local Contrast Distribution\nwith GMM Component Fit',
                  fontsize=10)
    ax6.legend(fontsize=7)
    ax6.grid(True, alpha=0.3)

    # ------------------------------------------------------------------ #
    # BOTTOM 3rd: 2D scatter — local contrast vs peak_to_local_bg        #
    # colored by consensus label                                          #
    # ------------------------------------------------------------------ #
    ax7 = fig.add_subplot(2, 4, 7)

    consensus_mask = enhanced_df['consensus_label'] == 1
    bg_mask        = enhanced_df['consensus_label'] == 0

    ax7.scatter(enhanced_df.loc[bg_mask,        'local_contrast'],
                enhanced_df.loc[bg_mask,        'peak_to_local_bg'],
                s=1, alpha=0.15, color='#2c2c54', label='Background')
    ax7.scatter(enhanced_df.loc[consensus_mask, 'local_contrast'],
                enhanced_df.loc[consensus_mask, 'peak_to_local_bg'],
                s=8, alpha=0.7, color='#f9ca24', label='Speckle (consensus)',
                zorder=5)

    ax7.axvline(local_contrast_threshold, color='orange',
                linewidth=1.5, linestyle='--',
                label=f'Contrast threshold: {local_contrast_threshold}')
    ax7.set_xlabel('Local Contrast', fontsize=10)
    ax7.set_ylabel('Peak Count / Local Background', fontsize=10)
    ax7.set_title('Feature Space\nLocal Contrast vs Peak/BG', fontsize=10)
    ax7.legend(fontsize=8, markerscale=2)
    ax7.grid(True, alpha=0.3)

    # ------------------------------------------------------------------ #
    # BOTTOM RIGHT: Summary table                                         #
    # ------------------------------------------------------------------ #
    ax8 = fig.add_subplot(2, 4, 8)
    ax8.axis('off')

    km_arr  = enhanced_df['kmeans_label'].values
    gmm_arr = enhanced_df['gmm1d_label'].values
    th_arr  = enhanced_df['threshold_label'].values
    cs_arr  = enhanced_df['consensus_label'].values

    th_km_agree  = np.mean(th_arr  == km_arr)  * 100
    th_gmm_agree = np.mean(th_arr  == gmm_arr) * 100
    km_gmm_agree = np.mean(km_arr  == gmm_arr) * 100

    table_data = [
        ['Metric',                   'Value'],
        ['Total q-ring pixels',      f'{len(enhanced_df)}'],
        ['--- Speckle Counts ---',   '---'],
        ['Threshold',                f'{results["threshold"]["n_speckles"]}'],
        ['GMM 1D',                   f'{results["gmm1d"]["n_speckles"]}'],
        ['K-Means',                  f'{results["kmeans"]["n_speckles"]}'],
        ['DBSCAN',                   f'{results["dbscan"]["n_speckles"]}'],
        ['Consensus',                f'{results["consensus"]["n_speckles"]}'],
        ['--- Agreement ---',        '---'],
        ['Threshold vs GMM',         f'{th_gmm_agree:.1f}%'],
        ['Threshold vs K-Means',     f'{th_km_agree:.1f}%'],
        ['GMM vs K-Means',           f'{km_gmm_agree:.1f}%'],
        ['--- Quality ---',          '---'],
        ['GMM BIC',                  f'{results["gmm1d"]["bic"]:.1f}'],
        ['K-Means silhouette',       f'{results["kmeans"]["silhouette"]:.3f}'],
        ['Contrast threshold used',  f'{local_contrast_threshold}'],
    ]

    table = ax8.table(cellText=table_data[1:],
                      colLabels=table_data[0],
                      cellLoc='center',
                      loc='center',
                      bbox=[0.0, 0.0, 1.0, 1.0])
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)

    # Style header
    for j in range(2):
        table[0, j].set_facecolor('#2c2c54')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Style section divider rows
    for row_idx in [2, 8, 12]:
        for j in range(2):
            table[row_idx, j].set_facecolor('#dfe6e9')
            table[row_idx, j].set_text_props(fontstyle='italic')

    # Highlight consensus row
    for j in range(2):
        table[7, j].set_facecolor('#ffeaa7')
        table[7, j].set_text_props(fontweight='bold')

    ax8.set_title('Summary', fontsize=11, pad=20)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig('step1_streamlined_speckle_finding.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("\nFigure saved as 'step1_streamlined_speckle_finding.png'")

def get_final_speckle_list(enhanced_df, method='consensus'):
    """
    Extract the final speckle pixel list for Step 2.

    Parameters:
        enhanced_df : DataFrame with all cluster label columns
        method      : 'consensus' (recommended), 'gmm1d',
                      'kmeans', 'threshold', 'dbscan'
    Returns:
        speckle_df  : DataFrame of speckle pixels only
    """
    label_col  = f'{method}_label'
    speckle_df = enhanced_df[enhanced_df[label_col] == 1].copy()
    speckle_df = speckle_df.reset_index(drop=True)

    print(f"\n--- Final speckle list ({method}) ---")
    print(f"  Speckle pixels          : {len(speckle_df)}")
    print(f"  Mean local contrast     : "
          f"{speckle_df['local_contrast'].mean():.3f}")
    print(f"  Std local contrast      : "
          f"{speckle_df['local_contrast'].std():.3f}")
    print(f"  Mean photon count       : "
          f"{speckle_df['mean_count'].mean():.3f}")
    print(f"  X range                 : "
          f"{int(speckle_df['x'].min())} to {int(speckle_df['x'].max())}")
    print(f"  Y range                 : "
          f"{int(speckle_df['y'].min())} to {int(speckle_df['y'].max())}")

    return speckle_df

