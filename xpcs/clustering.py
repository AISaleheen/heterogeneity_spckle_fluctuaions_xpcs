# =============================================================================
# STEP 3: Dynamic Clustering & Heterogeneity Mapping
# =============================================================================
# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines   import Line2D

from sklearn.preprocessing    import StandardScaler
from sklearn.mixture          import GaussianMixture
from sklearn.cluster          import DBSCAN
from sklearn.metrics          import silhouette_score
from sklearn.neighbors        import NearestNeighbors

from statsmodels.tsa.stattools import acf as sm_acf

# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def build_feature_matrix(speckle_props_df,
                          g2_data,
                          short_lag_frac=0.1,
                          decay_lag_frac=(0.05, 0.5),
                          long_lag_frac=(0.5, 1.0)):
    """
    Build the feature matrix for clustering from dynamic speckles only.

    Features (all physically motivated scalars):
        1. contrast         : g2(tau_1) - 1, amplitude of correlation
        2. mean_g2_decay    : mean g2 over the decay region
                              (decay_lag_frac of total lags)
        3. mean_g2_long     : mean g2 at long lag times
                              (long_lag_frac of total lags)
        4. slope_decay      : linear slope of g2 over decay region
                              (captures rate of decorrelation)
        5. relax_time       : model-free relaxation time estimate
                              (lag where g2 first crosses back through
                              upper CI boundary)

    Only dynamic speckles with all features finite are included.

    Parameters:
        speckle_props_df : DataFrame from Step 2
        g2_data          : dict of g2 curves from Step 2
        short_lag_frac   : fraction defining short lag window
        decay_lag_frac   : (start, end) fraction for decay region
        long_lag_frac    : (start, end) fraction for long lag region

    Returns:
        feature_df : DataFrame with speckle_id + 5 features
                     (only dynamic speckles, all finite)
        feature_names : list of feature column names
    """
    print("\n--- Building feature matrix ---")

    dynamic_df = speckle_props_df[
        speckle_props_df['is_dynamic']].copy()

    print(f"  Dynamic speckles available : {len(dynamic_df)}")

    rows = []

    for _, row in dynamic_df.iterrows():
        sid  = row['speckle_id']
        data = g2_data.get(sid)
        if data is None:
            continue

        g2        = data['g2']
        lag_times = data['lag_times']
        ci_upper  = data['ci_upper']
        n_lags    = len(g2)

        # ---- Feature 1: contrast ---------------------------------- #
        contrast = float(row['contrast'])

        # ---- Feature 2: mean g2 over decay region ----------------- #
        d_start = max(0, int(decay_lag_frac[0] * n_lags))
        d_end   = max(d_start + 1,
                      int(decay_lag_frac[1] * n_lags))
        mean_g2_decay = float(np.mean(g2[d_start:d_end]))

        # ---- Feature 3: mean g2 at long lag times ----------------- #
        l_start = max(0, int(long_lag_frac[0] * n_lags))
        l_end   = n_lags
        mean_g2_long = float(np.mean(g2[l_start:l_end]))

        # ---- Feature 4: slope over decay region ------------------- #
        # Fit a line to g2 vs lag_time over the decay window.
        # Negative slope = decorrelating (dynamic).
        # Near-zero slope = static or very slowly relaxing.
        decay_lags = lag_times[d_start:d_end]
        decay_g2   = g2[d_start:d_end]
        if len(decay_lags) >= 2:
            slope_decay = float(
                np.polyfit(decay_lags, decay_g2, 1)[0])
        else:
            slope_decay = 0.0

        # ---- Feature 5: model-free relaxation time ---------------- #
        # Use stored value if available, otherwise recompute
        # as the lag where g2 first drops back below upper CI.
        relax_time = float(row.get('relaxation_time', np.nan)) \
                     if 'relaxation_time' in row.index \
                     else np.nan

        if not np.isfinite(relax_time):
            # Fallback: find first crossing below CI upper
            above = g2 > ci_upper
            relax_time = np.nan
            for i in range(1, n_lags):
                if above[i-1] and not above[i]:
                    relax_time = float(lag_times[i])
                    break
            # Second fallback: use lag of half-maximum
            if not np.isfinite(relax_time):
                half_max = 1.0 + contrast / 2.0
                below_half = np.where(g2 < half_max)[0]
                if len(below_half) > 0:
                    relax_time = float(lag_times[below_half[0]])

        rows.append({
            'speckle_id'    : sid,
            'centroid_x'    : float(row['centroid_x']),
            'centroid_y'    : float(row['centroid_y']),
            'n_pixels'      : int(row['n_pixels']),
            'contrast'      : contrast,
            'mean_g2_decay' : mean_g2_decay,
            'mean_g2_long'  : mean_g2_long,
            'slope_decay'   : slope_decay,
            'relax_time'    : relax_time
        })

    feature_df    = pd.DataFrame(rows)
    feature_names = ['contrast', 'mean_g2_decay',
                     'mean_g2_long', 'slope_decay',
                     'relax_time']

    # Drop rows with any non-finite feature
    n_before = len(feature_df)
    feature_df = feature_df.dropna(subset=feature_names)
    feature_df = feature_df[
        np.all(np.isfinite(feature_df[feature_names].values),
               axis=1)]
    n_after = len(feature_df)

    if n_before - n_after > 0:
        print(f"  Dropped {n_before - n_after} speckles with "
              f"non-finite features.")

    print(f"  Final feature matrix     : "
          f"{len(feature_df)} speckles x "
          f"{len(feature_names)} features")
    print(f"\n  Feature summary:")
    print(feature_df[feature_names].describe().round(4))

    return feature_df, feature_names


def scale_features(feature_df, feature_names):
    """
    Apply StandardScaler to feature matrix.
    Returns scaled array and fitted scaler for inverse transform.

    Parameters:
        feature_df    : DataFrame from build_feature_matrix
        feature_names : list of feature column names

    Returns:
        X_scaled : (n_speckles, n_features) scaled numpy array
        scaler   : fitted StandardScaler instance
    """
    X      = feature_df[feature_names].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"\n  Feature scaling (StandardScaler):")
    for i, fname in enumerate(feature_names):
        print(f"    {fname:<20s} : "
              f"mean={scaler.mean_[i]:.4f}  "
              f"std={scaler.scale_[i]:.4f}")

    return X_scaled, scaler


# =============================================================================
# GMM CLUSTERING WITH BIC SELECTION
# =============================================================================

def run_gmm_bic(X_scaled, n_components_range=range(2, 6),
                n_init=20, random_state=42):
    """
    Fit GMM for each number of components in n_components_range.
    Select optimal n_components by minimum BIC.

    Parameters:
        X_scaled           : scaled feature matrix
        n_components_range : iterable of component counts to try
        n_init             : number of GMM initialisations per fit
        random_state       : random seed

    Returns:
        best_gmm      : fitted GMM with optimal n_components
        best_n        : optimal number of components
        bic_scores    : list of BIC scores
        aic_scores    : list of AIC scores
        all_gmms      : list of all fitted GMM objects
        n_range       : list of n_components tried
    """
    print("\n--- GMM clustering with BIC selection ---")

    bic_scores = []
    aic_scores = []
    all_gmms   = []
    n_range    = list(n_components_range)

    for n in n_range:
        gmm = GaussianMixture(
            n_components=n,
            covariance_type='full',
            n_init=n_init,
            random_state=random_state,
            max_iter=500
        )
        gmm.fit(X_scaled)
        bic_scores.append(gmm.bic(X_scaled))
        aic_scores.append(gmm.aic(X_scaled))
        all_gmms.append(gmm)

        # Silhouette only defined for n >= 2 with >1 cluster assigned
        labels = gmm.predict(X_scaled)
        if len(np.unique(labels)) > 1:
            sil = silhouette_score(X_scaled, labels)
        else:
            sil = np.nan

        print(f"  n={n}  BIC={gmm.bic(X_scaled):8.2f}  "
              f"AIC={gmm.aic(X_scaled):8.2f}  "
              f"Silhouette={sil:.3f}")

    best_idx = int(np.argmin(bic_scores))
    best_n   = n_range[best_idx]
    best_gmm = all_gmms[best_idx]

    print(f"\n  Optimal n_components (min BIC) : {best_n}")

    return best_gmm, best_n, bic_scores, aic_scores, \
           all_gmms, n_range


def assign_gmm_clusters(feature_df, X_scaled, best_gmm):
    """
    Assign GMM cluster labels and soft probabilities to feature_df.

    Parameters:
        feature_df : feature DataFrame
        X_scaled   : scaled feature matrix
        best_gmm   : fitted GMM

    Returns:
        feature_df : updated with columns:
                         gmm_cluster    : hard label (argmax prob)
                         gmm_prob_<k>   : soft probability for each k
                         gmm_max_prob   : max probability (confidence)
    """
    labels      = best_gmm.predict(X_scaled)
    probs       = best_gmm.predict_proba(X_scaled)
    n_comp      = best_gmm.n_components

    feature_df = feature_df.copy()
    feature_df['gmm_cluster']  = labels
    feature_df['gmm_max_prob'] = probs.max(axis=1)

    for k in range(n_comp):
        feature_df[f'gmm_prob_{k}'] = probs[:, k]

    # Order clusters by mean relax_time so cluster 0 = fastest
    cluster_means = feature_df.groupby('gmm_cluster')[
        'relax_time'].mean().sort_values()
    label_map = {old: new
                 for new, old in enumerate(cluster_means.index)}
    feature_df['gmm_cluster'] = feature_df[
        'gmm_cluster'].map(label_map)

    print(f"\n  GMM cluster assignments "
          f"(ordered by mean relax_time):")
    for k in sorted(feature_df['gmm_cluster'].unique()):
        sub = feature_df[feature_df['gmm_cluster'] == k]
        print(f"    Cluster {k} : n={len(sub):4d}  "
              f"relax_time={sub['relax_time'].mean():.3e} s  "
              f"contrast={sub['contrast'].mean():.4f}  "
              f"mean_prob={sub['gmm_max_prob'].mean():.3f}")

    return feature_df


def run_dbscan(X_scaled, feature_df,
               eps=None, min_samples=3):
    """
    Run DBSCAN as a cross-check against GMM.
    If eps is None, estimate it from the k-nearest-neighbour
    distance plot (elbow method).

    Parameters:
        X_scaled     : scaled feature matrix
        feature_df   : feature DataFrame
        eps          : DBSCAN epsilon (None = auto-estimate)
        min_samples  : minimum samples per core point

    Returns:
        feature_df   : updated with column 'dbscan_cluster'
                       (-1 = noise/outlier)
        eps_used     : epsilon value used
    """
    print("\n--- DBSCAN cross-check ---")

    if eps is None:
        k    = min_samples
        nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
        distances, _ = nbrs.kneighbors(X_scaled)
        knn_dists    = np.sort(distances[:, -1])[::-1]

        if len(knn_dists) > 4:
            d2      = np.diff(np.diff(knn_dists))
            elbow   = int(np.argmax(np.abs(d2))) + 2
            eps_est = float(knn_dists[elbow])
        else:
            eps_est = float(np.percentile(knn_dists, 30))

        eps_used = eps_est
        print(f"  Auto-estimated eps : {eps_used:.4f} "
              f"(from k={k} NN elbow)")
    else:
        eps_used = eps
        print(f"  Using provided eps : {eps_used:.4f}")

    db        = DBSCAN(eps=eps_used, min_samples=min_samples)
    db_labels = db.fit_predict(X_scaled)

    feature_df = feature_df.copy()
    feature_df['dbscan_cluster'] = db_labels

    n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    n_noise    = int(np.sum(db_labels == -1))

    print(f"  DBSCAN clusters found  : {n_clusters}")
    print(f"  Noise points (-1)      : {n_noise} "
          f"({100*n_noise/len(db_labels):.1f}%)")

    for k in sorted(set(db_labels)):
        sub   = feature_df[feature_df['dbscan_cluster'] == k]
        label = 'Noise' if k == -1 else f'Cluster {k}'
        print(f"    {label:<12s} : n={len(sub):4d}  "
              f"relax_time={sub['relax_time'].mean():.3e} s  "
              f"contrast={sub['contrast'].mean():.4f}")

    return feature_df, eps_used


# =============================================================================
# MEAN ACF PER CLUSTER
# =============================================================================

def compute_cluster_mean_acf(feature_df, g2_data,
                              cluster_col='gmm_cluster',
                              max_lag_plot=None):
    """
    Compute the mean g2 curve for each cluster for physical
    interpretability verification.

    Parameters:
        feature_df    : feature DataFrame with cluster assignments
        g2_data       : dict of g2 curves from Step 2
        cluster_col   : column name of cluster labels to use
        max_lag_plot  : maximum number of lags to include in mean
                        (None = use all)

    Returns:
        cluster_acfs  : dict mapping cluster_label ->
                            {'mean_g2'   : array,
                             'std_g2'    : array,
                             'lag_times' : array,
                             'n'         : int}
    """
    print(f"\n--- Computing mean g2 per cluster "
          f"(column: {cluster_col}) ---")

    cluster_acfs = {}
    labels       = sorted(feature_df[cluster_col].unique())

    for k in labels:
        sub  = feature_df[feature_df[cluster_col] == k]
        sids = sub['speckle_id'].values

        g2_list   = []
        lag_ref   = None

        for sid in sids:
            data = g2_data.get(sid)
            if data is None:
                continue
            g2_arr = data['g2']
            lt_arr = data['lag_times']

            if max_lag_plot is not None:
                g2_arr = g2_arr[:max_lag_plot]
                lt_arr = lt_arr[:max_lag_plot]

            if lag_ref is None:
                lag_ref = lt_arr
                g2_list.append(g2_arr)
            else:
                # Align to shortest common length
                min_len = min(len(lag_ref), len(g2_arr))
                lag_ref = lag_ref[:min_len]
                g2_list = [g[:min_len] for g in g2_list]
                g2_list.append(g2_arr[:min_len])

        if len(g2_list) == 0:
            print(f"  Cluster {k} : no g2 data found, skipping.")
            continue

        g2_matrix = np.vstack(g2_list)
        mean_g2   = np.mean(g2_matrix, axis=0)
        std_g2    = np.std(g2_matrix, axis=0)

        cluster_acfs[k] = {
            'mean_g2'  : mean_g2,
            'std_g2'   : std_g2,
            'lag_times': lag_ref,
            'n'        : len(g2_list)
        }

        print(f"  Cluster {k} : n={len(g2_list):4d}  "
              f"mean g2 at tau_1 = "
              f"{mean_g2[0]:.4f}  "
              f"mean g2 at tau_end = "
              f"{mean_g2[-1]:.4f}")

    return cluster_acfs


# =============================================================================
# PLOTTING
# =============================================================================

def plot_step3_results(feature_df, g2_data,
                        bic_scores, aic_scores, n_range,
                        cluster_acfs_gmm,
                        cluster_acfs_dbscan,
                        best_n,
                        figure_path='step3_heterogeneity.png'):
    """
    Step 3 results figure with six panels:

    Panel 1 (top left)     : BIC/AIC vs n_components
    Panel 2 (top middle)   : GMM cluster assignments in
                             feature space (relax_time vs contrast)
    Panel 3 (top right)    : DBSCAN cluster assignments in
                             same feature space
    Panel 4 (middle left)  : Mean g2 per GMM cluster
    Panel 5 (middle right) : Mean g2 per DBSCAN cluster
    Panel 6 (bottom)       : GMM soft assignment probability
                             distribution per cluster
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Step 3: Heterogeneity Mapping',
                 fontsize=15, fontweight='bold')

    gmm_labels    = sorted(feature_df['gmm_cluster'].unique())
    dbscan_labels = sorted(feature_df['dbscan_cluster'].unique())

    # Colormaps
    gmm_cmap    = plt.cm.get_cmap('tab10', max(len(gmm_labels), 2))
    dbscan_cmap = plt.cm.get_cmap('tab10',
                                   max(len(dbscan_labels), 2))

    # ------------------------------------------------------------------ #
    # Panel 1: BIC / AIC vs n_components                                 #
    # ------------------------------------------------------------------ #
    ax1 = axes[0, 0]
    ax1.plot(n_range, bic_scores, 'o-',
             color='steelblue', linewidth=2,
             markersize=7, label='BIC')
    ax1.plot(n_range, aic_scores, 's--',
             color='darkorange', linewidth=2,
             markersize=7, label='AIC')
    ax1.axvline(best_n, color='red', linewidth=2,
                linestyle=':', label=f'Optimal n={best_n}')
    ax1.set_xlabel('Number of GMM Components', fontsize=11)
    ax1.set_ylabel('Score', fontsize=11)
    ax1.set_title('GMM Model Selection\nBIC / AIC vs n_components',
                  fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(n_range)

    # ------------------------------------------------------------------ #
    # Panel 2: GMM clusters in feature space                             #
    # ------------------------------------------------------------------ #
    ax2 = axes[0, 1]
    for k in gmm_labels:
        sub = feature_df[feature_df['gmm_cluster'] == k]
        ax2.scatter(sub['relax_time'],
                    sub['contrast'],
                    s=40 + 60 * sub['gmm_max_prob'],
                    c=[gmm_cmap(k)] * len(sub),
                    alpha=0.7,
                    label=f'Cluster {k} (n={len(sub)})',
                    edgecolors='white', linewidths=0.5,
                    zorder=3)

    ax2.set_xscale('log')
    ax2.set_xlabel('Relaxation Time (s, log scale)', fontsize=11)
    ax2.set_ylabel('Contrast (beta)', fontsize=11)
    ax2.set_title(f'GMM Clusters in Feature Space\n'
                  f'(n={best_n} components, '
                  f'marker size = assignment confidence)',
                  fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ------------------------------------------------------------------ #
    # Panel 3: DBSCAN clusters in feature space                          #
    # ------------------------------------------------------------------ #
    ax3 = axes[1, 0]
    for k in dbscan_labels:
        sub   = feature_df[feature_df['dbscan_cluster'] == k]
        color = '#aaaaaa' if k == -1 else dbscan_cmap(k)
        label = f'Noise (n={len(sub)})' \
                if k == -1 else f'Cluster {k} (n={len(sub)})'
        ax3.scatter(sub['relax_time'],
                    sub['contrast'],
                    s=40,
                    c=color,
                    alpha=0.7,
                    label=label,
                    edgecolors='white', linewidths=0.5,
                    zorder=3)

    ax3.set_xscale('log')
    ax3.set_xlabel('Relaxation Time (s, log scale)', fontsize=11)
    ax3.set_ylabel('Contrast (beta)', fontsize=11)
    ax3.set_title('DBSCAN Cross-Check in Feature Space\n'
                  '(gray = noise/outlier points)',
                  fontsize=11)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # ------------------------------------------------------------------ #
    # Panel 4: Mean g2 per GMM cluster                                   #
    # ------------------------------------------------------------------ #
    ax4 = axes[1, 1]
    for k, acf_data in cluster_acfs_gmm.items():
        color    = gmm_cmap(k)
        lag_times = acf_data['lag_times']
        mean_g2   = acf_data['mean_g2']
        std_g2    = acf_data['std_g2']
        n         = acf_data['n']

        ax4.plot(lag_times, mean_g2,
                 color=color, linewidth=2.0,
                 label=f'Cluster {k} (n={n})',
                 zorder=3)
        ax4.fill_between(lag_times,
                         mean_g2 - std_g2,
                         mean_g2 + std_g2,
                         color=color, alpha=0.15,
                         zorder=2)

    ax4.axhline(1.0, color='gray', linewidth=1.0,
                linestyle='--', alpha=0.5,
                label='g2 = 1 (baseline)')
    ax4.set_xscale('log')
    ax4.set_xlabel('Lag time (s)', fontsize=11)
    ax4.set_ylabel('g2(tau)', fontsize=11)
    ax4.set_title('Mean g2 per GMM Cluster\n'
                  '(shaded region = +/- 1 std)',
                  fontsize=11)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # ------------------------------------------------------------------ #
    # Panel 5: Mean g2 per DBSCAN cluster                                #
    # ------------------------------------------------------------------ #
    ax5 = axes[2, 0]
    for k, acf_data in cluster_acfs_dbscan.items():
        color     = '#aaaaaa' if k == -1 else dbscan_cmap(k)
        lag_times = acf_data['lag_times']
        mean_g2   = acf_data['mean_g2']
        std_g2    = acf_data['std_g2']
        n         = acf_data['n']
        label     = f'Noise (n={n})' \
                    if k == -1 else f'Cluster {k} (n={n})'

        ax5.plot(lag_times, mean_g2,
                 color=color, linewidth=2.0,
                 label=label, zorder=3)
        ax5.fill_between(lag_times,
                         mean_g2 - std_g2,
                         mean_g2 + std_g2,
                         color=color, alpha=0.15,
                         zorder=2)

    ax5.axhline(1.0, color='gray', linewidth=1.0,
                linestyle='--', alpha=0.5,
                label='g2 = 1 (baseline)')
    ax5.set_xscale('log')
    ax5.set_xlabel('Lag time (s)', fontsize=11)
    ax5.set_ylabel('g2(tau)', fontsize=11)
    ax5.set_title('Mean g2 per DBSCAN Cluster\n'
                  '(shaded region = +/- 1 std)',
                  fontsize=11)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    

   # ------------------------------------------------------------------ #
    # Panel 6: GMM soft probability distributions                        #
    # ------------------------------------------------------------------ #
    ax6 = axes[2, 1]
    for k in gmm_labels:
        sub     = feature_df[feature_df['gmm_cluster'] == k]
        probs_k = sub['gmm_max_prob'].values
        color   = gmm_cmap(k)
        ax6.hist(probs_k, bins=20,
                 alpha=0.6, color=color,
                 edgecolor='none', density=True,
                 label=f'Cluster {k} (n={len(sub)})')

    ax6.axvline(0.5, color='red', linewidth=1.5,
                linestyle='--', alpha=0.7,
                label='p=0.5 (uncertain boundary)')
    ax6.axvline(0.9, color='green', linewidth=1.5,
                linestyle='--', alpha=0.7,
                label='p=0.9 (confident assignment)')
    ax6.set_xlabel('Max Assignment Probability', fontsize=11)
    ax6.set_ylabel('Density', fontsize=11)
    ax6.set_title('GMM Soft Assignment Confidence\n'
                  'Speckles near cluster boundaries have lower '
                  'max probability',
                  fontsize=11)
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)

    # Annotate fraction of uncertain assignments
    n_uncertain = int(np.sum(
        feature_df['gmm_max_prob'] < 0.7))
    n_total     = len(feature_df)
    ax6.text(0.03, 0.95,
             f'Uncertain (p<0.7): '
             f'{n_uncertain}/{n_total} '
             f'({100*n_uncertain/n_total:.1f}%)',
             transform=ax6.transAxes,
             ha='left', va='top', fontsize=9,
             bbox=dict(boxstyle='round',
                       facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(figure_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nFigure saved as '{figure_path}'")


# =============================================================================
# HETEROGENEITY MAP
# =============================================================================

def plot_heterogeneity_map(feature_df,
                            speckle_props_df,
                            image_height=512,
                            image_width=512,
                            full_image=None,
                            cluster_col='gmm_cluster',
                            figure_path='step3_heterogeneity_map.png'):
    """
    Generate the heterogeneity map on the detector image.

    Three panels:
        Left   : cluster assignment map (GMM or DBSCAN)
                 dynamic speckles colored by cluster,
                 non-dynamic in gray, background in black
        Middle : relaxation time map (diverging colormap)
                 fast-relaxing = blue, slow-relaxing = red,
                 non-dynamic in gray, background in black
        Right  : GMM assignment confidence map
                 colored by max assignment probability,
                 non-dynamic in gray, background in black

    Parameters:
        feature_df       : feature DataFrame with cluster assignments
        speckle_props_df : full speckle DataFrame from Step 2
        image_height     : detector image height in pixels
        image_width      : detector image width in pixels
        full_image       : optional background detector image
        cluster_col      : column to use for cluster coloring
        figure_path      : output file path
    """
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    fig.suptitle('Step 3: Heterogeneity Map',
                 fontsize=15, fontweight='bold')

    gmm_labels = sorted(feature_df[cluster_col].unique())
    n_clusters = len([k for k in gmm_labels if k >= 0])
    cmap_clust = plt.cm.get_cmap('tab10',
                                  max(n_clusters, 2))

    # Background image setup
    for ax in axes:
        if full_image is not None:
            ax.imshow(full_image, cmap='gray', alpha=0.3,
                      vmin=0,
                      vmax=np.percentile(full_image, 99))
        else:
            ax.set_facecolor('black')
        ax.set_xlabel('Pixel X', fontsize=10)
        ax.set_ylabel('Pixel Y', fontsize=10)
        ax.set_xlim(0, image_width)
        ax.set_ylim(image_height, 0)

    # Merge feature_df cluster info back onto speckle_props_df
    merge_cols = ['speckle_id', cluster_col, 'gmm_max_prob',
                  'relax_time']
    merge_cols = [c for c in merge_cols
                  if c in feature_df.columns]
    merged = speckle_props_df.merge(
        feature_df[merge_cols],
        on='speckle_id', how='left')

    dynamic_merged    = merged[merged['is_dynamic']]
    nondynamic_merged = merged[~merged['is_dynamic']]

    # ------------------------------------------------------------------ #
    # Left: Cluster assignment map                                        #
    # ------------------------------------------------------------------ #
    ax1 = axes[0]

    # Non-dynamic speckles in gray
    if len(nondynamic_merged) > 0:
        ax1.scatter(nondynamic_merged['centroid_x'],
                    nondynamic_merged['centroid_y'],
                    s=10, c='#636e72', alpha=0.5,
                    label='Non-dynamic', zorder=2)

    # Dynamic speckles colored by cluster
    legend_elements = []
    for k in gmm_labels:
        sub   = dynamic_merged[
            dynamic_merged[cluster_col] == k]
        if len(sub) == 0:
            continue
        color = '#aaaaaa' if k == -1 \
                else cmap_clust(k)
        label = 'DBSCAN noise' if k == -1 \
                else f'Cluster {k} (n={len(sub)})'
        ax1.scatter(sub['centroid_x'],
                    sub['centroid_y'],
                    s=30, c=color,
                    alpha=0.85, zorder=3,
                    edgecolors='white',
                    linewidths=0.3)
        legend_elements.append(
            Patch(facecolor=color, label=label))

    legend_elements.append(
        Patch(facecolor='#636e72',
              label=f'Non-dynamic '
                    f'(n={len(nondynamic_merged)})'))
    ax1.legend(handles=legend_elements,
               fontsize=8, loc='upper right')
    ax1.set_title(f'Cluster Assignment Map\n'
                  f'({cluster_col}, '
                  f'n_clusters={n_clusters})',
                  fontsize=11)

    # ------------------------------------------------------------------ #
    # Middle: Relaxation time map (diverging colormap)                   #
    # ------------------------------------------------------------------ #
    ax2 = axes[1]

    # Non-dynamic in gray
    if len(nondynamic_merged) > 0:
        ax2.scatter(nondynamic_merged['centroid_x'],
                    nondynamic_merged['centroid_y'],
                    s=10, c='#636e72', alpha=0.5,
                    zorder=2)

    # Dynamic speckles with valid relax_time
    dyn_valid = dynamic_merged[
        dynamic_merged['relax_time'].notna() &
        np.isfinite(dynamic_merged['relax_time']) &
        (dynamic_merged['relax_time'] > 0)]

    dyn_invalid = dynamic_merged[
        ~dynamic_merged['speckle_id'].isin(
            dyn_valid['speckle_id'])]

    if len(dyn_invalid) > 0:
        ax2.scatter(dyn_invalid['centroid_x'],
                    dyn_invalid['centroid_y'],
                    s=10, c='#b2bec3', alpha=0.4,
                    zorder=2,
                    label='Dynamic / no tau_r')

    if len(dyn_valid) > 0:
        log_relax = np.log10(dyn_valid['relax_time'].values)
        vmin_r    = np.percentile(log_relax, 5)
        vmax_r    = np.percentile(log_relax, 95)

        # Diverging colormap centred at median
        #vcenter   = float(np.median(log_relax))
        #norm      = mcolors.TwoSlopeNorm(   vmin=vmin_r,vcenter=vcenter,vmax=vmax_r)
        # --- AFTER (Safe Guardrail) ---
        # Diverging colormap centred at median (with safe fallback for boundary edge cases)
        vcenter = float(np.median(log_relax))
        if vmin_r < vcenter < vmax_r:
            norm = mcolors.TwoSlopeNorm(vmin=vmin_r, vcenter=vcenter, vmax=vmax_r)
        elif vmin_r < vmax_r:
            # Fallback to linear normalization if median matches min or max
            norm = mcolors.Normalize(vmin=vmin_r, vmax=vmax_r)
        else:
            # Fallback if all points share the exact same relaxation time
            norm = mcolors.Normalize(vmin=vmin_r - 0.1, vmax=vmax_r + 0.1)

        sc = ax2.scatter(dyn_valid['centroid_x'],
                         dyn_valid['centroid_y'],
                         s=30,
                         c=log_relax,
                         cmap='RdBu_r',
                         norm=norm,
                         alpha=0.9,
                         edgecolors='white',
                         linewidths=0.3,
                         zorder=3)
        cbar = fig.colorbar(sc, ax=ax2,
                            fraction=0.046, pad=0.04)

        # Format ticks as real time values
        tick_vals = np.linspace(vmin_r, vmax_r, 5)
        cbar.set_ticks(tick_vals)
        cbar.set_ticklabels(
            [f'{10**v:.2e} s' for v in tick_vals],
            fontsize=7)
        cbar.set_label('Relaxation Time tau_r\n'
                       'Blue=fast  Red=slow',
                       fontsize=9)

    ax2.set_title('Relaxation Time Map\n'
                  'Diverging colormap: '
                  'blue=fast, red=slow',
                  fontsize=11)

    # ------------------------------------------------------------------ #
    # Right: GMM assignment confidence map                                #
    # ------------------------------------------------------------------ #
    ax3 = axes[2]

    if len(nondynamic_merged) > 0:
        ax3.scatter(nondynamic_merged['centroid_x'],
                    nondynamic_merged['centroid_y'],
                    s=10, c='#636e72', alpha=0.5,
                    zorder=2, label='Non-dynamic')

    conf_valid = dynamic_merged[
        dynamic_merged['gmm_max_prob'].notna() &
        np.isfinite(dynamic_merged['gmm_max_prob'])]

    if len(conf_valid) > 0:
        sc2 = ax3.scatter(conf_valid['centroid_x'],
                          conf_valid['centroid_y'],
                          s=30,
                          c=conf_valid['gmm_max_prob'],
                          cmap='plasma',
                          vmin=0.5, vmax=1.0,
                          alpha=0.9,
                          edgecolors='white',
                          linewidths=0.3,
                          zorder=3)
        cbar2 = fig.colorbar(sc2, ax=ax3,
                             fraction=0.046, pad=0.04)
        cbar2.set_label('Max GMM Assignment\n'
                        'Probability',
                        fontsize=9)

    ax3.set_title('GMM Assignment Confidence Map\n'
                  'Low probability = boundary speckle',
                  fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(figure_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nFigure saved as '{figure_path}'")


def save_step3_outputs(feature_df,
                        cluster_acfs_gmm,
                        cluster_acfs_dbscan,
                        output_prefix='step3'):
    """
    Save all Step 3 outputs to disk for use in Step 4.

    Saves:
        {prefix}_features.csv       : feature matrix + cluster labels
        {prefix}_cluster_acfs.npz   : mean g2 per cluster
        {prefix}_summary.txt        : human-readable summary
    """
    # Save feature DataFrame
    csv_path = f'{output_prefix}_features.csv'
    feature_df.to_csv(csv_path, index=False)
    print(f"  Feature matrix saved    -> {csv_path}")

    # Save cluster mean g2 curves
    npz_path = f'{output_prefix}_cluster_acfs.npz'
    npz_dict = {}
    for k, data in cluster_acfs_gmm.items():
        npz_dict[f'gmm_mean_g2_{k}']   = data['mean_g2']
        npz_dict[f'gmm_std_g2_{k}']    = data['std_g2']
        npz_dict[f'gmm_lag_times_{k}'] = data['lag_times']
    for k, data in cluster_acfs_dbscan.items():
        npz_dict[f'dbscan_mean_g2_{k}']   = data['mean_g2']
        npz_dict[f'dbscan_std_g2_{k}']    = data['std_g2']
        npz_dict[f'dbscan_lag_times_{k}'] = data['lag_times']
    np.savez_compressed(npz_path, **npz_dict)
    print(f"  Cluster ACFs saved      -> {npz_path}")

    # Save summary
    txt_path  = f'{output_prefix}_summary.txt'
    n_dynamic = len(feature_df)
    gmm_labels    = sorted(feature_df['gmm_cluster'].unique())
    dbscan_labels = sorted(feature_df['dbscan_cluster'].unique())

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 55 + "\n")
        f.write("STEP 3 SUMMARY: HETEROGENEITY MAPPING\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Dynamic speckles in feature matrix : "
                f"{n_dynamic}\n\n")

        f.write("--- GMM Clustering ---\n")
        f.write(f"  Optimal n_components : "
                f"{len(gmm_labels)}\n")
        for k in gmm_labels:
            sub = feature_df[feature_df['gmm_cluster'] == k]
            f.write(f"  Cluster {k} : n={len(sub):4d}  "
                    f"relax_time={sub['relax_time'].mean():.3e} s  "
                    f"contrast={sub['contrast'].mean():.4f}  "
                    f"mean_prob="
                    f"{sub['gmm_max_prob'].mean():.3f}\n")

        n_uncertain = int(np.sum(
            feature_df['gmm_max_prob'] < 0.7))
        f.write(f"\n  Uncertain assignments (p<0.7) : "
                f"{n_uncertain} "
                f"({100*n_uncertain/n_dynamic:.1f}%)\n\n")

        f.write("--- DBSCAN Cross-Check ---\n")
        n_noise = int(np.sum(
            feature_df['dbscan_cluster'] == -1))
        f.write(f"  Clusters found : "
                f"{len([k for k in dbscan_labels if k >= 0])}\n")
        f.write(f"  Noise points   : {n_noise} "
                f"({100*n_noise/n_dynamic:.1f}%)\n")
        for k in dbscan_labels:
            sub   = feature_df[feature_df['dbscan_cluster'] == k]
            label = 'Noise' if k == -1 else f'Cluster {k}'
            f.write(f"  {label:<12s} : n={len(sub):4d}  "
                    f"relax_time={sub['relax_time'].mean():.3e} s  "
                    f"contrast={sub['contrast'].mean():.4f}\n")

        f.write("\n--- Feature Statistics ---\n")
        feature_names = ['contrast', 'mean_g2_decay',
                         'mean_g2_long', 'slope_decay',
                         'relax_time']
        for fname in feature_names:
            if fname in feature_df.columns:
                vals = feature_df[fname].dropna()
                f.write(f"  {fname:<20s} : "
                        f"mean={vals.mean():.4f}  "
                        f"std={vals.std():.4f}  "
                        f"min={vals.min():.4f}  "
                        f"max={vals.max():.4f}\n")

    print(f"  Summary saved           -> {txt_path}")


