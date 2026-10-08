# =============================================================================
# STEP 2: Speckle Properties & Autocorrelation Function Calculation
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import norm as scipy_norm
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')


from statsmodels.tsa.stattools import acf as sm_acf
from scipy.ndimage import label as nd_label


# =============================================================================
# Compute g2s for each pixel
# =============================================================================


def compute_pixel_g2(counts, max_lag=None, normalize=True):
    """
    Compute g2(tau) using statsmodels ACF for efficiency and
    reliability, then convert to XPCS g2 convention.

    statsmodels ACF uses FFT-based computation which is much faster
    than the explicit loop used previously, especially for long
    time series.

    Relationship:
        g2(tau) = 1 + ACF(tau) * (sigma_I / mean_I)^2

    Parameters:
        counts   : 1D array of photon counts per time bin
        max_lag  : maximum lag bin (default: len(counts)//4)
        normalize: if True return g2, if False return raw correlation

    Returns:
        lags : array of lag indices starting from 1
        g2   : array of g2 values
    """
    counts = np.array(counts, dtype=float)
    n      = len(counts)

    if max_lag is None:
        max_lag = n // 4

    mean_I = np.mean(counts)
    if mean_I == 0:
        return np.arange(1, max_lag + 1), np.ones(max_lag)

    var_I  = np.var(counts, ddof=1)

    # statsmodels acf returns values at lags 0, 1, 2, ..., max_lag
    # We use fft=True for speed and unbiased=False for consistency
    # with the standard XPCS estimator
    acf_vals = sm_acf(counts,
                      nlags=max_lag,
                      fft=True,
                      adjusted=False)

    # acf_vals[0] = 1.0 (lag 0, always)
    # acf_vals[1:] = ACF at lags 1, 2, ..., max_lag
    acf_at_lags = acf_vals[1:]   # shape (max_lag,)

    lags = np.arange(1, max_lag + 1)

    if normalize:
        # Convert ACF to g2 using the exact relationship
        # g2(tau) = 1 + ACF(tau) * (sigma / mean)^2
        cv_squared = var_I / (mean_I ** 2)
        g2 = 1.0 + acf_at_lags * cv_squared
    else:
        g2 = acf_at_lags

    return lags, g2


# =============================================================================
# Compute g2s for a speckle
# =============================================================================


def compute_speckle_g2(speckle_pixel_ids, binned_series_list,
                        pixel_ids_list, max_lag=None):
    """
    Compute g2 for a speckle by averaging pixel time series first,
    then computing g2 on the combined series using statsmodels ACF.

    Averaging before computing g2 improves SNR because it reduces
    single-pixel shot noise before the correlation is computed.
    """
    id_to_idx = {pid: idx for idx, pid in enumerate(pixel_ids_list)}

    count_arrays = []
    for pid in speckle_pixel_ids:
        if pid in id_to_idx:
            count_arrays.append(
                np.array(binned_series_list[id_to_idx[pid]],
                         dtype=float))

    if len(count_arrays) == 0:
        return None, None, 0.0, 0

    count_matrix = np.vstack(count_arrays) 
    combined = np.sum(count_matrix, axis=0)       # CHANGED: mean -> sum
    mean_I   = float(np.mean(count_matrix)) # This is the grand mean of all pixels and all time bins,
                                            # giving photons/pixel/bin which is comparable across speckles
                                                # of different sizes regardless of the sum operation above
    #combined     = np.mean(count_matrix, axis=0)
    #mean_I       = float(np.mean(combined))
    n_pixels     = len(count_arrays)

    if mean_I == 0:
        return None, None, 0.0, n_pixels

    if max_lag is None:
        max_lag = len(combined) // 4

    lags, g2 = compute_pixel_g2(combined, max_lag=max_lag)

    return lags, g2, mean_I, n_pixels




# =============================================================================
# Compute confidence interval for a g2
# =============================================================================

def compute_confidence_interval(counts, max_lag=None,
                                 n_bootstrap=200, ci_level=0.95):
    """
    Compute bootstrap confidence interval for g2 at each lag.
    This is the 'above_line_or_not' test baseline.

    For a purely Poisson (non-dynamic) pixel, g2(tau) should be flat
    at 1.0 for all tau > 0. The CI gives the expected fluctuation
    range under the null hypothesis of no dynamics.

    Parameters:
        counts      : 1D photon count array
        max_lag     : maximum lag to compute
        n_bootstrap : number of bootstrap resamples
        ci_level    : confidence interval level (default 0.95)

    Returns:
        ci_upper : upper CI bound at each lag
        ci_lower : lower CI bound at each lag
        ci_mean  : mean g2 under null hypothesis (should be ~1.0)
    """
    counts = np.array(counts, dtype=float)
    n      = len(counts)

    if max_lag is None:
        max_lag = n // 4

    alpha      = 1.0 - ci_level
    g2_boot    = np.zeros((n_bootstrap, max_lag))
     #np.random.permutation(n) generates a random ordering of the integers 0 through n-1. Indexing counts with this permutation produces a 
    #copy of the time series with the time bins randomly reordered. 
    #This shuffled series has the same mean and the same count distribution as the original but zero temporal autocorrelation by construction.
    for b in range(n_bootstrap):
        # Shuffle timestamps to destroy temporal correlations
        # This gives the null distribution (no dynamics)       
        shuffled      = counts[np.random.permutation(n)]
        _, g2_b       = compute_pixel_g2(shuffled, max_lag=max_lag)
        g2_boot[b, :] = g2_b
        # g2_boot[b, :] = g2_b stores this curve as the b-th row of the bootstrap matrix.

    ci_lower = np.percentile(g2_boot, 100 * alpha / 2,     axis=0)
    ci_upper = np.percentile(g2_boot, 100 * (1 - alpha/2), axis=0)
    ci_mean  = np.mean(g2_boot, axis=0)

    return ci_upper, ci_lower, ci_mean

# =============================================================================
# Compute if a g2 shows statistically significant correlation or not
# =============================================================================

def above_line_test(g2, ci_upper, ci_lower,
                     short_lag_frac=0.1,
                     intermediate_lag_frac=(0.1, 0.5)):
    """
    Statistical test classifying a speckle as dynamic or non-dynamic.

    A speckle is classified as DYNAMIC if its g2 curve exceeds the
    upper confidence interval at short lag times, indicating a
    measurable intensity autocorrelation above the noise floor.

    Also extracts:
        - Speckle contrast  : g2 value at the shortest lag (g2[0] - 1)
                              which estimates beta, the coherence factor
        - Mean g2 in intermediate lag range where decay is occurring
        - Relaxation time   : lag at which g2 first crosses the upper CI
                              from above (model-free estimate of tau_r)

    Parameters:
        g2                    : g2 array
        ci_upper              : upper CI bound array
        ci_lower              : lower CI bound array
        short_lag_frac        : fraction of lags considered 'short lag'
        intermediate_lag_frac : (start, end) fraction of lags for
                                intermediate range

    Returns:
        result dict with keys:
            is_dynamic       : bool
            contrast         : g2[0] - 1 (speckle contrast / beta)
            mean_g2_short    : mean g2 in short lag range
            mean_g2_intermed : mean g2 in intermediate lag range
            relaxation_lag   : lag index where g2 crosses upper CI
                               (None if never crosses)
            n_above_ci       : number of lags where g2 > ci_upper
    """
    n_lags = len(g2)

    short_end   = max(1, int(short_lag_frac * n_lags))
    intermed_start = max(1, int(intermediate_lag_frac[0] * n_lags))
    intermed_end   = max(2, int(intermediate_lag_frac[1] * n_lags))

    # Short lag statistics
    mean_g2_short = float(np.mean(g2[:short_end]))
    contrast      = float(g2[0] - 1.0)   # beta = g2(tau->0) - 1

    # Intermediate lag statistics
    mean_g2_intermed = float(np.mean(g2[intermed_start:intermed_end]))

    # Count lags where g2 exceeds upper CI
    above_ci   = g2 > ci_upper
    n_above_ci = int(np.sum(above_ci))

    # Dynamic classification: must be above CI at short lags
    is_dynamic = bool(np.any(above_ci[:short_end]))

    # Relaxation time: first lag where g2 drops below upper CI
    # after being above it (the crossing point)
    relaxation_lag = None
    if is_dynamic:
        # Find where g2 first crosses from above to below the CI
        for i in range(1, n_lags):
            if above_ci[i - 1] and not above_ci[i]:
                relaxation_lag = i
                break
        # If g2 stays above CI throughout, use last lag as lower bound
        if relaxation_lag is None and n_above_ci > 0:
            relaxation_lag = n_lags  # lower bound: longer than measured

    return {
        'is_dynamic'       : is_dynamic,
        'contrast'         : contrast,
        'mean_g2_short'    : mean_g2_short,
        'mean_g2_intermed' : mean_g2_intermed,
        'relaxation_lag'   : relaxation_lag,
        'n_above_ci'       : n_above_ci
    }

# =============================================================================
# Connected component labelling to group pixels and identify speckles
# =============================================================================


def identify_speckles_from_pixels(speckle_df,
                                   image_height=512,
                                   image_width=512,
                                   connectivity=2):
    """
    Group individual speckle pixels into spatially connected speckle
    objects using connected-component labeling.

    Each connected group of pixels is one speckle. The centroid of
    each group becomes the speckle's detector coordinate.

    Parameters:
        speckle_df   : DataFrame of speckle pixels with 'x', 'y', '1did'
        image_height : detector height
        image_width  : detector width
        connectivity : pixel connectivity (1=4-connected, 2=8-connected)

    Returns:
        speckle_objects : list of dicts, each with:
                          'speckle_id', 'pixel_ids', 'centroid_x',
                          'centroid_y', 'n_pixels'
    """
    from scipy.ndimage import label as nd_label

    print(f"\n--- Identifying connected speckle objects ---")
    print(f"  Input speckle pixels : {len(speckle_df)}")

    # Build binary mask on detector grid
    mask = np.zeros((image_height, image_width), dtype=int)
    for _, row in speckle_df.iterrows():
        mask[int(row['y']), int(row['x'])] = 1

    # 8-connectivity structure
    if connectivity == 2:
        struct = np.ones((3, 3), dtype=int)
    else:
        struct = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=int)

    labeled_array, n_objects = nd_label(mask, structure=struct)
    print(f"  Connected speckle objects found : {n_objects}")

    # Build pixel_id lookup
    pid_map = {}
    for _, row in speckle_df.iterrows():
        pid_map[(int(row['y']), int(row['x']))] = int(row['1did'])

    speckle_objects = []
    for obj_id in range(1, n_objects + 1):
        yx_coords = np.argwhere(labeled_array == obj_id)

        # Centroid
        centroid_y = float(np.mean(yx_coords[:, 0]))
        centroid_x = float(np.mean(yx_coords[:, 1]))

        # Pixel IDs
        pixel_ids = []
        for y, x in yx_coords:
            if (y, x) in pid_map:
                pixel_ids.append(pid_map[(y, x)])

        speckle_objects.append({
            'speckle_id': obj_id,
            'pixel_ids' : pixel_ids,
            'centroid_x': centroid_x,
            'centroid_y': centroid_y,
            'n_pixels'  : len(yx_coords)
        })

    # Size distribution summary
    sizes = [s['n_pixels'] for s in speckle_objects]
    print(f"  Speckle size (pixels): "
          f"min={min(sizes)}, median={np.median(sizes):.1f}, "
          f"max={max(sizes)}")
    
    return speckle_objects


# =============================================================================
# Compute features for all identified speckle objects
# =============================================================================


def extract_speckle_properties(speckle_objects,
                                binned_series_list,
                                pixel_ids_list,
                                dt,
                                max_lag=None,
                                n_bootstrap=200,
                                ci_level=0.95,
                                short_lag_frac=0.1,
                                intermediate_lag_frac=(0.1, 0.5),
                                min_pixels=1,
                                max_pixels=None):
    """
    Main Step 2 function. For each speckle object, compute all
    properties and organize into the central speckle DataFrame.
    """
    print("\n--- Extracting speckle properties ---")
    print(f"  Number of speckle objects : {len(speckle_objects)}")
    print(f"  dt                        : {dt:.3e}")
    print(f"  Bootstrap samples         : {n_bootstrap}")
    print(f"  CI level                  : {ci_level*100:.0f}%")

    rows    = []
    g2_data = {}
    intensity_data = {}   # ADD: store per-speckle I(t) traces

    # Build id_to_idx lookup once
    id_to_idx = {pid: idx for idx, pid in enumerate(pixel_ids_list)}

    for i, speckle in enumerate(speckle_objects):
        sid      = speckle['speckle_id']
        pids     = speckle['pixel_ids']
        n_pixels = speckle['n_pixels']

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Processing speckle {i+1} / "
                  f"{len(speckle_objects)} (id={sid}, "
                  f"n_pixels={n_pixels})...")

        # Skip speckles with too few pixels
        if n_pixels < min_pixels:
            continue
        # ADDED: Skip speckles with too many pixels
        if max_pixels is not None and n_pixels > max_pixels:
            print(f"  Skipping speckle {sid}: "
              f"n_pixels={n_pixels} exceeds max_pixels={max_pixels}")
            continue

        # ---------------------------------------------------------- #
        # Build combined speckle time series                          #
        # ---------------------------------------------------------- #
        count_arrays = []
        for pid in pids:
            if pid in id_to_idx:
                count_arrays.append(
                    np.array(binned_series_list[id_to_idx[pid]],
                             dtype=float))

        if len(count_arrays) == 0:
            continue
        

        #count_matrix = np.vstack(count_arrays)  # (n_pixels, n_bins)
        #combined     = np.mean(count_matrix, axis=0)
        #mean_I       = float(np.mean(combined))
        count_matrix = np.vstack(count_arrays)      # shape: (n_pixels, n_bins)
        combined     = np.sum(count_matrix, axis=0) # CHANGED: sum across pixels
        mean_I       = float(np.mean(count_matrix)) # CHANGED: grand mean of matrix

        if mean_I == 0:
            continue
        
        intensity_data[sid] = combined   # ADD: save I(t) trace
        # ---------------------------------------------------------- #
        # Compute g2                                                  #
        # ---------------------------------------------------------- #
        n_bins   = len(combined)
        max_lag_ = max_lag if max_lag is not None else n_bins // 4
        lags, g2 = compute_pixel_g2(combined, max_lag=max_lag_)
        lag_times = lags * dt

        # ---------------------------------------------------------- #
        # Compute confidence interval via bootstrap                   #
        # ---------------------------------------------------------- #
        ci_upper, ci_lower, ci_mean = compute_confidence_interval(
            combined,
            max_lag=max_lag_,
            n_bootstrap=n_bootstrap,
            ci_level=ci_level
        )

        # ---------------------------------------------------------- #
        # Above-line test and property extraction                     #
        # ---------------------------------------------------------- #
        test_result = above_line_test(
            g2, ci_upper, ci_lower,
            short_lag_frac=short_lag_frac,
            intermediate_lag_frac=intermediate_lag_frac
        )

        # Convert relaxation lag to physical time
        relax_lag  = test_result['relaxation_lag']
        relax_time = float(relax_lag * dt) if relax_lag is not None \
                     else np.nan

        # ---------------------------------------------------------- #
        # Store g2 curve data                                         #
        # ---------------------------------------------------------- #
        g2_data[sid] = {
            'lags'     : lags,
            'lag_times': lag_times,
            'g2'       : g2,
            'ci_upper' : ci_upper,
            'ci_lower' : ci_lower,
            'ci_mean'  : ci_mean
        }

        # ---------------------------------------------------------- #
        # Build row for DataFrame                                     #
        # ---------------------------------------------------------- #
        rows.append({
            'speckle_id'       : sid,
            'centroid_x'       : speckle['centroid_x'],
            'centroid_y'       : speckle['centroid_y'],
            'n_pixels'         : n_pixels,
            'mean_intensity'   : mean_I,
            'contrast'         : test_result['contrast'],
            'is_dynamic'       : test_result['is_dynamic'],
            'mean_g2_short'    : test_result['mean_g2_short'],
            'mean_g2_intermed' : test_result['mean_g2_intermed'],
            'relaxation_lag'   : relax_lag,
            'relaxation_time'  : relax_time,
            'n_above_ci'       : test_result['n_above_ci'],
            'n_lags'           : len(lags),
            'max_lag_time'     : float(lag_times[-1])
        })

    speckle_props_df = pd.DataFrame(rows)

    # ---------------------------------------------------------- #
    # Summary statistics                                          #
    # ---------------------------------------------------------- #
    n_total   = len(speckle_props_df)
    n_dynamic = int(speckle_props_df['is_dynamic'].sum()) \
                if n_total > 0 else 0

    print(f"\n  --- Step 2 Summary ---")
    print(f"  Total speckles processed  : {n_total}")
    print(f"  Dynamic speckles          : {n_dynamic} "
          f"({100*n_dynamic/n_total:.1f}%)" if n_total > 0
          else "  Dynamic speckles          : 0")
    print(f"  Non-dynamic speckles      : {n_total - n_dynamic}")

    if n_total > 0:
        print(f"\n  Mean intensity  : "
              f"{speckle_props_df['mean_intensity'].mean():.3f} "
              f"+/- {speckle_props_df['mean_intensity'].std():.3f}")
        print(f"  Mean contrast   : "
              f"{speckle_props_df['contrast'].mean():.4f} "
              f"+/- {speckle_props_df['contrast'].std():.4f}")

        dyn_df = speckle_props_df[speckle_props_df['is_dynamic'] &
                                   speckle_props_df[
                                       'relaxation_time'].notna()]
        if len(dyn_df) > 0:
            print(f"  Relaxation time : "
                  f"{dyn_df['relaxation_time'].mean():.3e} "
                  f"+/- {dyn_df['relaxation_time'].std():.3e} "
                  f"(dynamic speckles only)")

    return speckle_props_df, g2_data, intensity_data



# =============================================================================
# Plot results from step 2
# =============================================================================



def plot_step2_results(speckle_props_df, g2_data,
                        dt, max_curves=40,
                        ci_level=0.95,
                        figure_path='step2_speckle_properties.png'):
    """
    Generate the Step 2 results figure with four panels:

    Panel 1 (top left)     : g2 curves for dynamic speckles overlaid
                             with 95% CI bounds
    Panel 2 (top right)    : g2 curves for non-dynamic speckles
    Panel 3 (bottom left)  : Distribution of speckle contrast
    Panel 4 (bottom right) : Distribution of relaxation times
                             (dynamic speckles only)
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Step 2: Speckle Property Extraction',
                 fontsize=15, fontweight='bold')

    dynamic_df    = speckle_props_df[speckle_props_df['is_dynamic']]
    nondynamic_df = speckle_props_df[~speckle_props_df['is_dynamic']]

    print(f"\n  Plotting {min(len(dynamic_df), max_curves)} dynamic "
          f"and {min(len(nondynamic_df), max_curves)} non-dynamic "
          f"g2 curves...")

    # ------------------------------------------------------------------ #
    # Panel 1: Dynamic speckle g2 curves                                  #
    # ------------------------------------------------------------------ #
    ax1 = axes[0, 0]

    plot_dynamic = dynamic_df.sample(
        n=min(len(dynamic_df), max_curves),
        random_state=42) if len(dynamic_df) > 0 else dynamic_df

    ci_upper_ref  = None
    ci_lower_ref  = None
    ref_lag_times = None

    for _, row in plot_dynamic.iterrows():
        sid  = row['speckle_id']
        data = g2_data.get(sid)
        if data is None:
            continue
        ax1.plot(data['lag_times'], data['g2'],
                 color='steelblue', alpha=0.3, linewidth=0.8)
        if ci_upper_ref is None:
            ci_upper_ref  = data['ci_upper']
            ci_lower_ref  = data['ci_lower']
            ref_lag_times = data['lag_times']

    # Mean g2 across all plotted dynamic speckles
    if len(plot_dynamic) > 0:
        all_g2 = [g2_data[sid]['g2']
                  for sid in plot_dynamic['speckle_id']
                  if sid in g2_data]
        if len(all_g2) > 0:
            mean_g2 = np.mean(np.vstack(all_g2), axis=0)
            ax1.plot(ref_lag_times, mean_g2,
                     color='navy', linewidth=2.0,
                     label='Mean g2 (dynamic)', zorder=5)

    if ci_upper_ref is not None:
        ax1.axhline(np.mean(ci_upper_ref), color='red',
                    linewidth=1.5, linestyle='--',
                    label=f'{int(ci_level*100)}% CI upper')
        ax1.axhline(np.mean(ci_lower_ref), color='red',
                    linewidth=1.5, linestyle=':',
                    label=f'{int(ci_level*100)}% CI lower')
        ax1.fill_between(ref_lag_times,
                         ci_lower_ref, ci_upper_ref,
                         alpha=0.15, color='red', label='CI band')

    ax1.axhline(1.0, color='gray', linewidth=1.0,
                linestyle='--', alpha=0.5, label='g2 = 1 (baseline)')
    ax1.set_xscale('log')
    ax1.set_xlabel('Lag time (s)', fontsize=11)
    ax1.set_ylabel('g2(τ)', fontsize=11)
    ax1.set_title(f'Dynamic Speckle g2 Curves\n'
                  f'(n={len(dynamic_df)} dynamic, '
                  f'showing {len(plot_dynamic)})', fontsize=11)
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # ------------------------------------------------------------------ #
    # Panel 2: Non-dynamic speckle g2 curves                              #
    # ------------------------------------------------------------------ #
    ax2 = axes[0, 1]

    plot_nondynamic = nondynamic_df.sample(
        n=min(len(nondynamic_df), max_curves),
        random_state=42) if len(nondynamic_df) > 0 else nondynamic_df

    ci_upper_ref2  = None
    ci_lower_ref2  = None
    ref_lag_times2 = None

    for _, row in plot_nondynamic.iterrows():
        sid  = row['speckle_id']
        data = g2_data.get(sid)
        if data is None:
            continue
        ax2.plot(data['lag_times'], data['g2'],
                 color='#b2bec3', alpha=0.3, linewidth=0.8)
        if ci_upper_ref2 is None:
            ci_upper_ref2  = data['ci_upper']
            ci_lower_ref2  = data['ci_lower']
            ref_lag_times2 = data['lag_times']

    if len(plot_nondynamic) > 0:
        all_g2_nd = [g2_data[sid]['g2']
                     for sid in plot_nondynamic['speckle_id']
                     if sid in g2_data]
        if len(all_g2_nd) > 0:
            mean_g2_nd = np.mean(np.vstack(all_g2_nd), axis=0)
            ax2.plot(ref_lag_times2, mean_g2_nd,
                     color='#636e72', linewidth=2.0,
                     label='Mean g2 (non-dynamic)', zorder=5)

    if ci_upper_ref2 is not None:
        ax2.axhline(np.mean(ci_upper_ref2), color='red',
                    linewidth=1.5, linestyle='--',
                    label=f'{int(ci_level*100)}% CI upper')
        ax2.axhline(np.mean(ci_lower_ref2), color='red',
                    linewidth=1.5, linestyle=':',
                    label=f'{int(ci_level*100)}% CI lower')
        ax2.fill_between(ref_lag_times2,
                         ci_lower_ref2, ci_upper_ref2,
                         alpha=0.15, color='red', label='CI band')

    ax2.axhline(1.0, color='gray', linewidth=1.0,
                linestyle='--', alpha=0.5, label='g2 = 1 (baseline)')
    ax2.set_xscale('log')
    ax2.set_xlabel('Lag time (s)', fontsize=11)
    ax2.set_ylabel('g2(τ)', fontsize=11)
    ax2.set_title(f'Non-Dynamic Speckle g2 Curves\n'
                  f'(n={len(nondynamic_df)} non-dynamic, '
                  f'showing {len(plot_nondynamic)})', fontsize=11)
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.3)

    # ------------------------------------------------------------------ #
    # Panel 3: Contrast distribution                                       #
    # ------------------------------------------------------------------ #
    ax3 = axes[1, 0]

    if len(dynamic_df) > 0:
        ax3.hist(dynamic_df['contrast'],
                 bins=40, alpha=0.7, color='steelblue',
                 label=f'Dynamic (n={len(dynamic_df)})',
                 edgecolor='none', density=True)

    if len(nondynamic_df) > 0:
        ax3.hist(nondynamic_df['contrast'],
                 bins=40, alpha=0.7, color='#b2bec3',
                 label=f'Non-dynamic (n={len(nondynamic_df)})',
                 edgecolor='none', density=True)

    # Mark mean contrast for each group
    if len(dynamic_df) > 0:
        ax3.axvline(dynamic_df['contrast'].mean(),
                    color='navy', linewidth=2, linestyle='--',
                    label=f'Dynamic mean: '
                          f'{dynamic_df["contrast"].mean():.4f}')
    if len(nondynamic_df) > 0:
        ax3.axvline(nondynamic_df['contrast'].mean(),
                    color='#636e72', linewidth=2, linestyle='--',
                    label=f'Non-dynamic mean: '
                          f'{nondynamic_df["contrast"].mean():.4f}')

    ax3.axvline(0.0, color='red', linewidth=1.5,
                linestyle=':', alpha=0.7, label='contrast = 0')
    ax3.set_xlabel('Speckle Contrast  β = g2(τ₁) − 1', fontsize=11)
    ax3.set_ylabel('Density', fontsize=11)
    ax3.set_title('Distribution of Speckle Contrast\n'
                  '(broad = heterogeneous system)', fontsize=11)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Annotation: heterogeneity interpretation
    contrast_std = speckle_props_df['contrast'].std()
    ax3.text(0.97, 0.95,
             f'Overall std: {contrast_std:.4f}\n'
             f'{"Broad → heterogeneous" if contrast_std > 0.05 else "Narrow → homogeneous"}',
             transform=ax3.transAxes,
             ha='right', va='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ------------------------------------------------------------------ #
    # Panel 4: Relaxation time distribution                               #
    # ------------------------------------------------------------------ #
    ax4 = axes[1, 1]

    dyn_with_relax = dynamic_df[dynamic_df['relaxation_time'].notna() &
                                np.isfinite(dynamic_df['relaxation_time'])]

    if len(dyn_with_relax) > 0:
        relax_times = dyn_with_relax['relaxation_time'].values

        # Log-scale histogram is more informative for relaxation times
        log_relax = np.log10(relax_times[relax_times > 0])

        ax4.hist(log_relax, bins=40,
                 color='steelblue', alpha=0.8,
                 edgecolor='none', density=True)

        # Mark mean and median
        mean_log  = np.mean(log_relax)
        median_log = np.median(log_relax)
        ax4.axvline(mean_log, color='navy', linewidth=2,
                    linestyle='--',
                    label=f'Mean: {10**mean_log:.3e} s')
        ax4.axvline(median_log, color='red', linewidth=2,
                    linestyle='--',
                    label=f'Median: {10**median_log:.3e} s')

        # Format x-axis ticks as actual time values
        tick_vals = np.arange(np.floor(log_relax.min()),
                               np.ceil(log_relax.max()) + 1)
        ax4.set_xticks(tick_vals)
        ax4.set_xticklabels([f'\$10^{{{int(v)}}}$' for v in tick_vals])

        relax_std = np.std(log_relax)
        ax4.text(0.97, 0.95,
                 f'n = {len(dyn_with_relax)}\n'
                 f'log₁₀ std: {relax_std:.2f}\n'
                 f'{"Broad → heterogeneous" if relax_std > 0.3 else "Narrow → homogeneous"}',
                 transform=ax4.transAxes,
                 ha='right', va='top', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='wheat',
                           alpha=0.5))
    else:
        ax4.text(0.5, 0.5, 'No dynamic speckles\nwith finite relaxation time',
                 transform=ax4.transAxes,
                 ha='center', va='center', fontsize=12,
                 color='gray')

    ax4.set_xlabel('Relaxation Time τ_r  (log₁₀ scale)', fontsize=11)
    ax4.set_ylabel('Density', fontsize=11)
    ax4.set_title('Distribution of Relaxation Times\n'
                  '(dynamic speckles only, model-free estimate)',
                  fontsize=11)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(figure_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nFigure saved as '{figure_path}'")

def plot_speckle_map(speckle_props_df,
                     image_height=512, image_width=512,
                     full_image=None,
                     figure_path='step2_speckle_map.png'):
    """
    Plot the spatial map of all speckles on the detector, colored by:
        Left   : dynamic vs non-dynamic
        Middle : speckle contrast
        Right  : relaxation time (dynamic only)
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle('Step 2: Spatial Distribution of Speckle Properties',
                 fontsize=14, fontweight='bold')

    dynamic_df    = speckle_props_df[speckle_props_df['is_dynamic']]
    nondynamic_df = speckle_props_df[~speckle_props_df['is_dynamic']]

    for ax in axes:
        if full_image is not None:
            ax.imshow(full_image, cmap='gray', alpha=0.4,
                      vmin=0,
                      vmax=np.percentile(full_image, 99))
        ax.set_xlabel('Pixel X', fontsize=10)
        ax.set_ylabel('Pixel Y', fontsize=10)
        ax.set_xlim(0, image_width)
        ax.set_ylim(image_height, 0)

    # ---------------------------------------- #
    # Left: Dynamic vs non-dynamic             #
    # ---------------------------------------- #
    ax = axes[0]
    if len(nondynamic_df) > 0:
        ax.scatter(nondynamic_df['centroid_x'],
                   nondynamic_df['centroid_y'],
                   s=12, c='#b2bec3', alpha=0.6,
                   label=f'Non-dynamic (n={len(nondynamic_df)})',
                   zorder=2)
    if len(dynamic_df) > 0:
        ax.scatter(dynamic_df['centroid_x'],
                   dynamic_df['centroid_y'],
                   s=20, c='steelblue', alpha=0.8,
                   label=f'Dynamic (n={len(dynamic_df)})',
                   zorder=3)
    ax.set_title('Dynamic vs Non-Dynamic\nSpeckle Classification',
                 fontsize=11)
    ax.legend(fontsize=9, loc='upper right')

    # ---------------------------------------- #
    # Middle: Speckle contrast                 #
    # ---------------------------------------- #
    ax = axes[1]
    contrast_vals = speckle_props_df['contrast'].values
    vmin_c = np.percentile(contrast_vals, 2)
    vmax_c = np.percentile(contrast_vals, 98)

    sc1 = ax.scatter(speckle_props_df['centroid_x'],
                     speckle_props_df['centroid_y'],
                     s=12,
                     c=contrast_vals,
                     cmap='plasma',
                     vmin=vmin_c, vmax=vmax_c,
                     alpha=0.8, zorder=2)
    cbar1 = fig.colorbar(sc1, ax=ax, fraction=0.046, pad=0.04)
    cbar1.set_label('Contrast β = g2(τ₁) − 1', fontsize=9)
    ax.set_title('Speckle Contrast\nSpatial Distribution', fontsize=11)

    # ---------------------------------------- #
    # Right: Relaxation time (dynamic only)    #
    # ---------------------------------------- #
    ax = axes[2]

    dyn_with_relax = dynamic_df[
        dynamic_df['relaxation_time'].notna() &
        np.isfinite(dynamic_df['relaxation_time']) &
        (dynamic_df['relaxation_time'] > 0)]

    # Plot non-dynamic and dynamic without relaxation time in gray
    no_relax = speckle_props_df[
        ~speckle_props_df['speckle_id'].isin(
            dyn_with_relax['speckle_id'])]

    if len(no_relax) > 0:
        ax.scatter(no_relax['centroid_x'],
                   no_relax['centroid_y'],
                   s=8, c='#dfe6e9', alpha=0.4,
                   label='Non-dynamic / no τ_r',
                   zorder=2)

    if len(dyn_with_relax) > 0:
        log_relax = np.log10(dyn_with_relax['relaxation_time'].values)
        vmin_r    = np.percentile(log_relax, 2)
        vmax_r    = np.percentile(log_relax, 98)

        sc2 = ax.scatter(dyn_with_relax['centroid_x'],
                         dyn_with_relax['centroid_y'],
                         s=20,
                         c=log_relax,
                         cmap='RdYlBu_r',
                         vmin=vmin_r, vmax=vmax_r,
                         alpha=0.9, zorder=3)
        cbar2 = fig.colorbar(sc2, ax=ax, fraction=0.046, pad=0.04)

        # Format colorbar ticks as actual time values
        tick_vals = np.linspace(vmin_r, vmax_r, 5)
        cbar2.set_ticks(tick_vals)
        cbar2.set_ticklabels([f'{10**v:.2e} s' for v in tick_vals],
                              fontsize=7)
        cbar2.set_label('Relaxation time τ_r', fontsize=9)

    ax.set_title('Relaxation Time τ_r\n(Dynamic speckles only)',
                 fontsize=11)
    ax.legend(fontsize=9, loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(figure_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nFigure saved as '{figure_path}'")


# =============================================================================
# Save the results from step 2
# =============================================================================



def save_step2_outputs(speckle_props_df, g2_data,
                        output_prefix='step2'):
    """
    Save all Step 2 outputs to disk for use in Step 3.
    """
    # Save main DataFrame
    csv_path = f'{output_prefix}_speckle_properties.csv'
    speckle_props_df.to_csv(csv_path, index=False)
    print(f"  Speckle properties saved -> {csv_path}")

    # Save g2 curves as compressed numpy archive
    npz_path = f'{output_prefix}_g2_curves.npz'
    npz_dict = {}
    for sid, data in g2_data.items():
        npz_dict[f'g2_{sid}']        = data['g2']
        npz_dict[f'lags_{sid}']      = data['lags']
        npz_dict[f'lag_times_{sid}'] = data['lag_times']
        npz_dict[f'ci_upper_{sid}']  = data['ci_upper']
        npz_dict[f'ci_lower_{sid}']  = data['ci_lower']
    np.savez_compressed(npz_path, **npz_dict)
    print(f"  g2 curves saved         -> {npz_path}")

    # Save human-readable summary
    # FIX: explicit UTF-8 encoding, and replaced Unicode symbols
    # with ASCII equivalents to be safe on all platforms
    txt_path = f'{output_prefix}_summary.txt'
    n_total   = len(speckle_props_df)
    n_dynamic = int(speckle_props_df['is_dynamic'].sum())
    dyn_df    = speckle_props_df[
        speckle_props_df['is_dynamic'] &
        speckle_props_df['relaxation_time'].notna() &
        np.isfinite(speckle_props_df['relaxation_time'])]

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 55 + "\n")
        f.write("STEP 2 SUMMARY: SPECKLE PROPERTY EXTRACTION\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Total speckles           : {n_total}\n")
        if n_total > 0:
            f.write(f"Dynamic speckles         : {n_dynamic} "
                    f"({100*n_dynamic/n_total:.1f}%)\n")
        else:
            f.write(f"Dynamic speckles         : 0\n")
        f.write(f"Non-dynamic speckles     : {n_total - n_dynamic}\n\n")

        f.write("--- Contrast (beta = g2(tau_1) - 1) ---\n")
        f.write(f"  Mean  : "
                f"{speckle_props_df['contrast'].mean():.4f}\n")
        f.write(f"  Std   : "
                f"{speckle_props_df['contrast'].std():.4f}\n")
        f.write(f"  Min   : "
                f"{speckle_props_df['contrast'].min():.4f}\n")
        f.write(f"  Max   : "
                f"{speckle_props_df['contrast'].max():.4f}\n\n")

        f.write("--- Relaxation Time tau_r (dynamic only) ---\n")
        if len(dyn_df) > 0:
            f.write(f"  n with finite tau_r : {len(dyn_df)}\n")
            f.write(f"  Mean   : "
                    f"{dyn_df['relaxation_time'].mean():.3e} s\n")
            f.write(f"  Median : "
                    f"{dyn_df['relaxation_time'].median():.3e} s\n")
            f.write(f"  Std    : "
                    f"{dyn_df['relaxation_time'].std():.3e} s\n")
            f.write(f"  Min    : "
                    f"{dyn_df['relaxation_time'].min():.3e} s\n")
            f.write(f"  Max    : "
                    f"{dyn_df['relaxation_time'].max():.3e} s\n")
        else:
            f.write("  No dynamic speckles with finite tau_r found.\n")

        f.write("\n--- Mean Intensity ---\n")
        f.write(f"  Mean  : "
                f"{speckle_props_df['mean_intensity'].mean():.3f}\n")
        f.write(f"  Std   : "
                f"{speckle_props_df['mean_intensity'].std():.3f}\n")

    print(f"  Summary saved           -> {txt_path}")


