# ============================================================ 
# Step 4: Fitting To Stretched Exponential KWW Model                             
# ============================================================ 


import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
import warnings

# ============================================================ 
# KWW Stretched exponential function                      
# ============================================================ 

def kww_model(tau, beta, tau_r, gamma):
    """
    Kohlrausch-Williams-Watts (KWW) stretched exponential model.
    
    g2(tau) = 1 + beta * exp(-2 * (tau / tau_r)^gamma)
    
    Parameters:
        tau   : lag time array (physical units, e.g. seconds)
        beta  : contrast (0 < beta <= 1)
        tau_r : relaxation time (same units as tau)
        gamma : stretching exponent (0 < gamma <= 2)
                gamma = 1   : simple exponential (Brownian diffusion)
                gamma = 2   : compressed exponential (ballistic motion)
                0 < gamma<1 : stretched exponential (glassy dynamics)
    """
    return 1.0 + beta * np.exp(-2.0 * (tau / tau_r) ** gamma)


# ============================================================ 
# Compute g2 for each cluster                    
# ============================================================ 

    
def compute_cluster_g2(speckle_props_df, g2_data, cluster_label,
                        dt, max_lag=None):
    """
    Compute cluster-averaged g2 by averaging g2 curves from all
    dynamic speckles belonging to the given GMM cluster.
    
    Averaging g2 curves across many speckles dramatically improves
    SNR compared to any individual speckle curve.
    
    Parameters:
        speckle_props_df : speckle properties DataFrame
        g2_data          : dict of {speckle_id: {'lags': ..., 'g2': ...}}
        cluster_label    : GMM cluster label to average over
        dt               : time bin size in seconds
        max_lag          : maximum lag index to include
    
    Returns:
        lag_times  : physical lag times in seconds
        g2_mean    : cluster-averaged g2
        g2_std     : standard error of g2 across speckles
        g2_matrix  : full matrix of individual g2 curves (n_speckles, n_lags)
        n_speckles : number of speckles averaged
    """
    # Select dynamic speckles in this cluster
    mask = (
        (speckle_props_df['gmm_cluster'] == cluster_label) &
        (speckle_props_df['is_dynamic'] == True)
    )
    cluster_df = speckle_props_df[mask]
    
    if len(cluster_df) == 0:
        return None, None, None, None, 0
    
    # Collect g2 curves
    g2_curves = []
    for sid in cluster_df['speckle_id']:
        if sid in g2_data:
            g2_curve = np.array(g2_data[sid]['g2'], dtype=float)
            if max_lag is not None:
                g2_curve = g2_curve[:max_lag]
            g2_curves.append(g2_curve)
    
    if len(g2_curves) == 0:
        return None, None, None, None, 0
    
    # Trim all curves to the shortest length for safe stacking
    min_len    = min(len(c) for c in g2_curves)
    g2_matrix  = np.vstack([c[:min_len] for c in g2_curves])
    
    # Cluster-averaged g2 and standard error
    g2_mean    = np.mean(g2_matrix, axis=0)
    g2_std     = np.std(g2_matrix,  axis=0) / np.sqrt(len(g2_curves))
    lag_times  = np.arange(min_len) * dt
    n_speckles = len(g2_curves)
    
    return lag_times, g2_mean, g2_std, g2_matrix, n_speckles


# ============================================================ 
# Fit KWW stretched exponential equation to g2                     
# ============================================================ 


def fit_kww_single(lag_times, g2_mean, g2_std=None,
                   beta_init=None, tau_r_init=None, gamma_init=1.0,
                   use_differential_evolution=True):
    """
    Fit KWW model to a single g2 curve.
    
    Uses differential evolution for global optimization first to
    find a good starting point, then refines with curve_fit for
    proper uncertainty estimation from the covariance matrix.
    
    Parameters:
        lag_times   : physical lag times in seconds
        g2_mean     : g2 values to fit
        g2_std      : uncertainties on g2 (used as weights in fit)
                      if None, all points weighted equally
        beta_init   : initial guess for beta (if None, estimated
                      from data)
        tau_r_init  : initial guess for tau_r (if None, estimated
                      from data)
        gamma_init  : initial guess for gamma (default 1.0)
        use_differential_evolution : use global optimizer first
    
    Returns:
        result dict with fitted parameters and uncertainties
    """
    # Skip lag 0 - it is always g2[0] = 1 + beta by definition
    # and including it can bias the fit
    tau_fit = lag_times[1:]
    g2_fit  = g2_mean[1:]
    sigma   = g2_std[1:] if g2_std is not None else None
    
    # Estimate initial parameters from data if not provided
    if beta_init is None:
        beta_init  = float(np.clip(g2_mean[1] - 1.0, 0.01, 1.0))
    if tau_r_init is None:
        # Use the lag where g2 drops to halfway between its max
        # and baseline as the initial tau_r estimate
        g2_half    = 1.0 + (g2_mean[1] - 1.0) / 2.0
        cross_idx  = np.argwhere(g2_fit < g2_half)
        tau_r_init = float(tau_fit[cross_idx[0][0]]) \
                     if len(cross_idx) > 0 \
                     else float(tau_fit[len(tau_fit) // 4])
    
    # Parameter bounds: beta in (0,1], tau_r > 0, gamma in (0,2]
    bounds_lower = [0.0005,  tau_fit[0],    0.1]
    bounds_upper = [1.0,   tau_fit[-1]*5, 2.0]
    bounds       = (bounds_lower, bounds_upper)
    de_bounds    = list(zip(bounds_lower, bounds_upper))
    
    p0 = [beta_init, tau_r_init, gamma_init]
    
    # -------------------------------------------------------- #
    # Stage 1: Global optimization with differential evolution #
    # -------------------------------------------------------- #
    best_p0 = p0
    if use_differential_evolution:
        try:
            def residual(params):
                pred = kww_model(tau_fit, *params)
                if sigma is not None:
                    return np.sum(((g2_fit - pred) / sigma) ** 2)
                return np.sum((g2_fit - pred) ** 2)
            
            de_result = differential_evolution(
                residual,
                bounds=de_bounds,
                seed=42,
                maxiter=1000,
                tol=1e-8,
                workers=1
            )
            if de_result.success:
                best_p0 = de_result.x
        except Exception:
            best_p0 = p0  # fall back to manual initial guess
    
    # -------------------------------------------------------- #
    # Stage 2: Refinement with curve_fit for covariance matrix #
    # -------------------------------------------------------- #
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(
                kww_model,
                tau_fit,
                g2_fit,
                p0=best_p0,
                sigma=sigma,
                absolute_sigma=(sigma is not None),
                bounds=bounds,
                maxfev=10000
            )
        
        perr = np.sqrt(np.diag(pcov))
        
        # Goodness of fit
        g2_pred   = kww_model(tau_fit, *popt)
        residuals = g2_fit - g2_pred
        ss_res    = np.sum(residuals ** 2)
        ss_tot    = np.sum((g2_fit - np.mean(g2_fit)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        
        if sigma is not None:
            chi2_red = np.sum((residuals / sigma) ** 2) \
                       / (len(g2_fit) - 3)
        else:
            chi2_red = ss_res / (len(g2_fit) - 3)
        
        fit_success = True
        
    except (RuntimeError, ValueError) as e:
        # Fit failed to converge
        popt        = np.array([np.nan, np.nan, np.nan])
        perr        = np.array([np.nan, np.nan, np.nan])
        r_squared   = np.nan
        chi2_red    = np.nan
        fit_success = False
    
    return {
        'beta'       : float(popt[0]),
        'tau_r'      : float(popt[1]),
        'gamma'      : float(popt[2]),
        'beta_err'   : float(perr[0]),
        'tau_r_err'  : float(perr[1]),
        'gamma_err'  : float(perr[2]),
        'r_squared'  : float(r_squared),
        'chi2_red'   : float(chi2_red),
        'fit_success': fit_success
    }


# ============================================================ 
# Fit KWW model to cluster-averaged g2 for all GMM clusters.                  
# ============================================================ 

def fit_kww_all_clusters(speckle_props_df, g2_data, dt,
                          max_lag=None, n_clusters=None):
    """
    Fit KWW model to cluster-averaged g2 for all GMM clusters.
    
    Parameters:
        speckle_props_df : speckle properties DataFrame
        g2_data          : dict of g2 curves
        dt               : time bin size in seconds
        max_lag          : maximum lag index
        n_clusters       : number of GMM clusters (if None,
                           inferred from DataFrame)
    
    Returns:
        kww_results : dict of {cluster_label: result_dict}
        cluster_g2  : dict of {cluster_label: {lag_times, g2_mean,
                               g2_std, n_speckles}}
    """
    if n_clusters is None:
        n_clusters = int(speckle_props_df['gmm_cluster'].max()) + 1
    
    kww_results = {}
    cluster_g2  = {}
    
    print("\n--- KWW Fitting by GMM Cluster ---")
    
    for cluster_label in range(n_clusters):
        print(f"\n  Cluster {cluster_label}:")
        
        # Compute cluster-averaged g2
        lag_times, g2_mean, g2_std, g2_matrix, n_speckles = \
            compute_cluster_g2(
                speckle_props_df, g2_data,
                cluster_label, dt, max_lag=max_lag
            )
        
        if lag_times is None or n_speckles < 3:
            print(f"    Skipping: only {n_speckles} speckles")
            kww_results[cluster_label] = None
            continue
        
        print(f"    Speckles averaged : {n_speckles}")
        print(f"    Lag range         : "
              f"{lag_times[1]:.4f} - {lag_times[-1]:.4f} s")
        
        # Store cluster g2
        cluster_g2[cluster_label] = {
            'lag_times' : lag_times,
            'g2_mean'   : g2_mean,
            'g2_std'    : g2_std,
            'n_speckles': n_speckles,
            'g2_matrix' : g2_matrix
        }
        
        # Fit KWW
        result = fit_kww_single(
            lag_times, g2_mean, g2_std=g2_std
        )
        kww_results[cluster_label] = result
        
        if result['fit_success']:
            print(f"    beta      : "
                  f"{result['beta']:.3f} +/- {result['beta_err']:.3f}")
            print(f"    tau_r     : "
                  f"{result['tau_r']:.4f} +/- "
                  f"{result['tau_r_err']:.4f} s")
            print(f"    gamma     : "
                  f"{result['gamma']:.3f} +/- "
                  f"{result['gamma_err']:.3f}")
            print(f"    R²        : {result['r_squared']:.4f}")
            print(f"    chi2_red  : {result['chi2_red']:.4f}")
        else:
            print(f"    Fit failed to converge")
    
    return kww_results, cluster_g2


# ============================================================ 
# Plot the results with the fitted curves                  
# ============================================================ 


def plot_kww_fits(cluster_g2, kww_results, dt, figsize=(18, 5)):
    import matplotlib.pyplot as plt

    n_clusters = len(cluster_g2)
    fig, axes  = plt.subplots(1, n_clusters,
                               figsize=figsize, squeeze=False)

    for idx, (cluster_label, cg2) in enumerate(cluster_g2.items()):
        ax        = axes[0][idx]
        lag_times = cg2['lag_times']
        g2_mean   = cg2['g2_mean']
        g2_std    = cg2['g2_std']

        # Plot individual speckle g2 curves faintly
        for curve in cg2['g2_matrix']:
            ax.plot(lag_times[1:], curve[1:],
                    color='gray', alpha=0.1, linewidth=0.5)

        # Plot cluster mean with error bars
        ax.errorbar(lag_times[1:], g2_mean[1:],
                    yerr=g2_std[1:],
                    fmt='o', markersize=3,
                    color='black', zorder=5)

        # Plot KWW fit
        result = kww_results.get(cluster_label)
        if result is not None and result['fit_success']:
            tau_fine = np.linspace(lag_times[1], lag_times[-1], 500)
            g2_fit   = kww_model(tau_fine,
                                  result['beta'],
                                  result['tau_r'],
                                  result['gamma'])
            ax.plot(tau_fine, g2_fit,
                    color='red', linewidth=2, label='KWW fit')

            # Mark tau_r with vertical dashed line
            ax.axvline(result['tau_r'],
                       color='red', linestyle='--',
                       alpha=0.5, linewidth=1)

        # Baseline reference
        ax.axhline(1.0, color='blue', linestyle=':',
                   linewidth=1, alpha=0.7)

        ax.set_xscale('log')
        ax.set_xlabel('Lag time τ (s)', fontsize=10)
        ax.set_ylabel('g2(τ)',          fontsize=10)

        # ---------------------------------------------------- #
        # Fixed y-axis: use fixed limits based on the actual   #
        # range of the cluster mean, not individual curves     #
        # which can be very noisy                              #
        # ---------------------------------------------------- #
        g2_min = np.min(g2_mean[1:])
        g2_max = np.max(g2_mean[1:])
        margin = (g2_max - g2_min) * 0.5
        ax.set_ylim(max(0.995, g2_min - margin),
                    g2_max + margin)

        # Title carries cluster info so legend can be minimal
        r2_str = (f"R²={result['r_squared']:.3f}"
                  if result is not None and result['fit_success']
                  else "fit failed")
        ax.set_title(f'Cluster {cluster_label}\n'
                     f'{cg2["n_speckles"]} speckles  {r2_str}',
                     fontsize=10)

        # ---------------------------------------------------- #
        # Compact text box instead of legend                   #
        # Avoids overlap entirely                              #
        # ---------------------------------------------------- #
        if result is not None and result['fit_success']:
            textstr = (f"β={result['beta']:.3f}\n"
                       f"τ_r={result['tau_r']:.2f} s\n"
                       f"γ={result['gamma']:.2f}")
            ax.text(0.97, 0.97, textstr,
                    transform=ax.transAxes,
                    fontsize=8,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='white',
                              alpha=0.8,
                              edgecolor='gray'))

        ax.grid(True, alpha=0.3)

    plt.suptitle('KWW Fits to Cluster-Averaged g2 Curves',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return fig

def plot_kww_summary(kww_results, figsize=(12, 4)):
    import matplotlib.pyplot as plt

    clusters   = []
    betas      = []
    tau_rs     = []
    gammas     = []
    beta_errs  = []
    tau_r_errs = []
    gamma_errs = []

    for cluster_label, result in kww_results.items():
        if result is not None and result['fit_success']:
            clusters.append(cluster_label)
            betas.append(result['beta'])
            tau_rs.append(result['tau_r'])
            gammas.append(result['gamma'])
            beta_errs.append(result['beta_err'])
            tau_r_errs.append(result['tau_r_err'])
            gamma_errs.append(result['gamma_err'])

    if len(clusters) == 0:
        print("No successful fits to plot.")
        return None

    fig, axes  = plt.subplots(1, 3, figsize=figsize)
    x          = np.arange(len(clusters))
    x_labels   = [f'Cluster {c}' for c in clusters]

    # -------------------------------------------------------- #
    # Beta - rescale y-axis to actual data range               #
    # since all betas are near 0, not near 1                   #
    # -------------------------------------------------------- #
    axes[0].bar(x, betas, yerr=beta_errs,
                capsize=5, color='steelblue', alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(x_labels, rotation=15, ha='right')
    axes[0].set_ylabel('β (contrast)', fontsize=11)
    axes[0].set_title('Speckle Contrast β', fontsize=11)
    # Set y-axis to data range with a small margin
    beta_max = max(betas) + max(beta_errs) if beta_errs else max(betas)
    axes[0].set_ylim(0, beta_max * 2.5)
    # Add value labels on bars
    for i, (v, e) in enumerate(zip(betas, beta_errs)):
        axes[0].text(i, v + e + beta_max * 0.1,
                     f'{v:.4f}',
                     ha='center', va='bottom', fontsize=8)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Tau_r
    axes[1].bar(x, tau_rs, yerr=tau_r_errs,
                capsize=5, color='darkorange', alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(x_labels, rotation=15, ha='right')
    axes[1].set_ylabel('τ_r (s)', fontsize=11)
    axes[1].set_title('Relaxation Time τ_r', fontsize=11)
    for i, (v, e) in enumerate(zip(tau_rs, tau_r_errs)):
        axes[1].text(i, v + e + 5,
                     f'{v:.1f}',
                     ha='center', va='bottom', fontsize=8)
    axes[1].grid(True, alpha=0.3, axis='y')

    # Gamma
    axes[2].bar(x, gammas, yerr=gamma_errs,
                capsize=5, color='forestgreen', alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(x_labels, rotation=15, ha='right')
    axes[2].set_ylabel('γ (stretching exponent)', fontsize=11)
    axes[2].set_title('Stretching Exponent γ', fontsize=11)
    axes[2].set_ylim(0, 2.2)
    axes[2].axhline(1.0, color='blue',   linestyle='--',
                    alpha=0.5, label='γ=1 Brownian')
    axes[2].axhline(2.0, color='red',    linestyle='--',
                    alpha=0.5, label='γ=2 Ballistic')
    axes[2].axhline(0.5, color='purple', linestyle='--',
                    alpha=0.5, label='γ=0.5 Glassy')
    for i, (v, e) in enumerate(zip(gammas, gamma_errs)):
        axes[2].text(i, v + e + 0.05,
                     f'{v:.2f}',
                     ha='center', va='bottom', fontsize=8)
    axes[2].legend(fontsize=8, loc='upper right')
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.suptitle('KWW Fitted Parameters by GMM Cluster',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return fig

