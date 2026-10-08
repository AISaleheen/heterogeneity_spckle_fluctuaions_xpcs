## Summary

X-ray Photon Correlation Spectroscopy (XPCS) provides a window into temporal fluctuations. Across timescales ranging from milliseconds to hundreds of seconds, XPCS probes domain dynamics in magnetic and quantum materials. In coherent scattering experiments, speckle patterns arise due to the  interference of coherently scattered X-rays, whose relative phase shifts encode the domain morphology of the sample. Consequently, fluctuaions in speckle patterns represent the domain dynamics in the system. Usually, the dynamics at a given length scale is calculated by averaging the intensity autocorrelation function $g_2(\tau)$ azimuthally across a static $q$-ring. However, in many systems, the individual speckles  within that ring often exhibit diverse spatio-temporal behavior. Proving such dynamic heterogeneity requires segmenting and tracking individual speckles. 

This project implements an end-to-end unsupervised machine learning and signal processing pipeline to detect, isolate, and cluster speckles by their dynamical signatures. The raw data consists of continuous photon event streams from an MCP/Timepix3 detector recorded during a Coherent Resonant Soft X-ray Scattering (CRSXS) experiment on an amorphous Fe/Ge system at ALS beamline 7.0.1.1 (COSMIC). Key highlights include:

* **Scale:** Ingests and aggregates $\approx 10^8$ raw Timepix3 photon event timestamps at single-photon resolution.
* **Segmentation:** Isolates coherent speckles from diffuse background scatter using an unsupervised consensus ensemble (1D GMM, K-Means) and local spatial contrast filters.
* **Statistical Rigor:** Validates genuine physical dynamics using a 200-iteration permutation bootstrap hypothesis test (95% CI).
* **Heterogeneity Discovery:** Identifies distinct coexisting dynamical regimes via Gaussian Mixture Models (BIC-selected) and extracts physical relaxation mechanics using cluster-averaged KWW stretched exponential fits.

## **Pipeline Architecture** 

Raw Photon Timestamps (Timepix3: 1did, t)

**Step 0: Data Ingestion & Image Reconstruction**
- Collect photon timestamps at each pixel and sort chronologically via vectorized multi-key sort
- Calculate total number of photons at each pixel
- Reconstruct the 512 × 512 detector frame
- Isolate azimuthal q-ring (selects the characteristic scattering length scale of interest)

**Step 1: Feature Engineering & Speckle Segmentation**
- Bin photon counts at each pixel into 1 s discrete time bins (bin duration can be tuned)
- Calculate local contrast (mean count / local background mean, neighborhood radius r = 5 px) and local statistics
- Plot distributions of extracted features; local contrast provides the primary discriminating signal
- Unsupervised clustering & candidate segmentation:
    - 1D Gaussian Mixture Model (BIC model selection)
    - 2D K-Means Model (local contrast vs. peak-to-background ratio)
    - Hard contrast thresholding
    - Multi-model consensus voting
- Evaluate and select high-confidence speckle pixels for downstream analysis

**Step 2: Speckle Properties & Autocorrelation**
- Group speckle pixels into contiguous objects via 8-connected component labeling
- Size filtering based on beamline geometry: $8 \le \text{area} \le 50$ px  (removes noise hits and merged multi-speckle aggregates)
- Sum binned time series across member pixels within each speckle to maximize SNR prior to correlation
- Compute intensity autocorrelation g₂(τ) via FFT-accelerated statsmodels ACF
- 200-sample permutation bootstrap hypothesis test which shuffles time bins to establish an empirical 95% null CI, determining whether genuine temporal correlation exists above the experimental noise floor.
- Extract model-free crossing relaxation time $\tau_r$
- Classify and plot statistically significant dynamic speckles against non-dynamic baselines

**Step 3: Dynamic Clustering & Heterogeneity Mapping**
- 5D feature extraction per dynamic speckle: contrast $\beta$, mean $g_2$ decay, mean $g_2$ long baseline, decay slope, relaxation time $\tau_r$
- Feature standardization via StandardScaler
- Gaussian Mixture Model (GMM) clustering with BIC component selection
- DBSCAN cross-check to evaluate density boundaries and dynamical continuity
- Compute cluster-averaged g₂(τ) curves
- Generate detector spatial maps displaying cluster assignments, relaxation timescales, and assignment confidence

**Step 4: Fitting To Stretched Exponential Model**
- Average $g_2(\tau)$ curves across all speckles per cluster to suppress single-speckle noise
- Model: Kohlrausch-Williams-Watts (KWW) stretched exponential function: $g_2(\tau) = 1 + \beta\, exp[-2(\tau/\tau_r)^{\gamma}]$        
- Differential Evolution global optimization to locate robust initial parameter values
- Levenberg-Marquardt non-linear least squares refinement for parameter covariance estimation
- Extract physical parameters: relaxation time $\tau_r$, speckle contrast $\beta$, and stretching exponent $\gamma$
- Plot all the results, fitted curves, identify dynamic regimes, etc.

## Key Results & Figures

| Step | Analysis Deliverable | Description |
| :--- | :--- | :--- |
| **01** | `step1_streamlined_speckle_finding.png` | Consensus segmentation comparing intensity thresholding, 1D GMM local contrast, and 2D K-Means. |
| **02** | `step2_speckle_properties.png` | Individual speckle autocorrelation curves, SNR distributions, and 95% bootstrap dynamic classifications. |
| **03** | `step3_heterogeneity.png` / `_map.png` | BIC model selection curve, GMM confidence distributions, DBSCAN noise cross-check, and spatial relaxation time maps. |
| **04** | `step4_kww_fits.png` | Cluster-averaged $g_2(\tau)$ curves fitted to the KWW model  revealing distinct slow, intermediate, and fast kinetic regimes. |

---

## Repository Structure

```text

├── README.md
├── requirements.txt
├── 250Kanalysis_notebook.ipynb        # Fully rendered production workflow (247M events)
├── 200Kanalysis_notebook.ipynb        # Comparative temperature dataset workflow
├── data/
│   └── SampleData.csv                 # Lightweight reproducible benchmark slice (~3.6 s)
├── outputs/                           # Generated analytical figures and exported matrices
│   ├── step1_streamlined_speckle_finding.png
│   ├── step2_speckle_properties.png
│   ├── step3_heterogeneity.png
│   ├── step3_heterogeneity_map.png
│   ├── step4_kww_fits.png
│   ├── step1_speckle_pixels_final.csv
│   └── step2_speckle_properties.csv
└── xpcs/                              # Core algorithmic modules
    ├── __init__.py
    ├── data_reader_roi.py             # Event-stream parsing, ASIC indexing, and binning
    ├── speckle.py                     # Contrast enhancement and connected components
    ├── speckle_g2.py                  # Autocorrelation and bootstrap dynamics testing
    ├── clustering.py                  # GMM, BIC, and DBSCAN heterogeneity clustering
    └── fitting.py                     # Kohlrausch–Williams–Watts (KWW) nonlinear fitting
```

## Installation & Environment Setup

```bash
# Clone the repository
git clone [https://github.com/AISaleheen/heterogeneity_spckle_fluctuaions_xpcs.git](https://github.com/AISaleheen/heterogeneity_spckle_fluctuaions_xpcs.git)
cd heterogeneity_spckle_fluctuaions_xpcs

# Option A: Conda (Recommended for scikit-beam Cython extensions)
conda create -n xpcs_env python=3.10 -y
conda activate xpcs_env
conda install -c conda-forge scikit-beam -y
pip install -r requirements.txt

# Option B: Standard virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
## Running the Analysis

Open JupyterLab and launch analysis_notebook.ipynb.

The notebook is pre-configured to execute end-to-end on data/SampleData.csv out of the box, demonstrating the full ingestion, segmentation, and clustering sequence without external dependencies.

