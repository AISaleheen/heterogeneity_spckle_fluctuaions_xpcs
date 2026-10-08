#========================================================
#  Step 00: Data Ingestion and Image Reconstruction
# Step 01: Binning photon arrival times
#========================================================
import os
import pandas as pd
from matplotlib import pyplot as plt
from skimage.util import img_as_float, img_as_ubyte
import skbeam.core.utils as utils
import numpy as np
import time

def file_read(folder_path, file_name):
    """
    Read TimePix3 ascii data files with extension .t3pa.
    Parameters:
        folder_path
        file_name
    Returns:
        pandas data frame filedf
    Stores data in a data frame with columns: 
    '1did' : The reduced x and y pixel 
    't': Timestamps in ns. 
    1did = y * 512 + x
    x = 1did % 512 and y = 1did // 512
    Photon arrival time = (t * 200 ) ns
    """
    thefilepath = os.path.join(folder_path, file_name)
    filedf = pd.read_csv(thefilepath, sep = '\t', header = None, skiprows = 1)
    filedf.columns = ['1did', 't']
    return filedf

#=============================================
# Read raw data from the detector
#=============================================
def file_read_chunk(folder_path, file_name, csize):
    """
    Read TimePix3 ascii data files with extension .t3pa.
    The csize parameter allows to read data upto the chunk size to save memory.
    Parameters:
        folder_path
        file_name
        csize : an integer that defines how many photon/entries to be read.
    Returns:
        pandas data frame filedf
    Stores data in a data frame with columns: 
    '1did' : The reduced x and y pixel 
    't': Timestamps in ns. 
    1did = y * 512 + x
    x = 1did % 512 and y = 1did // 512
    Photon arrival time = (t * 200 ) ns
    """
    thefilepath = os.path.join(folder_path, file_name)
    print(f"Loading data file '{file_name}' with csize = {csize:,}...")
    start_time = time.time()
    filedf = pd.read_csv(thefilepath, sep = '\t', header = None, skiprows = 1,
                        chunksize = csize )
    firstchunk = next(filedf)
    #filedf.columns = ['1did', 't']
    firstchunk.columns = ['1did', 't']
    elapsed = time.time() - start_time
    print(f"Loading is complete. ({len(firstchunk):,} rows loaded in {elapsed:.2f} s)")
    return firstchunk

#===============================================================
# Create photon event data frame and reconstruct detector image
#===============================================================

def imager_mask_remover_updated(inputdf, contrast0, contrast1, image_size=512):
    """
    Aggregates photon events and reconstructs detector frame (image).
    
    Parameters:
        inputdf   : DataFrame containing sorted ['1did', 't']
        contrast0 : vmin for detector image display
        contrast1 : vmax for detector image display
        image_size: detector dimensions (default 512x512)
        
    Returns:
        imagedf  : DataFrame with ['1did', 'PhNo', 'ts', 'x', 'y'] (262,144 rows)
        imagedf6 : 512x512 DataFrame (index='x', columns='y') of photon counts
    """
    print(f"Process started, may take a few minutes......")
    total_pixels = image_size * image_size
    
    # 1. Extract underlying NumPy arrays for zero-overhead vector operations
    pids = inputdf['1did'].to_numpy()
    times = inputdf['t'].to_numpy()
    
    # 2. Stable sort by 1did (preserves chronological 't' order within each pixel)
    #order = np.argsort(pids, kind='stable')
    # 2. Multi-key sort: primary sort by 'pids', secondary sort by 'times'
    # In np.lexsort, the primary key goes LAST: (secondary, primary)
    order = np.lexsort((times, pids))
    sorted_pids = pids[order]
    sorted_times = times[order]
    del order  # Free index array memory immediately
    
    # 3. Find pixel boundaries and counts at C-speed
    unique_pids, split_indices, counts = np.unique(
        sorted_pids, return_index=True, return_counts=True
    )
    ends = np.append(split_indices[1:], len(sorted_times))
    
    # 4. Slicing timestamps directly into Python lists
    ts_list = [sorted_times[s:e].tolist() for s, e in zip(split_indices, ends)]
    
    # 5. Build active pixels DataFrame (pixels that received >= 1 photon)
    active_df = pd.DataFrame({
        '1did': unique_pids,
        'PhNo': counts.astype(float),
        'ts': ts_list
    })
    
    # 6. Fast set difference for zero-count pixels (replaces full drop_duplicates)
    zero_pids = np.setdiff1d(np.arange(total_pixels), unique_pids)
    zero_df = pd.DataFrame({
        '1did': zero_pids,
        'PhNo': np.zeros(len(zero_pids), dtype=float)
    })
    
    # 7. Concatenate active + zero-count pixels (zero pixels automatically get ts=NaN)
    imagedf = pd.concat([active_df, zero_df], ignore_index=True)
    imagedf['x'] = imagedf['1did'] % image_size
    imagedf['y'] = imagedf['1did'] // image_size
    
    # 8. Reconstruct 512x512 imaging frame directly (bypasses slow .pivot())
    grid = np.zeros((image_size, image_size), dtype=float)
    # x = 1did % 512 (rows), y = 1did // 512 (columns)
    grid[unique_pids % image_size, unique_pids // image_size] = counts
    
    imagedf6 = pd.DataFrame(
        grid,
        index=pd.Index(range(image_size), name='x'),
        columns=pd.Index(range(image_size), name='y')
    )
    
    # 9. Plot detector view
    plt.figure()
    plt.imshow(imagedf6, vmin=contrast0, vmax=contrast1)
    print(f"Completed.")
    return imagedf, imagedf6

#===========================================================
# View and select a q-ring
#===========================================================
def q_roi_viewer_new(imagefile, center_beam, distance, energy_ev,  rad_ini, rad_fin, agrid_ini, agrid_fin,
                 vm1, vm2):
    """
    Creates a q-ring ROI and shows the ROI on the summed image.
    Parameters:
       imagefile : the summed image obtained from imager_mask_remover
       center_beam: center of the beam
       distance   : sample to detector distance in meter
       energy_ev: X-ray energy in eV
       rad_ini :  r_min
       rad_fin : r_max
       agrid_ini: angle_min
       agrid_fin : angle_max
       vm1   : low contrat  # For plotting
       vm2     : high contrast  # for plotting reduce this value for a higher contrast

    Returns:
       figure with the q-ring shaped ROI shown
                       
    """
    center = center_beam #(201, 213)   #(300,223)  
    binning = 1
    pixel_size = (binning*55.0e-6, binning*55.0e-6)
    sample_distance = 1 #1.0 #m
    wavelength = (1240/energy_ev)  #(1240/706) #1e-9
    shape = (512,512) # imagefile.shape #(512,512)  #Img[0:512,0:512].shape #data[0].shape

    rad_grid = utils.radial_grid(center=center, shape=shape, pixel_size=pixel_size)
    twotheta_grid = utils.radius_to_twotheta(dist_sample=sample_distance, radius=rad_grid)
    q_grid = utils.twotheta_to_q(two_theta=twotheta_grid, wavelength=wavelength)
    angle_grid = utils.angle_grid(center=center, shape=shape, pixel_size=pixel_size)

    #df1 = masktest[0].copy(deep = True)

    df_Img = imagefile #masktest[1] #pd.DataFrame(Img[0:512, 0:512]) # Just the image. Here row and column
    # indices are nothing but the pixel cooridiantes, i.e., x = (0 -511) and y = (0-511)
    #df_q = pd.DataFrame(q_grid)  # converts the xy pixel coordinates into qx qy
    df_rad = pd.DataFrame(rad_grid) # Radial distance from the center
    #df_2theta = pd.DataFrame(twotheta_grid) # Two theta
    df_angle = pd.DataFrame(angle_grid)  # Angle 
    #Don't need all that info
    #tgf = [df_Img, df_q, df_rad, df_2theta, df_angle]
    #full = pd.concat(tgf, axis = 1, keys = ['Pixel', 'qgrid', 'rad', '2theta', 'agrid'],  ignore_index= False)
    tgf = [df_Img, df_rad, df_angle]
    full = pd.concat(tgf, axis = 1, keys = ['Pixel', 'rad', 'agrid'],  ignore_index= False)

    q_ring_df = full['Pixel'][ (full['rad'] > rad_ini) & (full['rad'] < rad_fin)
                       & (full['agrid'] > agrid_ini) & (full['agrid'] < agrid_fin)]

    plt.figure()
    plt.imshow(q_ring_df, vmin = 0, vmax = vm1, alpha = 1)
    plt.imshow(df_Img, vmin = 0, vmax = vm2,  cmap = 'viridis', alpha = 0.6)
    #plt.title("{}{}{}{}{}{}{}".format(rad_ini,'_',rad_fin, '_', agrid_ini,'_', agrid_fin))
    plt.title(f"radial limit:[{rad_ini}-- {rad_fin}]; phi limit: [{agrid_ini}--{agrid_fin}]")


#===========================================================
# Extract data for pixels within the q-ring
#===========================================================


def timestamps_from_ring_movie(datafile, imagefile, vm1, vm2,   center_beam, distance, energy_ev,
                             rad_ini, rad_fin, agrid_ini, agrid_fin, 
                ):
    """
    Extracts the pixel coordinates and photon arrival times within a q-ring shaped ROI.

    Parameters:
        datafile : is the actual dataframe that has all the x,y, and timestamp. 
                   Its shape is (# of pixels, 5 ). 
                   Columns are [1did	PhNo	ts	x	y]
        imagefile : Summed image obtained from imager_mask_remover
        center_beam: center of the beam
        distance   : sample to detector distance in meter
        energy_ev: X-ray energy in eV
        rad_ini :  r_min
        rad_fin : r_max
        agrid_ini: angle_min
        agrid_fin : angle_max
        vm1   : low contrat  # For plotting
        vm2     : high contrast  # for plotting reduce this value for a higher contrast
    Returns:
          new_ts : a pandas Series object containing all the photon arrival times within the ROI.
          new_ring_roi1: a pandas DataFrame containing all the pixels within the ROI 
          and the photon arrival times. It has two columns [1did, ts]. All the photon arrival times
          are grouped by their corresponding pixel 1did. For instance:
          
               1did       t
         0    128928	[1481241, 2767503, 6299251, 12066717, 13550733..]   
    """
    center = center_beam #(201, 213)   #(300,223)  
    binning = 1
    pixel_size = (binning*55.0e-6, binning*55.0e-6)
    sample_distance = distance #1 #1.0 #m
    wavelength = (1240/energy_ev)   #(1240/706) #1e-9
    shape = (512,512) # imagefile.shape #(512,512)  #Img[0:512,0:512].shape #data[0].shape

    rad_grid = utils.radial_grid(center=center, shape=shape, pixel_size=pixel_size)
    twotheta_grid = utils.radius_to_twotheta(dist_sample=sample_distance, radius=rad_grid)
    q_grid = utils.twotheta_to_q(two_theta=twotheta_grid, wavelength=wavelength)
    angle_grid = utils.angle_grid(center=center, shape=shape, pixel_size=pixel_size)

    #df1 = masktest[0].copy(deep = True)

    df_Img = imagefile #masktest[1] #pd.DataFrame(Img[0:512, 0:512]) # Just the image. Here row and column
# indices are nothing but the pixel cooridiantes, i.e., x = (0 -511) and y = (0-511)
    #df_q = pd.DataFrame(q_grid)  # converts the xy pixel coordinates into qx qy
    df_rad = pd.DataFrame(rad_grid) # Radial distance from the center
    #df_2theta = pd.DataFrame(twotheta_grid) # Two theta
    df_angle = pd.DataFrame(angle_grid)  # Angle 
    
    tgf = [df_Img, df_rad, df_angle]
    full = pd.concat(tgf, axis = 1, keys = ['Pixel', 'rad', 'agrid'],  ignore_index= False)
    
    q_ring_df = full['Pixel'][ (full['rad'] > rad_ini) & (full['rad'] < rad_fin)
                       & (full['agrid'] > agrid_ini) & (full['agrid'] < agrid_fin)]    
    plt.figure()
    plt.imshow(q_ring_df, vmin = 0, vmax = vm1, alpha = 1)
    plt.imshow(df_Img, vmin = 0, vmax = vm2,  cmap = 'viridis', alpha = 0.6)
    #plt.title("{}{}{}{}{}{}{}".format(rad_ini,'_',rad_fin, '_', agrid_ini,'_', agrid_fin))
    plt.title(f"radial limit:[{rad_ini}-- {rad_fin}]; phi limit: [{agrid_ini}--{agrid_fin}]")
       
    df_rad_1d = df_rad.stack().reset_index()
    df_rad_1d.columns = ['xrad', 'yrad', 'radval']
    df_rad_1d['id_rad_1d'] = df_rad_1d['yrad']*512 + df_rad_1d['xrad']

    df_angle_1d = df_angle.stack().reset_index()
    df_angle_1d.columns = ['xangle', 'yangle', 'angleval']
    df_angle_1d['id_angle_1d'] = df_angle_1d['yangle']*512 + df_angle_1d['xangle']

    
    df1 = datafile[['1did', 'ts']].dropna()  #df1 = datafile.copy(deep = True)
    
    ## We only need to find which 1did is common between the ROI dataframe and the original
    ### full dataframe. From the common dataframe we can extract all the timestamps
    df_rad_1d_sorted = df_rad_1d[['id_rad_1d', 'radval']].sort_values(by = ['id_rad_1d'], ignore_index = True)
    df_angle_1d_sorted = df_angle_1d[['id_angle_1d', 'angleval']].sort_values(by = ['id_angle_1d'], ignore_index = True)
    
    #combining the radial and angular dataframes with only their 1DID (pixel coordinates) and corre
    ## corresponding radial q and angualr phi values
    df_rad_angle_comb = df_rad_1d_sorted[['id_rad_1d', 'radval']].merge(
                                            df_angle_1d_sorted[['id_angle_1d', 'angleval']],
                                              left_on = 'id_rad_1d', right_on = 'id_angle_1d')
    ## Now based on the RoI condition, select the 1DIDs that fall within the condition.
    ## Note id_rad_1D and id_angle_1D are the same thing. Pixel coords in 1D
    df_rad_angle_ring = df_rad_angle_comb[ (df_rad_angle_comb['radval'] > rad_ini) & (df_rad_angle_comb['radval'] < rad_fin) &\
    (df_rad_angle_comb['angleval'] > agrid_ini) &(df_rad_angle_comb['angleval'] < agrid_fin)]
    ## We only need to know the 1DID, so only taking that info and renaming it to '1did' so that
    ## we can compare it to the full dataframe with timestamps.
    comb = pd.DataFrame(df_rad_angle_ring['id_rad_1d']).rename(columns = {'id_rad_1d':'1did'})
    ##### Now we find entries in the full dataframe that has the IDs identified in the
    ####### previous df (comb). This IDs (1D pixel coordinates) fall within the selected ROI
    ### This merging of two dataframes take only the common element
    new_ring_roi1 = df1.merge(comb, on = '1did', how = 'inner')   
    ### Extract all the timestamps. The timestamps are grouped by their 1DIds so you need to
    ### explode
    
    new_ts = new_ring_roi1['ts'].explode()
   
    

    return new_ts, new_ring_roi1


#===========================================================
# Bin the photon arrival times for each pixel
#===========================================================

def compute_binned_series(q_ring_ts, q_ring_index=1, dt=1e9):
    """
    Bin photon timestamps into discrete time series for each pixel.
    
    Parameters:
        q_ring_ts    : tuple (ts_series, pixel_df) or DataFrame containing ['1did', 'ts']
        q_ring_index : index of DataFrame if q_ring_ts is a tuple (default 1)
        dt           : bin width in native nanoseconds (e.g. 1e9 = 1.0 s, 1e7 = 10 ms)
        
    Returns:
        binned_series_list : list of 1D count arrays (one per pixel)
        bins               : time bin edges in nanoseconds
        pixel_ids          : 1D array of 1did pixel IDs
        n_bins             : number of time bins
    """
    print("--- Binning photon timestamps ---")
    
    # 1. Unpack DataFrame and global timestamp stream
    if isinstance(q_ring_ts, (tuple, list)):
        q_ring_df = q_ring_ts[q_ring_index]
        global_ts_series = q_ring_ts[0]
    else:
        q_ring_df = q_ring_ts
        global_ts_series = None

    # 2. Extract true global min and max across all valid timestamps
    if global_ts_series is not None and len(global_ts_series) > 0:
        # Use flat series across all ring photons for instant boundary lookup
        valid_ts = pd.to_numeric(global_ts_series, errors='coerce').dropna().to_numpy()
        if len(valid_ts) > 0:
            global_min = float(valid_ts.min()) * 200.0
            global_max = float(valid_ts.max()) * 200.0
        else:
            global_min, global_max = 0.0, 0.0
    else:
        # Fallback: inspect non-empty timestamp lists in the DataFrame
        all_mins, all_maxs = [], []
        for ts in q_ring_df['ts']:
            if isinstance(ts, (list, np.ndarray)) and len(ts) > 0:
                arr = np.asarray(ts, dtype=float) * 200.0
                all_mins.append(arr.min())
                all_maxs.append(arr.max())
        global_min = float(min(all_mins)) if all_mins else 0.0
        global_max = float(max(all_maxs)) if all_maxs else 0.0

    duration = global_max - global_min
    
    # 3. Guard against zero or inverted durations
    if duration <= 0:
        print(f"Warning: Calculated duration is <= 0 ({duration:.2f} ns). Setting fallback window.")
        global_max = global_min + dt
        duration = dt

    # 4. Construct bin edges
    n_bins = max(1, int(np.floor(duration / dt)))
    bins = np.linspace(global_min, global_max, n_bins + 1)

    print(f"  Duration         : {duration / 1e9:.2f} seconds")
    print(f"  Number of bins   : {n_bins}")
    print(f"  Pixels to process: {len(q_ring_df)}")

    # 5. Fast histogramming per pixel (handles zero-count/NaN pixels safely)
    binned_series_list = []
    pixel_ids = q_ring_df['1did'].to_numpy()
    ts_column = q_ring_df['ts'].to_numpy()

    for ts in ts_column:
        if isinstance(ts, (list, np.ndarray)) and len(ts) > 0:
            ts_ns = np.sort(np.asarray(ts, dtype=float) * 200.0)
            counts, _ = np.histogram(ts_ns, bins=bins)
        else:
            # Active pixel in ROI received zero photons during this time slice
            counts = np.zeros(n_bins, dtype=int)
            
        binned_series_list.append(counts)

    print(f"  Done. Binned series shape: ({len(binned_series_list)}, {n_bins})")
    
    return binned_series_list, bins, pixel_ids, n_bins







