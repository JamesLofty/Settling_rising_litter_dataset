# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 11:46:11 2025

@author: jlofty
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 08:36:36 2025

@author: jameslofty
# %%
"""

# ==============================================================
#  IMPORTS
# ==============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re
from scipy import stats
from scipy.signal import welch
import shutil
from scipy.ndimage import gaussian_filter1d

# ==============================================================
#  TRAJECTORY NORMALIZATION UTIL
# ==============================================================
def normalize_trajectory(df, col1, col2, col3):
    df = df.copy()
    for col in [col1, col2, col3]:
        if df[col].isna().all():
            raise ValueError(f"Column {col} is all NaN.")
        df[col] = df[col].interpolate(limit_direction='both')  # fill small gaps

    # Normalize X, Y
    df[col1] = df[col1] - df[col1].iloc[0]
    df[col2] = df[col2] - df[col2].iloc[0]

    # Normalize Z safely
    start, end = df[col3].iloc[0], df[col3].iloc[-1]
    if pd.isna(start) or pd.isna(end):
        raise ValueError(f"Z normalization failed: start or end NaN")
    if end < start:
        df[col3] = df[col3] - start
    else:
        df[col3] = df[col3] - start - 50
    return df


# ==============================================================
#  EXCLUSION RULES PER CATEGORY
# ==============================================================
exclusion_rules = {
    "002Megered_coordinates/Candy, snack and crisps packaging": {},
    }

Hz_cut = {

      "002Megered_coordinates/Candy, snack and crisps packaging":0.5,


}

# ==============================================================
#  INPUT DATASET SELECTION
# ==============================================================

file_path = "002Megered_coordinates/Candy, snack and crisps packaging/*"



# ==============================================================
#  LOAD ALL MATCHING EXCEL FILES
# ==============================================================
files = glob.glob(file_path)
print(len(files))

base_path = file_path.rsplit("/*.xlsx", 1)[0]

# ==============================================================
#  FIRST DELETE EVERYTHING IN SAVING FOLDER
# ==============================================================
folder = '003Cleaned_coordinates/'
parts = os.path.normpath(file_path).split(os.sep)
subfolder = parts[1]

# ==============================================================
#  Extract Hz cut limit 
# ==============================================================

hz_current = Hz_cut.get(base_path)

if hz_current is None:
    for key in Hz_cut.keys():
        if key in base_path or base_path in key:
            Hz_cut_lim = Hz_cut[key]
            break
# ==============================================================
#  READ, TAG, AND APPLY EXCLUSION RULES
# ==============================================================
all_data = []
for file in files:
    df = pd.read_excel(file)
    file_id = file

    id_number = int(file.split("_")[-1].split(".")[0])  # Example: extract numeric ID from filename
    base_path = str(file_path).replace("/*", "").strip()
    if base_path in exclusion_rules and id_number in exclusion_rules[base_path]:
        print(f"Skipping ID {id_number} from file {file} due to exclusion rule.")
        continue

    df['file_id'] = file
    df['id_number'] = id_number
    all_data.append(df)
   
# ==============================================================
#  CONCAT ALL DATA & BASIC TIME COLUMN
# ==============================================================
all_data = pd.concat(all_data, ignore_index=True)

ID = all_data['file_id']
framerate = 1/60
all_data["tp_seconds"] = all_data["tp"]*framerate

target_file = "002Megered_coordinates/Foam plastic pieces 2.5-50 cm/ID_15.xlsx"


# %%
# ==============================================================
#  PROCESS EACH FILE SEPARATELY

for file in sorted(all_data['file_id'].unique()):
    
    
    # if file != target_file:
    #     continue
    
    df = all_data[all_data['file_id'] == file].copy()
    # ----------------------------------------------------------
    # 2) cut trajectories outside tank bounds
    # ----------------------------------------------------------
    
    df['center_X_raw'] = df['center_X']
    df['center_Y_raw'] = df['center_Y']
    df['center_Z_raw'] = df['center_Z']
    df = normalize_trajectory(df, 'center_X_raw', 'center_Y_raw', 'center_Z_raw')

    plt.figure(figsize = (3, 6))
    plt.scatter(df['center_X_raw'], df['center_Z_raw'] )
    plt.scatter(df['center_Y_raw'], df['center_Z_raw'] )

    df_cut = df[
    (df['center_X'].between(0, 126)) &
    (df['center_Y'].between(0, 126)) &
    (df['center_Z'].between(0, 126))
    ].copy()
    
    df = df_cut
    
    if df.empty or len(df) < 10:
        print("Empty or too small DataFrame — skipping.")
    else:
        # Compute displacement safely
        dis = df['center_Z'].iloc[0] - df['center_Z'].iloc[-1]
        print(f"Z displacement: {dis:.2f}")
    
        if abs(dis) < 5:
            print("Z displacement < 5 cm — skipping.")
        else:
            # ======================================================
            #  IQR-BASED SPATIAL FILTER ON X/Y
            # ======================================================
        
 
            
            Q1_x, Q3_x = np.percentile(df['center_X'], [25, 75])
            IQR_x = Q3_x - Q1_x
            
            Q1_y, Q3_y = np.percentile(df['center_Y'], [25, 75])
            IQR_y = Q3_y - Q1_y
            
            thres = 1.5
            
            df_filtered = df[
                (df['center_X'] >= Q1_x - thres * IQR_x) & (df['center_X'] <= Q3_x + thres * IQR_x) &
                (df['center_Y'] >= Q1_y - thres * IQR_y) & (df['center_Y'] <= Q3_y + thres * IQR_y) 
            ]
    
            print("IQR filtering complete.")
            
        
            # %%
            # ======================================================
            #  NORMALISE TRAJECTORY TO ZERO
            # ======================================================
    
            if df_filtered.empty:
                print("empty df")
            else:
                df_filtered = normalize_trajectory(df_filtered, 'center_X', 'center_Y', 'center_Z')
        
    # %%
                # ==================================================
                #  PSD / WELCH + GLOBAL SNR ESTIMATE
                # ==================================================
                fs = 60.0
                fc_target = Hz_cut_lim  # desired cutoff in Hz
                
                # Compute N corresponding to ~2 Hz cutoff
                N_base = int(round(0.443 * fs / fc_target))
                if N_base % 2 == 0:
                    N_base += 1  # force odd for symmetric window
                
                # Same N for all axes, since cutoff is fixed
                N_x_base = N_y_base = N_z_base = N_base
            
                x = df_filtered['center_X'].to_numpy()
                y = df_filtered['center_Y'].to_numpy()
                z = df_filtered['center_Z'].to_numpy()
                
                f_x, Pxx_x = welch(x, fs=fs, nperseg=min(1024, len(x)))
                f_y, Pxx_y = welch(y, fs=fs, nperseg=min(1024, len(y)))
                f_z, Pxx_z = welch(z, fs=fs, nperseg=min(1024, len(z)))
                
                       
                fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharey=True)
                fig.suptitle(f"{file} — PSD and SNR", fontsize=14)
                
                for ax, (f, Pxx, label) in zip(
                    axes,
                    [
                        (f_x, Pxx_x, "X"),
                        (f_y, Pxx_y, "Y"),
                        (f_z, Pxx_z, "Z"),
                    ],
                ):
                    ax.semilogy(f, Pxx, label=f"PSD {label}")
                    ax.axvline(fc_target, color="r", linestyle="--", label=f"Cutoff = {fc_target} Hz")
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                    ax.set_xlabel("Frequency [Hz]")
                    ax.set_ylabel("Power Spectral Density")
                    ax.legend()
                    ax.grid(True, which="both")
                
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                plt.show()
                
                # ==================================================
                #  GAUSSIAN SMOOTHING (PER AXIS)
                # ==================================================
                from statsmodels.nonparametric.smoothers_lowess import lowess
                from scipy.ndimage import gaussian_filter1d

                # LOWESS smoothing with equivalent cutoff window
                N_total = len(df_filtered)
                frac_equiv = N_base / N_total  # same effective window size as Gaussian smoothing
                frac_equiv = min(1.0, max(frac_equiv, 3/N_total))  # clamp to valid range
                sigma_x = N_x_base / 2.355   # 2.355 ≈ convert FWHM → σ (width in frames)
                sigma_y = N_y_base / 2.355
                sigma_z = N_z_base / 2.355
                

                def smooth_const_velocity(x, sigma, dt, truncate=3.0, pad_multiplier=2):
                    x = np.asarray(x)
                    n = len(x)
                    
                    radius = int(truncate * sigma + 0.5)
                    pad = pad_multiplier * radius
                
                    if n <= 2 or radius < 1 or pad < 1:
                        return gaussian_filter1d(x, sigma=sigma, truncate=truncate)
                
                    v0 = (x[1] - x[0]) / dt
                    vN = (x[-1] - x[-2]) / dt
                    
                    t_left = -np.arange(pad, 0, -1) * dt
                    t_right = np.arange(1, pad + 1) * dt
                    
                    left = x[0] + v0 * t_left
                    right = x[-1] + vN * t_right
                    
                    x_pad = np.concatenate([left, x, right])
                    y_pad = gaussian_filter1d(x_pad, sigma=sigma, truncate=truncate)
                    y = y_pad[pad:-pad]
                    return y
                
                dt = 60  # your frame interval
                
                
                df_filtered['center_X_smooth1'] = smooth_const_velocity(
                    df_filtered['center_X'].to_numpy(), sigma_x, dt
                )
                df_filtered['center_Y_smooth1'] = smooth_const_velocity(
                    df_filtered['center_Y'].to_numpy(), sigma_y, dt
                )
                df_filtered['center_Z_smooth1'] = smooth_const_velocity(
                    df_filtered['center_Z'].to_numpy(), sigma_z, dt
                )
                
                df_filtered = normalize_trajectory(df_filtered, 'center_X_smooth1', 'center_Y_smooth1', 'center_Z_smooth1')
                # 
              
                df_filtered = df_filtered[
                    (df_filtered['center_Z_smooth1'].between(-50, 0))
                ].copy()

    # %%
                # ==================================================
                #  RAW vs SMOOTHED TRAJECTORY PLOTS
                # ==================================================
                fig, axes = plt.subplots(1, 2, figsize=(13, 10), sharey=True, sharex=True)
                fig.suptitle(file, fontsize=14)
                
                # --- Subplot 1: Raw Data ---
            
                
                axes[0].scatter(df_filtered['center_X'], df_filtered['center_Z'], c='r', label="X")
                axes[0].scatter(df_filtered['center_Y'], df_filtered['center_Z'], c='k', label="Y")
                axes[0].set_title("filtered  Data")
                axes[0].legend()
                
                axes[1].scatter(df_filtered['center_X_smooth1'], df_filtered['center_Z_smooth1'], c='r', label="X")
                axes[1].scatter(df_filtered['center_Y_smooth1'], df_filtered['center_Z_smooth1'], c='k', label="Y")
                axes[1].set_title("Smoothed")
                axes[1].legend()
            
                axes[1].set_xlim(-25, 25)
                axes[1].set_ylim(-60, 10)
    
                plt.show()
                
                if df_filtered.empty:
                    print("empty df")
                else:
                    
    # %%
                    # ==================================================
                    #  SIGNAL TO NOISE RATIO
                    # ==================================================
             
                    # --- Compute residuals (noise estimate) ---
                    df_filtered['res_X'] = df_filtered['center_X'] - df_filtered['center_X_smooth1']
                    df_filtered['res_Y'] = df_filtered['center_Y'] - df_filtered['center_Y_smooth1']
                    df_filtered['res_Z'] = df_filtered['center_Z'] - df_filtered['center_Z_smooth1']
                    
                    # --- Rolling SNR function ---
                    def rolling_snr(signal, noise, window):
                        signal_power = signal**2
                        noise_power = noise**2
                        snr_linear = (signal_power.rolling(window).mean() /
                                      noise_power.rolling(window).mean())
                        return snr_linear
                    
                    # --- Choose window (e.g., 1 second) ---
                    window_sec = 1
                    window_samples = int(window_sec * fs)
                    
                    # --- Compute rolling SNR for each axis ---
                    df_filtered['SNR_X'] = rolling_snr(df_filtered['center_X_smooth1'], df_filtered['res_X'], window_samples)
                    df_filtered['SNR_Y'] = rolling_snr(df_filtered['center_Y_smooth1'], df_filtered['res_Y'], window_samples)
                    df_filtered['SNR_Z'] = rolling_snr(df_filtered['center_Z_smooth1'], df_filtered['res_Z'], window_samples)
                    
                    # --- Compute average SNRs (ignore NaNs) ---
                    snr_x_mean = np.nanmean(df_filtered['SNR_X'])
                    snr_y_mean = np.nanmean(df_filtered['SNR_Y'])
                    snr_z_mean = np.nanmean(df_filtered['SNR_Z'])
                    
                    print(f"Mean Signal-to-Noise Ratio X: {snr_x_mean:.2f} dB")
                    print(f"Mean Signal-to-Noise Ratio Y: {snr_y_mean:.2f} dB")
                    print(f"Mean Signal-to-Noise Ratio Z: {snr_z_mean:.2f} dB")
                    
                    # --- Plot Rolling SNRs ---
                    time = np.arange(len(df_filtered)) / fs
                    
    # %%
                    
                    if df_filtered.empty or len(df_filtered) < 10:
                        print("Empty or too small df")
                    else:
                        dis = df_filtered['center_Z'].iloc[0] - df_filtered['center_Z'].iloc[-1]
                        print(f"Z displacement: {dis:.2f}")
                    
                        if abs(dis) < 5:
                            print("Z displacement less than 5 cm")
                        else:
                           
                    
                            fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
                            fig.suptitle(f"{file} — Rolling SNR (window={window_sec}s)", fontsize=14)
                        
                            for ax, col, label in zip(axes, ['SNR_X', 'SNR_Y', 'SNR_Z'], ['X', 'Y', 'Z']):
                                ax.plot(time, df_filtered[col], label=f"SNR {label}")
                                ax.axhline(10, color='r', linestyle='--', label="10 dB threshold")
                                ax.set_ylabel("SNR [dB]")
                                ax.set_title(f"{label}-axis")
                                ax.grid(True)
                                ax.legend(loc="upper right")
                            
                            axes[-1].set_xlabel("Time [s]")
                            plt.tight_layout(rect=[0, 0, 1, 0.95])
                            plt.show()
                        
                        
                # %%
               
                            # ==================================================
                            #  VELOCITY / ANGLE DERIVATION
                            # ==================================================
                
                            def process_particle_data(df, dt=framerate, frac=0.5):
                        
                                if len(df) < 2:
                                    print("Warning: Not enough points to compute gradients. Skipping this file.")
                                    return pd.DataFrame()  # or return df as-is
                            
                                # Remove duplicate or zero-diff time points before anything else
                                df = df[df['tp_seconds'].diff() != 0].copy()
                                df.reset_index(drop=True, inplace=True)
                                if len(df) < 2:
                                    print("Skipping: not enough timepoints after cleaning.")
                                    return pd.DataFrame()
                            
                                #################################################### 
                                # --- Compute velocities ---
                                ####################################################         
                                for axis in ['X', 'Y', 'Z']:
                                    
                                    raw = df[f"center_{axis}"]
                                    smoothed = df[f"center_{axis}_smooth1"]
                                
                                    # compute velocity from BOTH raw and smoothed
                                    v_raw = np.gradient(raw, df["tp_seconds"])
                                    v_smooth = np.gradient(smoothed, df["tp_seconds"])
                                
                                    # flip sign
                                    v_raw = -v_raw
                                    v_smooth = -v_smooth
                                
                                    df[f"v{axis.lower()}_raw"] = v_raw
                                    df[f"v{axis.lower()}_smooth"] = v_smooth
                                    
                                    
                                ####################################################
                                # --- Filter smoothed velocities using raw percentiles ---
                                ####################################################
                                
                                for axis in ['x', 'y', 'z']:
                                    v_raw = df[f"v{axis}_raw"]
                                    v_smooth = df[f"v{axis}_smooth"]
                                
                                    # Compute thresholds only from *finite* raw values
                                    q1 = np.nanpercentile(v_raw, 10)
                                    q3 = np.nanpercentile(v_raw, 90)
                                
                                    # Mark smoothed values outside [Q1, Q3] as NaN
                                    mask = (v_smooth < q1) | (v_smooth > q3)
                                    df.loc[mask, f"v{axis}_smooth"] = np.nan
                                                            
                                                            
                                #################################################### 
                                # ---  average angle from 3 cameras ---
                                #################################################### 
                                def fold_angle(x):
                                    x = abs(x) % 180
                                    return x if x <= 90 else 180 - x
                                
                                # Fold each camera angle
                                for cam in [1, 2, 3]:
                                    col = f'angle_cam{cam}'
                                    df[col] = df[col].apply(fold_angle)
                                
                                # Average of folded angles
                                angle = df[['angle_cam1','angle_cam2','angle_cam3']].mean(axis=1)
                                
                                # Smooth the averaged angle
                                df['angle'] = smooth_const_velocity(angle, sigma_x, dt)
                                
                                # Enforce final ≤ 90° constraint
                                df['angle'] = df['angle'].apply(fold_angle)
                                
                                return df
                            
                            df_filtered = process_particle_data(df_filtered)
                            if df_filtered.empty:
                                print(f"Skipping file {file} - insufficient data after processing.")
                                continue
                            
                            
                         
                            
                            #%%
                            # ==================================================
                            #  MULTI-PANEL DIAGNOSTIC PLOT
                            # ==================================================
                            fig, axes = plt.subplots(1, 5, figsize=(15, 7), sharey=False)  # 1 row, 5 columns
                            fig.suptitle(file, fontsize=16)  
                            # Column 1: Velocity X
                            axes[0].scatter(df_filtered['vx_smooth'], df_filtered['center_Z_smooth1'], c="tab:red", label="vx smooth")
                            axes[0].scatter(df_filtered['vx_raw'], df_filtered['center_Z'], c="tab:red", alpha=0.3, label="vx raw")
                            axes[0].set_xlabel("Velocity X")
                            axes[0].set_ylabel("Z (cm)")
                            axes[0].set_title("Velocity in X")
                            axes[0].legend()
                            
                            # Column 2: Velocity Y
                            axes[1].scatter(df_filtered['vy_smooth'], df_filtered['center_Z_smooth1'], c="tab:orange", label="vy smooth")
                            axes[1].scatter(df_filtered['vy_raw'], df_filtered['center_Z'], c="tab:orange", alpha=0.3, label="vy raw")
                            axes[1].set_xlabel("Velocity Y")
                            axes[1].set_title("Velocity in Y")
                            axes[1].legend()
                            
                            # Column 3: Velocity Z
                            axes[2].scatter(df_filtered['vz_smooth'], df_filtered['center_Z_smooth1'], c="tab:blue", label="vz smooth")
                            axes[2].scatter(df_filtered['vz_raw'], df_filtered['center_Z'], c="tab:blue", alpha=0.3, label="vz raw")
                            x_vals = np.linspace(min(df_filtered['center_Z_smooth1']), max(df_filtered['center_Z_smooth1']), 100)
                            
                            
                            mean_vz = df_filtered['vz_smooth'].mean()
                            median_vz = df_filtered['vz_smooth'].median()
                            
                            axes[2].axvline(mean_vz, color='red', linestyle='--', linewidth=2, label=f'Mean vz = {mean_vz:.2f}')
                            axes[2].axvline(median_vz, color='green', linestyle='--', linewidth=2, label=f'Median vz = {median_vz:.2f}')

                            axes[2].set_xlabel("Velocity Z")
                            axes[2].set_title("Velocity in Z ")
                            axes[2].legend()
                            
                            # Column 4: Angle
                            axes[3].scatter(df_filtered['angle_cam1'], df_filtered['center_Z'], c="tab:red", alpha=0.5, label="angle1")
                            axes[3].scatter(df_filtered['angle_cam2'], df_filtered['center_Z'], c="tab:orange", alpha=0.5, label="angle2")
                            axes[3].scatter(df_filtered['angle_cam3'], df_filtered['center_Z'], c="tab:blue", alpha=0.5, label="angle3")
                            axes[3].scatter(df_filtered['angle'], df_filtered['center_Z_smooth1'], c="tab:purple", label="angle avg")
                            axes[3].set_xlabel("Angle (deg)")
                            axes[3].set_title("Angle")
                            axes[3].legend()
                            
                            # Column 5: XZ & YZ trajectories together
                            axes[4].scatter(df_filtered['center_X_smooth1'], df_filtered['center_Z_smooth1'], c="tab:green", label="X-Z")
                            axes[4].scatter(df_filtered['center_Y_smooth1'], df_filtered['center_Z_smooth1'], c="tab:cyan", label="Y-Z")
                            axes[4].set_xlabel("X / Y (cm)")
                            axes[4].set_title("Trajectories X-Z & Y-Z")
                            axes[4].legend()
                            
                            plt.tight_layout()
                            plt.show()
                        
                        #%%
                            # ==================================================
                            #  EXPORT CLEANED / SMOOTHED DATA
                            # ==================================================
                            df_filtered['center_X_smooth'] = df_filtered['center_X_smooth1']
                            df_filtered['center_Y_smooth'] = df_filtered['center_Y_smooth1']
                            df_filtered['center_Z_smooth'] = df_filtered['center_Z_smooth1']
                            
                            df = df_filtered[[
                                'tp', 'tp_seconds',
                                'center_X_smooth', 'center_Y_smooth', 'center_Z_smooth', 
                                'angle_cam1', 'angle_cam2', 'angle_cam3', 'angle',
                                'vx_smooth', 'vy_smooth', 'vz_smooth', 
                                'id_number','file_id',  
                            ]].copy()
                          
                            folder = '003Cleaned_coordinates/'
                            parts = os.path.normpath(file_path).split(os.sep)
                            subfolder = parts[1]
                            
                            match = re.search(r'ID_(\d+)', file)
                            if not match:
                                print(f"Skipping file (no ID found): {file}")
                                continue
                            full_id = match.group(0)       # e.g., 'ID_14'
                            id_number = int(match.group(1))  # e.g., 14
                            
                            
                            output_dir = os.path.join(folder, subfolder)
                            os.makedirs(output_dir, exist_ok=True)  # Create folder if it doesn't exist
                            
                            # Create output directory and file path
                            filename = folder + subfolder + '/'+ full_id +'.xlsx'
                            
                            df.to_excel(filename, index=False)
                            print(f"Saved: {filename}")
                            
                   
                          # # Apply LOWESS per axis
                          # x_lowess = lowess(df_filtered['center_X'], np.arange(N_total), frac=frac_equiv, return_sorted=False, it = 3)
                          # y_lowess = lowess(df_filtered['center_Y'], np.arange(N_total), frac=frac_equiv, return_sorted=False, it = 3)
                          # z_lowess = lowess(df_filtered['center_Z'], np.arange(N_total), frac=frac_equiv, return_sorted=False, it = 3)
                          
                          # df_filtered['center_X_smooth1'] = x_lowess
                          # df_filtered['center_Y_smooth1'] = y_lowess
                          # df_filtered['center_Z_smooth1'] = z_lowess
                          
                          
                          
                             
                # sigma_x = N_x_base / 2.355   # 2.355 ≈ convert FWHM → σ (width in frames)
                # sigma_y = N_y_base / 2.355
                # sigma_z = N_z_base / 2.355
                
                # df_filtered['center_X_smooth1'] = gaussian_filter1d(df_filtered['center_X'], sigma=sigma_x)
                # df_filtered['center_Y_smooth1'] = gaussian_filter1d(df_filtered['center_Y'], sigma=sigma_y)
                # df_filtered['center_Z_smooth1'] = gaussian_filter1d(df_filtered['center_Z'], sigma=sigma_z)
                
                # df_filtered = normalize_trajectory(df_filtered, 'center_X_smooth1', 'center_Y_smooth1', 'center_Z_smooth1')
    