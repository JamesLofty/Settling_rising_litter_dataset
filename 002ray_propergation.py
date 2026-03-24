

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 13:03:51 2025

@author: jameslofty
"""

# ==============================================================
#  IMPORTS
# ==============================================================
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
import re

import warnings
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="invalid value encountered in divide")

# ==============================================================
#  GEOMETRY / OPTICS HELPERS (RAY, REFRACTION, INTERSECTION)
# ==============================================================
def L2P_intersect(p_line, v_line, p_plane, n_plane):
    """Line-plane intersection: where a ray meets the water surface plane."""
    n_d_u = n_plane.dot(v_line)
    if abs(n_d_u) < 1e-9:
        return None  # parallel, no intersection
    w = p_line - p_plane
    si = -n_plane.dot(w) / n_d_u
    return p_line + si * v_line

def ref_ray(i, n_sur, n1, n2):
    """Snell's law in 3D: compute refracted ray direction."""
    i = i / np.linalg.norm(i)
    n_sur = n_sur / np.linalg.norm(n_sur)
    if n_sur.dot(i) > 0:
        n_sur = -n_sur
        n1, n2 = n2, n1
    r = n1 / n2
    c = -n_sur.dot(i)
    k = 1 - r**2 * (1 - c**2)
    if k < 0:
        return None  # total internal reflection
    return r * i + (r * c - np.sqrt(k)) * n_sur

def ray_from_camera_through_water(x, y, intrinsic_matrix, rvec, tvec,
                                  p_air_glass, n_air_glass,
                                  p_glass_water, n_glass_water,
                                  n_air=1.0, n_glass=1.5, n_water=1.33):
    
    """
        experiments go from:
        
        CAMERA (y = 226)----> AIR ----> GLASS (y=128) ----> WATER (y=126) ----> Tank back (y=0)
        
        """
    # Step 1: Air ray
    cam_origin, ray_dir_air = ray_from_camera(x, y, intrinsic_matrix, rvec, tvec)
    
    # Step 2: Hit air-glass plane
    P1 = L2P_intersect(cam_origin, ray_dir_air, p_air_glass, n_air_glass)
    # print(P1)
    if P1 is None:
        return None, None
    
    ray_dir_glass = ref_ray(ray_dir_air, n_air_glass, n_air, n_glass)
    if ray_dir_glass is None:
        return None, None
    
    # Step 3: Hit glass-water plane  
    P2 = L2P_intersect(P1, ray_dir_glass, p_glass_water, n_glass_water)
    if P2 is None:
        return None, None
    
    ray_dir_water = ref_ray(ray_dir_glass, n_glass_water, n_glass, n_water)
    if ray_dir_water is None:
        return None, None
    
    # Return VIRTUAL origin at water surface with water-ray direction
    return P2, ray_dir_water


# ==============================================================
#  RAY FROM CAMERA (NO REFRACTION)
# ==============================================================
def ray_from_camera(x, y, intrinsic_matrix, rvec, tvec):
    # Convert pixel coordinates to normalized camera coordinates
    pixel_coords = np.array([x, y, 1])  # 2D pixel coordinates in homogeneous form
    normalized_coords = np.linalg.inv(intrinsic_matrix).dot(pixel_coords)  # Convert to normalized camera coordinates

    # Construct the camera matrix (rotation and translation)
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    camera_position = -np.dot(rotation_matrix.T, tvec)  # Camera position in world coordinates
    
    # The normalized coordinates define the direction of the ray in camera coordinates
    ray_direction = normalized_coords[:3]  # Remove the homogeneous component (1)
    # Transform the ray to world coordinates using the camera's rotation
    ray_direction_world = np.dot(rotation_matrix.T, ray_direction)
    ray_direction_world /= np.linalg.norm(ray_direction_world)

    return camera_position, ray_direction_world

# ==============================================================
#  RAY / TRIANGULATION UTILITIES
# ==============================================================
def pairwise_closest_points(o1, d1, o2, d2):
    d1 = d1 / np.linalg.norm(d1)
    d2 = d2 / np.linalg.norm(d2)
    A = np.stack([d1, -d2], axis=1)
    b = o2 - o1
    t = np.linalg.lstsq(A, b, rcond=None)[0]
    Q1 = o1 + t[0] * d1
    Q2 = o2 + t[1] * d2
    return Q1, Q2

def dx_pair(o1, d1, o2, d2):
    Q1, Q2 = pairwise_closest_points(o1, d1, o2, d2)
    return float(Q1[0] - Q2[0])  # X-component difference only

def angle_XY_between(d1, d2):
    # angle between directions projected onto the X–Y plane (indices 0 and 2 in your arrays)
    v1 = np.array([d1[0], d1[2]], dtype=float)
    v2 = np.array([d2[0], d2[2]], dtype=float)
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-4 or n2 < 1e-4:
        return 0.0
    v1 /= n1; v2 /= n2
    c = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return float(np.arccos(c))  # radians

def intersect_rays(origins, directions):
    """
    Compute pairwise closest points between 3 rays, and their mean.
    origins: (3,3) array of ray origins
    directions: (3,3) array of ray directions
    """
    def closest_point_between_rays(p1, d1, p2, d2, eps=1e-6):
        d1 = d1 / np.linalg.norm(d1)
        d2 = d2 / np.linalg.norm(d2)
    
        cross = np.cross(d1, d2)
        denom = np.linalg.norm(cross) ** 2
    
        if denom < eps:  
            # Rays are nearly parallel
            # Project p2 onto ray1 and take midpoint
            t = np.dot((p2 - p1), d1)
            proj_p2 = p1 + t * d1
            return (proj_p2 + p2) / 2.0
        else:
            # General case: solve least squares
            A = np.stack([d1, -d2], axis=1)
            b = p2 - p1
            t = np.linalg.lstsq(A, b, rcond=None)[0]
            return (p1 + t[0] * d1 + p2 + t[1] * d2) / 2.0

    o1, o2, o3 = origins
    d1, d2, d3 = directions

    p12 = closest_point_between_rays(o1, d1, o2, d2)
    p13 = closest_point_between_rays(o1, d1, o3, d3)
    p23 = closest_point_between_rays(o2, d2, o3, d3)
    
    p12_13 = (p12 + p13) / 2
    p13_23 = (p13 + p23) / 2
    p12_23 = (p12 + p23) / 2
    p_mean = (p12 + p13 + p23) / 3.0
    
    points = np.stack([p12, p13, p23], axis=0)
    p_median = np.median(points, axis=0)

    return p_mean, p_median, p12, p13, p23, p12_13, p13_23, p12_23 


########################################


# ==============================================================
#  EXPERIMENT CONFIGURATION
# ==============================================================
experiment_folder = "001Dectection_coordinates/Candy, snack and crisps packaging/" 
experiment_name = os.path.basename(os.path.normpath(experiment_folder))
print(experiment_name)

framerate = 1/60

# List all files in the experiment folder
all_files = os.listdir(experiment_folder)

repeats = set()

# ==============================================================
#  LOAD CAMERA CALIBRATION (INTRINSICS)
# ==============================================================
# --- Camera intrinsics and distortion ---
camera_matrix_cam1 = np.load("Calibration_codes/camera_matrix_cam1_OBS.npz")['camera_matrix']
dist_coeffs_cam1   = np.load("Calibration_codes/dist_coeffs_cam1_OBS.npz")['dist_coeffs']

camera_matrix_cam2 = np.load("Calibration_codes/camera_matrix_cam2_OBS.npz")['camera_matrix']
dist_coeffs_cam2   = np.load("Calibration_codes/dist_coeffs_cam2_OBS.npz")['dist_coeffs']

camera_matrix_cam3 = np.load("Calibration_codes/camera_matrix_cam3_OBS.npz")['camera_matrix']
dist_coeffs_cam3   = np.load("Calibration_codes/dist_coeffs_cam3_OBS.npz")['dist_coeffs']

# ==============================================================
#  LOAD GROUND TRUTH EXTRINSICS (WORLD MARKERS)
# ==============================================================
########################################
# --- Ground truth world coordinates (from your Excel file) ---
extrinsic_data = pd.read_excel("Calibration_codes/Extrinsic_calibration/extrinsic_coordinates_LAB_IMPERIAL.xlsx")
extrinsics = {}
for _, row in extrinsic_data.iterrows():
    extrinsics[row['id']] = np.array([
        (row['x0'], row['y0'], row['z0']),
        (row['x1'], row['y1'], row['z1']),
        (row['x2'], row['y2'], row['z2']),
        (row['x3'], row['y3'], row['z3']),
    ], dtype=np.float32)
########################################

# ==============================================================
#  GROUP INPUT FILES BY REPEAT
# ==============================================================
# for file in all_files:
#     if file.endswith('.csv'):
#         match = re.search(r'\((\d+)\)', file)
#         if match:
#             repeat = match.group(1)
#             repeats.add(repeat)
            
for file in all_files:
    if file.endswith('.csv'):
        
        # Case 1: cam1  (00012)_BLOBS.csv → repeat inside parentheses
        m1 = re.search(r'\((\d+)\)', file)
        
        # Case 2: cam1  00012_BLOBS.csv → repeat without parentheses
        m2 = re.search(r'  (\d+)_BLOBS', file)
        
        if m1:
            repeats.add(m1.group(1))
        elif m2:
            repeats.add(m2.group(1))

# ==============================================================
#  MAIN PROCESSING LOOP OVER REPEATS
# ==============================================================
for repeat in sorted(repeats):

    # if repeat != "00001": 
    #     continue
    
    print(f"\nProcessing: {repeat}")
    try:
        # ======================================================
        #  LOAD BLOB DETECTIONS (PLASTIC) FOR 3 CAMERAS
        # ======================================================
        try:
            cam1_coords_blobs = pd.read_csv(experiment_folder + f"cam1  ({repeat})_BLOBS.csv")
            cam2_coords_blobs = pd.read_csv(experiment_folder + f"cam2  ({repeat})_BLOBS.csv")
            cam3_coords_blobs = pd.read_csv(experiment_folder + f"cam3  ({repeat})_BLOBS.csv")
        except FileNotFoundError:
            # Fallback to zero-padded version
            cam1_coords_blobs = pd.read_csv(experiment_folder + f"cam1  {repeat.zfill(5)}_BLOBS.csv")
            cam2_coords_blobs = pd.read_csv(experiment_folder + f"cam2  {repeat.zfill(5)}_BLOBS.csv")
            cam3_coords_blobs = pd.read_csv(experiment_folder + f"cam3  {repeat.zfill(5)}_BLOBS.csv")
                
        # ======================================================
        #  ROI / TANK BOUNDARY CROP
        # ======================================================
        cut = {
            "cam1": (600, 1200, 300, 900),            
            "cam2": (600, 1200, 300, 900),
            "cam3": (600, 1200, 300, 900),
        }
        for cam, df in zip(
            ["cam1", "cam2", "cam3"],
            [cam1_coords_blobs, cam2_coords_blobs, cam3_coords_blobs]
        ):
            x_min, x_max, y_min, y_max = cut[cam]
            x_col, y_col = "xp", "yp"
            
            df.drop(
                df[
                    (df[x_col] < x_min) | (df[x_col] > x_max) |
                    (df[y_col] < y_min) | (df[y_col] > y_max)
                ].index,
                inplace=True
                
    )
    
# %%
    
        plt.figure()
        plt.title(repeat)
        plt.scatter(cam1_coords_blobs['xp'], cam1_coords_blobs['yp'])
        plt.scatter(cam2_coords_blobs['xp'], cam2_coords_blobs['yp'])
        plt.scatter(cam3_coords_blobs['xp'], cam3_coords_blobs['yp'])
        plt.xlabel("x and y (px)")
        plt.ylabel('z (px)')
        plt.xlim(0, 1500)
        plt.ylim(200, 1000)

        plt.show()

        # ======================================================
        #  LOAD MARKER DETECTIONS FOR 3 CAMERAS
        # ======================================================
        try:
            cam1_coords_marker = pd.read_csv(experiment_folder + f"cam1  ({repeat})_MARKER.csv")
            cam2_coords_marker = pd.read_csv(experiment_folder + f"cam2  ({repeat})_MARKER.csv")
            cam3_coords_marker = pd.read_csv(experiment_folder + f"cam3  ({repeat})_MARKER.csv")
        except FileNotFoundError:
            cam1_coords_marker = pd.read_csv(experiment_folder + f"cam1  {repeat.zfill(5)}_MARKER.csv")
            cam2_coords_marker = pd.read_csv(experiment_folder + f"cam2  {repeat.zfill(5)}_MARKER.csv")
            cam3_coords_marker = pd.read_csv(experiment_folder + f"cam3  {repeat.zfill(5)}_MARKER.csv")
        ########################################
        # ======================================================
        #  ALIGN/CLEAN BLOB TABLES (PREFIX, MERGE ON tp)
        # ======================================================
        # Optionally rename columns to avoid overlaps (excluding 'tp')
        cam1_coords_blobs = cam1_coords_blobs.add_prefix('cam1_')
        cam2_coords_blobs = cam2_coords_blobs.add_prefix('cam2_')
        cam3_coords_blobs = cam3_coords_blobs.add_prefix('cam3_')
        
        # Restore the original 'tp' column
        cam1_coords_blobs = cam1_coords_blobs.rename(columns={'cam1_tp': 'tp'})
        cam2_coords_blobs = cam2_coords_blobs.rename(columns={'cam2_tp': 'tp'})
        cam3_coords_blobs = cam3_coords_blobs.rename(columns={'cam3_tp': 'tp'})
        
        # Merge all three on 'tp'
        data_blobs = (
            cam1_coords_blobs
            .merge(cam2_coords_blobs, on='tp', how='inner')
            .merge(cam3_coords_blobs, on='tp', how='inner')
        )
        
        ########################################
        # ======================================================
        #  ALIGN/CLEAN MARKER TABLES (PREFIX, MERGE ON tp/ID/Corner)
        # ======================================================
        # Optionally rename columns to avoid overlaps (excluding 'tp', 'ID', 'Corner')
        cam1_coords_marker = cam1_coords_marker.add_prefix('cam1_')
        cam2_coords_marker = cam2_coords_marker.add_prefix('cam2_')
        cam3_coords_marker = cam3_coords_marker.add_prefix('cam3_')
        
        # Restore the original 'tp', 'ID', 'Corner' columns
        cam1_coords_marker = cam1_coords_marker.rename(columns={'cam1_tp': 'tp', 'cam1_ID': 'ID', 'cam1_Corner': 'Corner'})
        cam2_coords_marker = cam2_coords_marker.rename(columns={'cam2_tp': 'tp', 'cam2_ID': 'ID', 'cam2_Corner': 'Corner'})
        cam3_coords_marker = cam3_coords_marker.rename(columns={'cam3_tp': 'tp', 'cam3_ID': 'ID', 'cam3_Corner': 'Corner'})
        
        # Merge all three on 'tp', 'ID', and 'Corner'
        merged_df = cam1_coords_marker.merge(cam2_coords_marker, on=['tp', 'ID', 'Corner'], how='outer')
        data_markers = merged_df.merge(cam3_coords_marker, on=['tp', 'ID', 'Corner'], how='outer')
        data_markers = data_markers.dropna()

        # ======================================================
        #  SANITY CHECKS FOR EMPTY INPUTS
        # ======================================================
        if cam1_coords_blobs.empty or cam2_coords_blobs.empty or cam3_coords_blobs.empty:
            print(f"Skipping {repeat} due to empty blob coordinate file.")
            continue

        if cam1_coords_marker.empty or cam2_coords_marker.empty or cam3_coords_marker.empty:
            print(f"Skipping {repeat} due to empty marker coordinate file.")
            continue
        
        ########################################
        # ======================================================
        #  SPLIT MARKERS: FRONT (NO REFRACTION) VS BACK (REFRACTION)
        # ======================================================
        front_ids = [0, 1, 2, 3]  # front of tank water
        back_ids  = [4, 5, 6, 7]  # back of tank 
        
        world_points_front, img_cam1_front, img_cam2_front, img_cam3_front = [], [], [], []
        world_points_back,  img_cam1_back,  img_cam2_back,  img_cam3_back  = [], [], [], []
        
        # Iterate over every detection (every frame, every marker corner)
        for _, row in data_markers.iterrows():
            marker_id  = int(row['ID'])
            corner_idx = int(row['Corner'])
        
            # 3D world coordinates of this corner
            wp = extrinsics[marker_id][corner_idx]
        
            # 2D detections from each camera
            pt1 = [row['cam1_x'], row['cam1_y']]
            pt2 = [row['cam2_x'], row['cam2_y']]
            pt3 = [row['cam3_x'], row['cam3_y']]
        
            if marker_id in front_ids:
                world_points_front.append(wp)
                img_cam1_front.append(pt1)
                img_cam2_front.append(pt2)
                img_cam3_front.append(pt3)
        
        # Convert to numpy arrays (so they can be passed to OpenCV)
        world_points_front = np.array(world_points_front, dtype=np.float32)
        img_cam1_front     = np.array(img_cam1_front, dtype=np.float32)
        img_cam2_front     = np.array(img_cam2_front, dtype=np.float32)
        img_cam3_front     = np.array(img_cam3_front, dtype=np.float32)
        
        # ======================================================
        #  SOLVE EXTRINSICS (PnP) FROM FRONT MARKERS ONLY
        # ======================================================
        success1, rvec1, tvec1 = cv2.solvePnP(world_points_front, img_cam1_front, camera_matrix_cam1, dist_coeffs_cam1)
        success2, rvec2, tvec2 = cv2.solvePnP(world_points_front, img_cam2_front, camera_matrix_cam2, dist_coeffs_cam2)
        success3, rvec3, tvec3 = cv2.solvePnP(world_points_front, img_cam3_front, camera_matrix_cam3, dist_coeffs_cam3)
        
        rvec1, tvec1 = rvec1.flatten(), tvec1.flatten()
        rvec2, tvec2 = rvec2.flatten(), tvec2.flatten()
        rvec3, tvec3 = rvec3.flatten(), tvec3.flatten()
        
        # plane here is where air->glass is in real depth (y)
        p_plane_air_glass  = np.array([0, 0, 128])
        n_plane_air_glass  = np.array([0, 0, 1])  
        
        # plane here is where glass->water is in real depth (y)
        p_plane_glass_water = np.array([0, 0, 126])
        n_plane_glass_water = np.array([0, 0, 1])  

        print("#######################EXTRINSICS LOADED#############################")
        ########################################
        #%%
        
        ########################################
        #LETS GET 3D POINTS FOR PLASTIC
        ########################################
        
        particle_3d = []
        particle_3d1 = []
        marker_3d = []
      
        for index, row in data_blobs.iterrows():
            # Get the pixel coordinates
            x1, y1 = row['cam1_xp'], row['cam1_yp']
            x2, y2 = row['cam2_xp'], row['cam2_yp']
            x3, y3 = row['cam3_xp'], row['cam3_yp']
        
            # Refracted ray directions only
            cam1_origin_phys, cam1_ray = ray_from_camera_through_water(
                x1, y1, camera_matrix_cam1, rvec1, tvec1,
                p_plane_air_glass, n_plane_air_glass,
                p_plane_glass_water, n_plane_glass_water
            )
            cam2_origin_phys, cam2_ray = ray_from_camera_through_water(
                x2, y2, camera_matrix_cam2, rvec2, tvec2,
                p_plane_air_glass, n_plane_air_glass,
                p_plane_glass_water, n_plane_glass_water
            )
            cam3_origin_phys, cam3_ray = ray_from_camera_through_water(
                x3, y3, camera_matrix_cam3, rvec3, tvec3,
                p_plane_air_glass, n_plane_air_glass,
                p_plane_glass_water, n_plane_glass_water
            )

            
            origins = [cam1_origin_phys, cam2_origin_phys, cam3_origin_phys]
            directions = [cam1_ray, cam2_ray, cam3_ray]

# %%
            # ======================================================
            #HERE WE CHOOSE WHICH TRAJECTORY TO USE BASED ON MINMISING TRIANGULATION ERROR"""
            # ======================================================
           
            p_mean, p_median, p12, p13, p23, p12_13, p13_23, p12_23 = intersect_rays(origins, directions)
        
            theta12 = np.degrees(angle_XY_between(cam1_ray, cam2_ray))
            theta13 = np.degrees(angle_XY_between(cam1_ray, cam3_ray))
            theta23 = np.degrees(angle_XY_between(cam2_ray, cam3_ray))
            
            # pick best triangulation pair based on max parallax angle
            thetas = [theta12, theta13, theta23]
            points_y = [p12[2], p13[2], p23[2]]  # Y coordinate from each pair
        
            THETA_MIN = 10
            valid = [(t, y) for t, y in zip(thetas, points_y) if not np.isnan(t) and t >= THETA_MIN]
            if valid:
                best_y = max(valid, key=lambda t: t[0])[1] 
                # print('Max angle picked') # choose Y from pair with largest θ
            else:
                best_y = p_mean[2] 
                # print('mean trajectory picked') # fallback to 3-ray mean
    
            particle_3d.append({
                'center_X': p_mean[0],
                'center_Z': p_mean[1],
                'center_Y': p_mean[2],
                'Y_BEST':best_y,
                
                'p12_y':p12[2] ,
                'p13_y':p13[2] ,
                'p23_y':p23[2] ,

                'theta12':theta12,
                'theta13':theta13,
                'theta23':theta23,
        
                'tp': row['tp'],
                
                'angle_cam1': row['cam1_angle'],
                'angle_cam2': row['cam2_angle'],
                'angle_cam3': row['cam3_angle'],
            })
            
        particle_3d_df = pd.DataFrame(particle_3d)
        #################### 
  
        # ======================================================
        # CUT the data to inside the tank
        # ======================================================
        df_cut = particle_3d_df[
        (particle_3d_df['center_X'].between(0, 126)) &
        (particle_3d_df['Y_BEST'].between(0, 126)) &
        (particle_3d_df['center_Z'].between(0, 126))
        ].copy()
        
        particle_3d_df = df_cut
        
        # ======================================================
        #  DIAGNOSTIC PLOTS FOR TRIANGULATION QUALITY
        # ======================================================
        # plt.figure()
        # plt.title(f"VIDEO = {repeat}")
        # plt.scatter(particle_3d_df['dx_mean'], particle_3d_df['center_Z'], label='dx_mean')
        # plt.scatter(particle_3d_df['dx_median'], particle_3d_df['center_Z'], label='dx_median')
        # plt.scatter(particle_3d_df['dx12'], particle_3d_df['center_Z'], label='dx12')
        # plt.scatter(particle_3d_df['dx13'], particle_3d_df['center_Z'], label='dx13')
        # plt.scatter(particle_3d_df['dx23'], particle_3d_df['center_Z'], label='dx23')
        # plt.xlabel('DELTA X ERROR (cm)')
        # plt.legend()
        # plt.show()
    
        
        plt.figure()
        plt.title(f"VIDEO = {repeat}")
        plt.scatter(particle_3d_df['theta12'], particle_3d_df['center_Z'], label='angle 12')
        plt.scatter(particle_3d_df['theta13'], particle_3d_df['center_Z'], label='angle 13')
        plt.scatter(particle_3d_df['theta23'], particle_3d_df['center_Z'], label='angle 23')
        plt.xlabel('ANGLE')
        plt.legend()
        plt.show()
        
        # # plot for sanity
        plt.figure(figsize = (10, 6))
        plt.title(f"VIDEO = {repeat}")
        plt.scatter(particle_3d_df['center_X'], particle_3d_df['center_Z'], label = 'X_mean')
        plt.scatter(particle_3d_df['center_Y'], particle_3d_df['center_Z'], label = 'Y_mean', alpha = 0.8)
        plt.scatter(particle_3d_df['p12_y'], particle_3d_df['center_Z'], label = 'Y_p12', alpha = 0.8)
        plt.scatter(particle_3d_df['p13_y'], particle_3d_df['center_Z'], label = 'Y_p13', alpha = 0.8)
        plt.scatter(particle_3d_df['p23_y'], particle_3d_df['center_Z'], label = 'Y_p23', alpha = 0.8)
        plt.scatter(particle_3d_df['Y_BEST'], particle_3d_df['center_Z'], label = 'Y_best' )

        plt.legend()
        plt.xlim(0,128)
        plt.xlabel('x y (cm)')
        plt.ylabel('z (cm)')
        plt.show()
                
   
        print("#######################3D PLASTICS CORRECTED #############################")

# %%

        # ########################################
        #LETS GET 3D POINTS FOR MARKERS FOR VALIDATION
        # ########################################
        
        for index, row in data_markers.iterrows():
            x1_m, y1_m = row['cam1_x'], row['cam1_y']
            x2_m, y2_m = row['cam2_x'], row['cam2_y']
            x3_m, y3_m = row['cam3_x'], row['cam3_y']
        
            if row["ID"] <= 3:
                # Front markers (in air) -> straight rays
                cam1_origin_m, cam1_ray_m = ray_from_camera(x1_m, y1_m, camera_matrix_cam1, rvec1, tvec1)
                cam2_origin_m, cam2_ray_m = ray_from_camera(x2_m, y2_m, camera_matrix_cam2, rvec2, tvec2)
                cam3_origin_m, cam3_ray_m = ray_from_camera(x3_m, y3_m, camera_matrix_cam3, rvec3, tvec3)
                
            else:
             
                cam1_origin_m, cam1_ray_m = ray_from_camera_through_water(
                    x1_m, y1_m, camera_matrix_cam1, rvec1, tvec1,
                    p_plane_air_glass, n_plane_air_glass,
                    p_plane_glass_water, n_plane_glass_water
                )
                cam2_origin_m, cam2_ray_m = ray_from_camera_through_water(
                    x2_m, y2_m, camera_matrix_cam2, rvec2, tvec2,
                    p_plane_air_glass, n_plane_air_glass,
                    p_plane_glass_water, n_plane_glass_water
                )
                cam3_origin_m, cam3_ray_m = ray_from_camera_through_water(
                    x3_m, y3_m, camera_matrix_cam3, rvec3, tvec3,
                    p_plane_air_glass, n_plane_air_glass,
                    p_plane_glass_water, n_plane_glass_water
                )
                        
            origins_m = [cam1_origin_m, cam2_origin_m, cam3_origin_m]
            directions_m = [cam1_ray_m, cam2_ray_m, cam3_ray_m]
        
        ############ Find the intersection of the two rays
        
            intersection_center_marker, p_median, p12, p13, p23, p12_13, p13_23, p12_23= intersect_rays(origins_m, directions_m)          

        ############ Append the 3D position to the result list
        
            marker_3d.append({
            'X': intersection_center_marker[0],
            'Z': intersection_center_marker[1],
            'Y': intersection_center_marker[2],
            'ID': row['ID'],
            'Corner': row['Corner']
        })
            
        marker_3d_df = pd.DataFrame(marker_3d)
        
        # ############
        
        print("#######################3D MARKERS#############################")

        #%%
        # # ########################################
        # # COMPARE MARKERS AND POINTS
        # # ########################################
                
        extrinsics_x = []
        extrinsics_z = []
        extrinsics_y = []
        
        for marker in extrinsics.values():
            extrinsics_x.extend(marker[:, 0])  # x values
            extrinsics_z.extend(marker[:, 1])  # y values
            extrinsics_y.extend(marker[:, 2])  # y values
            
        # 3D scatter plot
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(marker_3d_df['X'], marker_3d_df['Y'], marker_3d_df['Z'], label='Reconstructed', c='b')
        ax.scatter(extrinsics_x, extrinsics_y, extrinsics_z, label='Ground truth', c='r', marker='^')
        ax.scatter(particle_3d_df['center_X'], particle_3d_df['Y_BEST'], particle_3d_df['center_Z'], 
                   label='Plastics', c='g', marker='o', alpha=0.6)
        
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_zlabel('Z (cm)')
        ax.set_xlim(0,128)
        ax.set_ylim(0,128)
        ax.set_zlim(0,100)
        ax.legend()
        ax.set_title('3D Reconstruction vs Ground Truth')
        plt.show()
        
        # ======================================================
        #  MARKER RECONSTRUCTION ERROR VS GROUND TRUTH
        # ======================================================
        errors = []
        front_ids = [0, 1, 2, 3, 4, 5, 6]

        for _, row in marker_3d_df.iterrows():
            mid, corner = int(row['ID']), int(row['Corner'])
            
            if mid not in front_ids:
                continue
        
            gt = extrinsics[mid][corner]
        
            # reorder GT to (X, Y, Z)
            X_gt, Y_gt, Z_gt = gt[0], gt[2], gt[1]
        
            dX = row['X'] - X_gt
            dY = row['Y'] - Y_gt
            dZ = row['Z'] - Z_gt
            err = np.sqrt(dX**2 + dY**2 + dZ**2)
        
            errors.append([mid, corner, dX, dY, dZ, err])
        
        err_df = pd.DataFrame(errors, columns=['ID','Corner','dX','dY','dZ','err'])
        
        # plt.figure
        # plt.plot(err_df['dX'])
        # plt.plot(err_df['dY'])
        # plt.plot(err_df['dZ'])
        # plt.show()
        
        print("Front markers only")
        print("Mean abs error (cm):", err_df[['dX','dY','dZ']].abs().mean().to_dict())
        print("std abs error (cm):", err_df[['dX','dY','dZ']].abs().std().to_dict())
        print("Mean 3D error (cm):", err_df['err'].mean())
        # print("#######################MARKER ERROR CALCULATED#############################")

# %%
     

# %%


        """FINAL CHECK OUT"""
        
        plt.figure()
        plt.title(f"FINAL CHECK VIDEO = {repeat}")
        plt.scatter(particle_3d_df['center_X'], particle_3d_df['center_Z'], label = 'X_mean')
        plt.scatter(particle_3d_df['Y_BEST'], particle_3d_df['center_Z'], label = 'Y_BEST' )

        plt.legend()
        plt.xlabel('x or y (cm)')
        plt.ylabel('z(cm)')
        plt.show()
        
#%%
        # ======================================================
        #  FINAL CLEANUP & SAVE MERGED COORDINATES
        # ======================================================
        
        # Overwrite original coordinates with corrected ones
        particle_3d_df['center_X'] = particle_3d_df['center_X']
        particle_3d_df['center_Y'] = particle_3d_df['Y_BEST']  
        particle_3d_df['center_Z'] = particle_3d_df['center_Z']
        
        # Drop the helper columns
        particle_3d_df = particle_3d_df.drop(columns=['center_X_median',
               'center_Z_median', 'center_Y_median', 'p12_y', 'p13_y', 'p23_y',
               'p12_13_y', 'p13_23_y', 'p12_23_y', 'dx12', 'dx13', 'dx23',
               'dx_mean_12_13', 'dx_mean_13_23', 'dx_mean_12_23', 'dx_median', 'dx_mean',
               'Y_BEST','dy_mean', 'theta12', 'theta13', 'theta23', 'Y_per_frame', 'Y_global'],
                                             errors='ignore')
    
        folder = '002Megered_coordinates/'
        repeat = str(repeat)  # Ensure it's a string
        experiment_name = experiment_name + '/'
    
        output_dir = os.path.join(folder, experiment_name)
        os.makedirs(output_dir, exist_ok=True)  # Create folder if it doesn't exist
    
        output_file = os.path.join(output_dir, f'ID_{repeat}.xlsx')
                
        # Save the DataFrame
        particle_3d_df.to_excel(output_file, index=False)
        
        print("######################")
        print("EVERYTHING SAVED ")
        print("######################")
    
    except FileNotFoundError as e:
        print(f"❌ Missing file for repeat {repeat}: {e}")
        continue

    except (np.linalg.LinAlgError, KeyError, ValueError) as e:
        print(f"⚠️ Skipping repeat {repeat} due to error: {e}")
        continue
    