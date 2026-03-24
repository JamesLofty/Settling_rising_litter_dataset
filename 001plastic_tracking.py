import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


input_folder = "videos/Candy, snack and crisps packaging cam1"

# --------- CONFIGURATION ----------
output_folder = "001Dectection_coordinates/Candy, snack and crisps packaging"  # CHECK BEFORE SAVING !!!!!!!
extrinsic_file = "Calibration_codes/Extrinsic_calibration/extrinsic_coordinates_LAB_IMPERIAL.xlsx"
calibration_dir = "Calibration_codes"
video_pattern = "*.mp4"
# -----------------------------------

os.makedirs(output_folder, exist_ok=True)

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

# Load extrinsic coordinates
extrinsic_data = pd.read_excel(extrinsic_file)
extrinsics = {
    row["id"]: np.array(
        [
            (row["x0"], row["y0"], row["z0"]),
            (row["x1"], row["y1"], row["z1"]),
            (row["x2"], row["y2"], row["z2"]),
            (row["x3"], row["y3"], row["z3"]),
        ],
        dtype=np.float32,
    )
    for _, row in extrinsic_data.iterrows()
}

roi_values_inner = {
    "cam1": (400, 1800, 300, 900),
    "cam2": (400, 1800, 300, 900),
    "cam3": (600, 1600, 300, 900),
}


# Helper to clip ROI to frame bounds
def clip_roi(roi, frame_w, frame_h):
    x_min, x_max, y_min, y_max = roi
    x_min = max(0, min(frame_w - 1, x_min))
    x_max = max(0, min(frame_w, x_max))
    y_min = max(0, min(frame_h - 1, y_min))
    y_max = max(0, min(frame_h, y_max))
    return (x_min, x_max, y_min, y_max)


# Process each video
for file in glob.glob(os.path.join(input_folder, video_pattern)):
    print(f"Processing {file}")
    filename = os.path.splitext(os.path.basename(file))[0]

    # Detect camera from filename
    if "cam1" in file:
        camera = "cam1"
    elif "cam2" in file:
        camera = "cam2"
    elif "cam3" in file:
        camera = "cam3"
    else:
        print(f"Camera not found in filename: {file}")
        continue

    # --- Load calibration ---
    camera_matrix = np.load(f"{calibration_dir}/camera_matrix_{camera}_OBS.npz")["camera_matrix"]
    dist_coeffs = np.load(f"{calibration_dir}/dist_coeffs_{camera}_OBS.npz")["dist_coeffs"]

    cap = cv2.VideoCapture(file, cv2.CAP_FFMPEG)

    # --- Initialize other stuff ---
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    bs = cv2.createBackgroundSubtractorKNN(detectShadows=True)
    bs.setHistory(400)

    tp_list = []
    xp_list = []
    yp_list = []
    area_list = []
    angle_list = []
    normalized_angle_list = []
    aspect_list = []

    blob_detections = []
    tp_aruco_list = []
    aruco_corners_detected = []
    frame_pose_data = []
    intensity_per_frame = []

    frame_ID_blobs = 0
    frame_ID_aruco = 0

    # --- Processing loop ---
    while True:
        ret, frame = cap.read()
        if not ret:
            print("NO FRAME")
            break

        frame_ID_blobs += 1
        print(f"Frame {frame_ID_blobs}")

        frame_h, frame_w = frame.shape[:2]
        ix_min, ix_max, iy_min, iy_max = clip_roi(roi_values_inner[camera], frame_w, frame_h)

        # --- Continue with your processing ---
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        mean_intensity = frame_gray.mean()
        intensity_per_frame.append((frame_ID_blobs, mean_intensity))

        frame_clahe = clahe.apply(frame_gray)
        frame_blur = cv2.GaussianBlur(frame_clahe, (5, 5), 1)
        prepared_frame = frame_blur

        # Background subtract
        fgmask = bs.apply(prepared_frame)
        fgmask = cv2.morphologyEx(
            fgmask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        fgmask = cv2.GaussianBlur(fgmask, (7, 7), 0)

        dilate_kernel = np.ones((100, 100))
        dilute_frame = cv2.dilate(fgmask, dilate_kernel, iterations=1)
        thresh_frame = cv2.threshold(dilute_frame, 1, 255, cv2.THRESH_BINARY)[1]

        frame_copy = frame.copy()

        # Draw inner ROI
        cv2.rectangle(frame_copy, (ix_min, iy_min), (ix_max, iy_max), (0, 255, 0), 4)

        # SKIP THE FIRST 20 FRAMES
        if frame_ID_blobs > 20:
            # Contour detection within INNER ROI
            inner_thresh = thresh_frame[iy_min:iy_max, ix_min:ix_max]
            contours, _ = cv2.findContours(
                inner_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                area = cv2.contourArea(contour)

                if area < 100:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                x_full = x + ix_min
                y_full = y + iy_min

                contour_full = contour.copy()
                contour_full[:, :, 0] += ix_min
                contour_full[:, :, 1] += iy_min

                if len(contour_full) >= 5:
                    edge_margin = 5  # tune as needed

                    if (
                        x_full <= ix_min + edge_margin
                        or y_full <= iy_min + edge_margin
                        or x_full + w >= ix_max - edge_margin
                        or y_full + h >= iy_max - edge_margin
                    ):
                        continue

                    ellipse = cv2.fitEllipse(contour_full)
                    center, axes, angle = ellipse
                    normalized_angle = abs(angle if angle < 90 else angle - 180)
                    center = tuple(map(int, center))

                    angle_rad = np.deg2rad(angle)
                    for color, offset in [((0, 0, 255), 0), ((255, 0, 0), np.pi / 2)]:
                        x1, y1 = center
                        x2 = int(x1 + 100 * np.cos(angle_rad + offset))
                        y2 = int(y1 + 100 * np.sin(angle_rad + offset))
                        cv2.line(frame_copy, (x1, y1), (x2, y2), color, 2)

                    cv2.ellipse(frame_copy, ellipse, (255, 255, 0), 4)

                    # Store blob info
                    blob_detections.append([x_full, y_full, w, h])
                    tp_list.append(frame_ID_blobs)

                    # Use moments for a stable centroid
                    M = cv2.moments(contour_full)
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]

                    # Append smoothed centroid and other blob info
                    xp_list.append(cx)
                    yp_list.append(cy)
                    area_list.append(area)
                    aspect_list.append(w / h)
                    normalized_angle_list.append(normalized_angle)
                    angle_list.append(angle)

                    # Draw visualization
                    cv2.rectangle(
                        frame_copy,
                        (x_full, y_full),
                        (x_full + w, y_full + h),
                        (0, 255, 0),
                        3,
                    )
                    cv2.circle(frame_copy, (int(cx), int(cy)), 5, (0, 0, 255), -1)

############################################################

            # --- ArUco detection on full frame ---
            frame_ID_aruco += 1
            corners, ids, _ = detector.detectMarkers(frame)

            all_object_points = []
            all_image_points = []

            if ids is not None:
                for i, corner in enumerate(corners):
                    marker_id = int(ids[i][0])

                    if marker_id == 10:
                        continue

                    # Draw corner points
                    for ci in range(4):
                        px = int(corner[0][ci][0])
                        py = int(corner[0][ci][1])
                        cv2.circle(frame_copy, (px, py), 5, (0, 0, 255), 10)
                        tp_aruco_list.append(frame_ID_aruco)
                        aruco_corners_detected.append((marker_id, ci, (px, py)))

                    # Collect points for solvePnP if needed
                    if marker_id in extrinsics:
                        all_object_points.extend(extrinsics[marker_id])
                        all_image_points.extend(corner[0])

                    # Draw marker outline
                    pts = [(int(p[0]), int(p[1])) for p in corner[0]]
                    cv2.polylines(
                        frame_copy,
                        [np.array(pts, dtype=np.int32)],
                        isClosed=True,
                        color=(0, 255, 255),
                        thickness=2,
                    )

            # Only run solvePnP once per frame if points exist
            if all_object_points and all_image_points:
                obj_pts = np.array(all_object_points, dtype=np.float32)
                img_pts = np.array(all_image_points, dtype=np.float32)

                success, frame_rvec, frame_tvec = cv2.solvePnP(
                    obj_pts,
                    img_pts,
                    camera_matrix,
                    dist_coeffs.reshape(-1, 1),
                )

                if success:
                    # Project back for visual comparison

                    # proj_pts_dist, _ = cv2.projectPoints(obj_pts, frame_rvec, frame_tvec,
                    #                                      camera_matrix, dist_coeffs.reshape(-1,1))
                    # proj_pts_dist = proj_pts_dist.reshape(-1, 2)

                    # plt.figure(figsize=(6,6))
                    # plt.scatter(img_pts[:,0], img_pts[:,1], c='red', label='Detected (img_pts)', marker='x')
                    # plt.scatter(proj_pts_dist[:,0], proj_pts_dist[:,1], c='blue', label='Reprojected (proj_pts)', marker='o')
                    # for (x1,y1), (x2,y2) in zip(img_pts, proj_pts_dist):
                    #     plt.plot([x1, x2], [y1, y2], 'k--', linewidth=0.5)

                    # plt.gca().invert_yaxis()
                    # plt.xlabel("x (pixels)")
                    # plt.ylabel("y (pixels)")
                    # plt.legend()
                    # plt.title("Detected vs Reprojected points")
                    # plt.show()

                    frame_pose_data.append(
                        {
                            "tp": frame_ID_aruco,
                            "rvec1_frame": frame_rvec[0],
                            "rvec2_frame": frame_rvec[1],
                            "rvec3_frame": frame_rvec[2],
                            "tvec1_frame": frame_tvec[0],
                            "tvec2_frame": frame_tvec[1],
                            "tvec3_frame": frame_tvec[2],
                        }
                    )
                else:
                    # SolvePnP failed, still record the tp
                    frame_pose_data.append({"tp": frame_ID_aruco})

###########################################

        # Display
        cv2.imshow('frame', cv2.resize(frame_copy, (800, 450)))
        # cv2.imshow('thresh_frame', cv2.resize(thresh_frame, (800, 450)))
        # cv2.imshow('prepared_frame', cv2.resize(prepared_frame, (800, 450)))

        # out.write(frame_copy)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    # out.release()  # VERY IMPORTANT

    cv2.destroyAllWindows()
    cv2.waitKey(1)

    # --- Save Data ---
    blob_df = pd.DataFrame(
        {
            "tp": tp_list,
            "xp": xp_list,
            "yp": yp_list,
            "area": area_list,
            "aspect": aspect_list,
            "angle": angle_list,
            "normalized_angle": normalized_angle_list,
        }
    )

    if blob_df.empty:
        print("df empty")
    else:
        plt.figure()
        plt.scatter(xp_list, yp_list)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

        ##############################

        pose_df = pd.DataFrame(frame_pose_data)

        for col in pose_df.columns[1:]:
            pose_df[col] = pose_df[col].apply(
                lambda x: float(x) if isinstance(x, (np.ndarray, list)) else x
            )

        aruco_corners_all = []
        for marker_id, corner_index, corner_point in aruco_corners_detected:
            x, y = corner_point
            aruco_corners_all.append(
                {
                    "ID": marker_id,
                    "Corner": corner_index,
                    "x": x,
                    "y": y,
                }
            )

        aruco_df = pd.DataFrame(aruco_corners_all)
        aruco_df["tp"] = tp_aruco_list

        plt.figure()
        plt.scatter(aruco_df["x"], aruco_df["y"])
        plt.title("ACRUCOS")
        plt.show()

        for col in pose_df.columns[1:]:
            pose_df[col] = pose_df[col].apply(
                lambda x: x[0] if isinstance(x, np.ndarray) else x
            )

        merged_marker = pd.merge(aruco_df, pose_df, on="tp", how="left")
        merged_blobs = pd.merge(blob_df, pose_df, on="tp", how="left")

        merged_marker.to_csv(
            os.path.join(output_folder, f"{filename}_MARKER.csv"),
            index=False,
        )
        merged_blobs.to_csv(
            os.path.join(output_folder, f"{filename}_BLOBS.csv"),
            index=False,
        )

        print(f"Saved data for {filename}")