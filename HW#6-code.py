# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 15:28:16 2024

@author: zezva
"""

import cv2
import numpy as np

# Step 1: Capture video frames and create a background model
video_path = "C:\\Users\\zezva\\Desktop\\HW#6\\1.mp4"
cap = cv2.VideoCapture(video_path)

# Collect a number of frames to create the background model
num_frames_for_bg = 32
frames = []

while len(frames) < num_frames_for_bg:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

# Compute the background model by averaging the frames
bg_model = np.mean(frames, axis=0).astype(dtype=np.uint8)

# Step 2: Process each frame to detect moving objects
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to the first frame
frame_count = 0

# Define the codec and create VideoWriter objects for output videos
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
original_output = cv2.VideoWriter("C:\\Users\\zezva\\Desktop\\HW#6\\original_output.avi", fourcc, 30.0, (bg_model.shape[1], bg_model.shape[0]))
difference_output = cv2.VideoWriter("C:\\Users\\zezva\\Desktop\\HW#6\\difference_output.avi", fourcc, 30.0, (bg_model.shape[1], bg_model.shape[0]))
detection_output = cv2.VideoWriter("C:\\Users\\zezva\\Desktop\\HW#6\\detection_output.avi", fourcc, 30.0, (bg_model.shape[1], bg_model.shape[0]))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Step 3: Calculate the difference between the background model and the current frame
    diff = cv2.absdiff(frame, bg_model)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # Step 4: Threshold the difference to get the foreground mask
    _, fg_mask = cv2.threshold(gray_diff, 50, 255, cv2.THRESH_BINARY)
    
    # Step 5: Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    
    # Step 6: Find connected components to get the ROIs
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)
    
    # Step 7: Filter ROIs based on area
    min_area = 5000  # Lower limit for area
    max_area = 50000  # Upper limit for area
    rois = []
    
    for i in range(1, num_labels):  # Start from 1 to skip the background
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area < area < max_area:
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            rois.append((x, y, w, h))
            # Draw rectangle on the original frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Write frames to output videos
    original_output.write(frame)
    difference_output.write(cv2.cvtColor(gray_diff, cv2.COLOR_GRAY2BGR))
    detection_output.write(cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR))
    
    # Display frames for debugging (optional)
    cv2.imshow('Original', frame)
    cv2.imshow('Difference', gray_diff)
    cv2.imshow('Detection', fg_mask)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
original_output.release()
difference_output.release()
detection_output.release()
cv2.destroyAllWindows()

print("Processing completed and videos saved successfully.")
