from ultralytics import YOLO
import cv2
import os
import numpy as np

import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv

def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  #-- top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  #-- bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  #-- top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  #-- bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img

def is_in_detection_zone(bbox, zone):
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return (zone[0] <= center_x <= zone[2] and 
            zone[1] <= center_y <= zone[3])

results = {}
license_plates_dict = {}  # To store license plate crops and numbers

mot_tracker = Sort()

# load models
if not os.path.exists('yolov8x.pt'):
    print("Downloading YOLOv8x model...")
    coco_model = YOLO('yolov8x')  # this will download the model
else:
    coco_model = YOLO('/Users/mustafaumutozbek/Documents/finance_analysis/factory_analysis/plate_detection/yolov8x.pt')

license_plate_detector = YOLO('/Users/mustafaumutozbek/Documents/finance_analysis/factory_analysis/plate_detection/license_plate_detector.pt')

# load video
cap = cv2.VideoCapture('/Users/mustafaumutozbek/Documents/finance_analysis/factory_analysis/plate_detection/sample_short.mp4')

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define detection zone (20% to 80% width, 60% to 90% height)
DETECTION_ZONE = (
    int(frame_width * 0.1),    # x1: 20% from left
    int(frame_height * 0.65),   # y1: 60% from top
    int(frame_width * 0.9),    # x2: 80% from left
    int(frame_height * 0.8)    # y2: 80% from top
)

# Create VideoWriter object
out = cv2.VideoWriter('/Users/mustafaumutozbek/Documents/finance_analysis/factory_analysis/plate_detection/output_detection.mp4', 
                     cv2.VideoWriter_fourcc(*'mp4v'),
                     fps, (frame_width, frame_height))

vehicles = [2, 3, 5, 7]

# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        
        # Add text about exit instruction
        cv2.putText(frame, 
                    "Press 'q' to exit", 
                    (50, 50),  # Position in top-left corner
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (255, 255, 255),  # White color
                    2)

        # Draw detection zone - just the border, no fill
        cv2.rectangle(frame, 
                     (DETECTION_ZONE[0], DETECTION_ZONE[1]), 
                     (DETECTION_ZONE[2], DETECTION_ZONE[3]), 
                     (0, 255, 255), 3)  # Solid yellow border, thicker line

        # Add zone label
        cv2.putText(frame, 
                    "Detection Zone", 
                    (DETECTION_ZONE[0] + 10, DETECTION_ZONE[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 255),  # Yellow color
                    2)
        
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        
        # Draw vehicle detections
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1 and is_in_detection_zone([xcar1, ycar1, xcar2, ycar2], DETECTION_ZONE):
                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                'text': license_plate_text,
                                                                'bbox_score': score,
                                                                'text_score': license_plate_text_score}}
                    
                    # Store the license plate crop and text
                    if car_id not in license_plates_dict:
                        # Calculate proper size for license plate display
                        box_width = int(xcar2 - xcar1)
                        display_width = min(box_width * 0.8, 300)  # 80% of box width or max 300px
                        display_height = int(display_width * 0.2)  # Maintain aspect ratio

                        license_plate_crop_resized = cv2.resize(license_plate_crop, 
                                                              (int(display_width), display_height))
                        license_plates_dict[car_id] = {
                            'crop': license_plate_crop_resized,
                            'text': license_plate_text
                        }

                    # Draw border around car with exact screenshot style
                    draw_border(frame, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), 
                              color=(0, 255, 0), thickness=8,  # Thinner lines for cleaner look
                              line_length_x=50, line_length_y=50)  # Shorter corner lines

                    # Draw license plate box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                                (0, 0, 255), 4)  # Red box with proper thickness

                    try:
                        crop = license_plates_dict[car_id]['crop']
                        H, W, _ = crop.shape

                        # Calculate positions for clean layout
                        x_offset = int((xcar2 + xcar1 - W) / 2)
                        y_offset = int(ycar1 - H - 20)

                        # Add white background with padding
                        padding = 10
                        bg_height = H + 60  # Height for text
                        
                        # Draw white background
                        cv2.rectangle(frame,
                                    (x_offset - padding, y_offset - bg_height - padding),
                                    (x_offset + W + padding, y_offset + H + padding),
                                    (255, 255, 255),
                                    -1)

                        # Display license plate crop
                        frame[y_offset:y_offset + H,
                              x_offset:x_offset + W, :] = crop

                        # Calculate text size and position
                        font_scale = min(W / 300 * 2, 2.0)  # Proportional font size
                        thickness = max(int(font_scale * 2), 2)

                        (text_width, text_height), _ = cv2.getTextSize(
                            license_plate_text,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale,
                            thickness)

                        text_x = x_offset + (W - text_width) // 2
                        text_y = y_offset - 20

                        # Draw text
                        cv2.putText(frame,
                                  license_plate_text,
                                  (text_x, text_y),
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  font_scale,
                                  (0, 0, 0),
                                  thickness)

                    except Exception as e:
                        print(f"Error displaying license plate: {e}")

        # Display the frame in high resolution
        frame_resized = cv2.resize(frame, (1920, 1080))
        cv2.imshow('License Plate Detection', frame_resized)
        
        # Write the frame to video
        out.write(frame)
        
        # Modified exit message when 'q' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nExiting program...")
            print("Saving results to test.csv...")
            break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()

# write results
write_csv(results, './test.csv')