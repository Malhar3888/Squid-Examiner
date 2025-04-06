import cv2
import mediapipe as mp
import numpy as np
import time
from ultralytics import YOLO

# Initialize Mediapipe face mesh for head pose estimation
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize Mediapipe face detection for individual person detection
mp_face_detection = mp.solutions.face_detection

# Initialize video capture
cap = cv2.VideoCapture(0)

# Load YOLO model for pose detection
model = YOLO("yolov8n-pose.pt")  # load a pretrained model

def offline_process_frames(cap):
    start_time = None
    looking_direction = None
    duration_threshold = 5  # in seconds

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Convert BGR image to RGB for both face detection and head pose estimation
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces using face detection for individual person detection
        with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
            face_results = face_detection.process(rgb_frame)

            if face_results.detections:
                for i, detection in enumerate(face_results.detections):
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                    # Draw bounding box around the face
                    cv2.rectangle(frame, bbox, (0, 255, 0), 2)

                    # Crop the face region
                    face_img = frame[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]

                    # Ensure that face_img is not empty
                    if face_img.size == 0:
                        continue

                    # Resize frame for YOLOv8 pose detection
                    frame_resized = cv2.resize(frame, (640, 480))

                    # Use the YOLO model to predict the poses in the frame
                    results = model.predict(frame_resized, save=True)
                    detection = results[0].plot()

                    # Display the resulting image with the predicted poses
                    cv2.imshow('YOLOv8 Pose Detection', detection)

                    # Convert the cropped face to RGB for head pose estimation
                    face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

                    # Get the face landmarks and head pose estimation
                    results = face_mesh.process(face_img_rgb)

                    if results.multi_face_landmarks:
                        for face_landmarks in results.multi_face_landmarks:
                            face_3d = []
                            face_2d = []

                            for idx, lm in enumerate(face_landmarks.landmark):
                                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                                    if idx == 1:
                                        nose_2d = (lm.x * bbox[2], lm.y * bbox[3])
                                        nose_3d = (lm.x * bbox[2], lm.y * bbox[3], lm.z * 3000)

                                    x_lm, y_lm = int(lm.x * bbox[2]), int(lm.y * bbox[3])

                                    # Get the 2d coordinates
                                    face_2d.append([x_lm + bbox[0], y_lm + bbox[1]])

                                    # Get the 3d coordinates
                                    face_3d.append([x_lm + bbox[0], y_lm + bbox[1], lm.z])

                            # Convert to numpy arrays
                            face_2d = np.array(face_2d, dtype=np.float64)
                            face_3d = np.array(face_3d, dtype=np.float64)

                            # Camera matrix
                            focal_length = 1 * bbox[2]
                            cam_matrix = np.array([[focal_length, 0, bbox[2] / 2],
                                                [0, focal_length, bbox[3] / 2],
                                                [0, 0, 1]])

                            # Distance matrix
                            dist_matrix = np.zeros((4, 1), dtype=np.float64)

                            # Solve PnP
                            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                            # Get rotational matrix
                            rmat, jac = cv2.Rodrigues(rot_vec)

                            # Get angles
                            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                            # Get the y rotation degree
                            x = angles[0] * 360
                            y = angles[1] * 360
                            z = angles[2] * 360

                            # See where the user's head tilting
                            if y < -20:
                                text = "Looking Left"
                                if looking_direction != 'left':
                                    start_time = time.time()
                                    looking_direction = 'left'
                                else:
                                    if time.time() - start_time > duration_threshold:
                                        print(f"Person {i+1} is looking left for more than {duration_threshold} seconds!")
                            elif y > 18:
                                text = "Looking Right"
                                if looking_direction != 'right':
                                    start_time = time.time()
                                    looking_direction = 'right'
                                else:
                                    if time.time() - start_time > duration_threshold:
                                        print(f"Person {i+1} is looking right for more than {duration_threshold} seconds!")
                            # elif x < -15:
                            #     text = "Looking Down"
                            # elif x > 15:
                            #     text = "Looking Up"
                            else:
                                text = "Forward"

                            # Display the nose direction
                            nose_3d_projection, jacobin = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix,
                                                                            dist_matrix)

                            p1 = (int(nose_2d[0] + x * 10), int(nose_2d[1] - y * 10))
                            p2 = (int(nose_2d[0] + z * 10), int(nose_2d[1] - x * 10))

                            # Convert nose_2d and p1 coordinates to integers
                            nose_2d = (int(nose_2d[0]), int(nose_2d[1]))
                            p1 = (int(p1[0]), int(p1[1]))

                            # Draw line representing head pose
                            # cv2.line(frame, nose_2d, p1, (255, 0, 0), 2)

                            # Display head pose estimation text
                            cv2.putText(frame, f"P{i+1}: " + text, (bbox[0], bbox[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        (0, 0, 255), 2)
            else:
                print("Face not Detected!")

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
