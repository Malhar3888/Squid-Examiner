import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize Mediapipe face mesh for head pose estimation
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize Mediapipe face detection for individual person detection
mp_face_detection = mp.solutions.face_detection

def online_process_frames(cap):
    start_time = None
    looking_direction = None
    duration_threshold = 5  # in seconds
    num_persons_detected = 0  # Track the number of persons detected

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
            face_results = face_detection.process(rgb_frame)

            if face_results.detections:
                num_persons_detected = len(face_results.detections)  # Update the number of persons detected
                if num_persons_detected > 1:
                    print("More than one person detected!")  # Print message if more than one person detected
                for i, detection in enumerate(face_results.detections):
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                    cv2.rectangle(frame, bbox, (0, 255, 0), 2)

                    face_img = frame[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]

                    if face_img.size == 0:
                        continue

                    face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

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

                                    face_2d.append([x_lm + bbox[0], y_lm + bbox[1]])
                                    face_3d.append([x_lm + bbox[0], y_lm + bbox[1], lm.z])

                            face_2d = np.array(face_2d, dtype=np.float64)
                            face_3d = np.array(face_3d, dtype=np.float64)

                            focal_length = 1 * bbox[2]
                            cam_matrix = np.array([[focal_length, 0, bbox[2] / 2],
                                                   [0, focal_length, bbox[3] / 2],
                                                   [0, 0, 1]])

                            dist_matrix = np.zeros((4, 1), dtype=np.float64)

                            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                            rmat, _ = cv2.Rodrigues(rot_vec)

                            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

                            x = angles[0] * 360
                            y = angles[1] * 360
                            z = angles[2] * 360

                            if y < -30:
                                text = "Looking Left"
                                if looking_direction != 'left':
                                    start_time = time.time()
                                    looking_direction = 'left'
                                else:
                                    if time.time() - start_time > duration_threshold:
                                        print(f"Person {i+1} is looking left for more than {duration_threshold} seconds!")
                            elif y > 30:
                                text = "Looking Right"
                                if looking_direction != 'right':
                                    start_time = time.time()
                                    looking_direction = 'right'
                                else:
                                    if time.time() - start_time > duration_threshold:
                                        print(f"Person {i+1} is looking right for more than {duration_threshold} seconds!")
                            else:
                                text = "Forward"

                            nose_3d_projection, _ = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix,
                                                                       dist_matrix)

                            p1 = (int(nose_2d[0] + x * 10), int(nose_2d[1] - y * 10))
                            p2 = (int(nose_2d[0] + z * 10), int(nose_2d[1] - x * 10))

                            nose_2d = (int(nose_2d[0]), int(nose_2d[1]))
                            p1 = (int(p1[0]), int(p1[1]))

                            cv2.putText(frame, f"P{i+1}: " + text, (bbox[0], bbox[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        (0, 0, 255), 2)
            else:
                print("Face not Detected!")

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
