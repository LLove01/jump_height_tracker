import cv2
import mediapipe as mp
import time

# User defined variables
user_height_in_cm = 183
landing_tolerance = 0.05
display_mp_pose = True  # Set to True to display MediaPipe pose
save_heights = True  # Set to True to save jump heights to a txt file

# Variables for calibration, jump tracking, and history
is_calibrated = False
calibration_message_time = None
pixels_per_cm = None
jump_start_position = None
max_jump_height_pixels = 0
in_jump = False
jump_heights = []  # List to store jump heights

# Initialize MediaPipe pose detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5)

# Open the video capture
cap = cv2.VideoCapture(0)

def is_full_body_visible(landmarks):
    # Define the necessary landmarks
    necessary_landmarks = [
        mp_pose.PoseLandmark.NOSE,
        mp_pose.PoseLandmark.LEFT_HEEL,
        mp_pose.PoseLandmark.RIGHT_HEEL,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_ELBOW,
        mp_pose.PoseLandmark.RIGHT_ELBOW
    ]

    # Check if all necessary landmarks are visible
    return all(landmarks.landmark[landmark.value].visibility > 0.5 for landmark in necessary_landmarks)

def are_arms_horizontal(landmarks):
    # Tolerance for y-coordinate variation to consider arms horizontal
    tolerance = 0.05
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_elbow = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    right_elbow = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value]

    # Check if elbows and shoulders are approximately at the same height
    return abs(left_shoulder.y - left_elbow.y) < tolerance and abs(right_shoulder.y - right_elbow.y) < tolerance

def calibrate_height(results):
    global pixels_per_cm, is_calibrated
    if results.pose_landmarks and is_full_body_visible(results.pose_landmarks) and are_arms_horizontal(results.pose_landmarks):
        head_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE.value].y
        foot_y = min(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL.value].y,
                     results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL.value].y)
        height_in_pixels = abs(head_y - foot_y)
        pixels_per_cm = user_height_in_cm / height_in_pixels
        is_calibrated = True

def detect_jump(image, results):
    global jump_start_position, max_jump_height_pixels, in_jump

    if results.pose_landmarks:
        left_hip_y = results.pose_landmarks.landmark[23].y
        right_hip_y = results.pose_landmarks.landmark[24].y
        avg_hip_y = (left_hip_y + right_hip_y) / 2

        if jump_start_position is None:
            jump_start_position = avg_hip_y

        # Start of jump detection
        if not in_jump and avg_hip_y < jump_start_position - 0.05:  # Adjust threshold as needed
            in_jump = True

        # Jump height calculation
        if in_jump:
            jump_height_pixels = jump_start_position - avg_hip_y
            max_jump_height_pixels = max(max_jump_height_pixels, jump_height_pixels)

         # End of jump detection
        if in_jump and abs(avg_hip_y - jump_start_position) < landing_tolerance:
            in_jump = False
            max_jump_height_cm = round(max_jump_height_pixels * pixels_per_cm, 1) if pixels_per_cm else 0
            jump_heights.append(max_jump_height_cm)
            if save_heights:
                write_jump_height_to_file(max_jump_height_cm)
            max_jump_height_pixels = 0  # Reset for next jump

        # Display jump state and current jump height at the lower left corner
        jump_state_text = "Jumping" if in_jump else "Not Jumping"
        jump_state_color = (0, 255, 0) if in_jump else (0, 0, 255)
        cv2.putText(image, jump_state_text, (10, frame.shape[0] - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, jump_state_color, 2, cv2.LINE_AA)
        
    max_jump_height_cm = round(max_jump_height_pixels * pixels_per_cm, 1) if pixels_per_cm else 0
    cv2.putText(image, f'Jump Height: {max_jump_height_cm:.1f}cm', (10, frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return image

def write_jump_height_to_file(height_cm):
    with open("jump_heights.txt", "a") as file:
        file.write(f"{height_cm}\n")

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if display_mp_pose and results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    if not is_calibrated:
        cv2.putText(frame, "Stand in T-pose for calibration", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        calibrate_height(results)
        if is_calibrated:
            # Start the timer when calibration is done
            calibration_message_time = time.time()  
    else:
        if calibration_message_time and time.time() - calibration_message_time < 2:
            # Delete message after 2s
            cv2.putText(frame, "T-pose accepted", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        frame = detect_jump(frame, results)
        # Display top 3 jump heights at the bottom right corner
        top_jumps = sorted(jump_heights, reverse=True)[:3]
        for i, jump in enumerate(top_jumps):
            cv2.putText(frame, f"Top {i+1}: {round(jump, 1)}cm", 
                        (frame.shape[1] - 190, frame.shape[0] - 30 * (3 - i)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Jump Height Tracker', frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        # exit
        break
    elif key & 0xFF == ord('p'):
        # Toggle the mp_pose
        display_mp_pose = not display_mp_pose  
cap.release()
cv2.destroyAllWindows()

