import cv2
import mediapipe as mp
import numpy as np
import math
import torch
from ultralytics import YOLO

class GolfSwingAnalyzer:
    def train_yolo_model(self):
        """
        Trains a YOLO model on the golf club head dataset.
        Returns the path to the best saved model weights.
        """
        model = YOLO()  # Initialize YOLO model (specific version like v11 handled by data.yaml or model file)
        # Train the model
        # Note: The epochs and imgsz are examples and can be adjusted.
        # The project and name arguments ensure a predictable output path.
        results = model.train(
            data='Golf-club-head.v2i.yolov11/data.yaml',
            epochs=50,  # Example: 50 epochs
            imgsz=640, # Example: image size 640
            project='runs/train',
            name='golf_club_exp',
            exist_ok=True # Allows overwriting if the experiment name already exists
        )
        # The path to the best weights is typically in results.save_dir or can be constructed
        # results.save_dir should be 'runs/train/golf_club_exp'
        # best_model_path = results.save_dir + '/weights/best.pt' # This is one way if results object has save_dir
        # For ultralytics YOLO, the path is usually predictable based on project and name
        best_model_path = 'runs/train/golf_club_exp/weights/best.pt'
        print(f"YOLO model training complete. Best weights saved at: {best_model_path}")
        return best_model_path

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Train YOLO model and get the path to the best weights
        model_path = self.train_yolo_model()
        
        # Initialize YOLO model
        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
        print(f"Trained model loaded from {model_path}")
        
        # Initialize club head trace
        self.club_head_trace = []
        self.club_positions = [] # This line seems to be kept from original, subtask did not mention removing it.

    def calculate_angle(self, a, b, c):
        """Calculate angle between three points."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
                 np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle

    def calculate_club_speed(self, positions, fps):
        """Calculate club head speed from positions."""
        if len(positions) < 2:
            return 0
        
        # Calculate distance between last two points
        p1 = np.array(positions[-2])
        p2 = np.array(positions[-1])
        distance = np.linalg.norm(p2 - p1)  # pixels
        
        # Convert to speed (pixels per second)
        speed = distance * fps
        
        # You might want to convert pixels to real-world units here
        # This would require calibration with known distances
        return speed

    def detect_club_head(self, frame):
        """Detect club head using the YOLOv11 model."""
        results = self.yolo_model(frame)
        
        # Process results
        # Assuming results.pred[0] contains the detections [x1, y1, x2, y2, confidence, class_id]
        # This might need adjustment based on the exact version of YOLOv5 and ultralytics
        detections = results.pred[0] 
        
        club_head_bbox = None
        max_confidence = 0.5  # Minimum confidence threshold

        for det in detections:
            x1, y1, x2, y2, confidence, class_id = det
            # Assuming 'club-head' is class ID 0. 
            # It's better to verify this using self.yolo_model.names if available
            # For example: if self.yolo_model.names[int(class_id)] == 'club-head':
            if int(class_id) == 0 and confidence > max_confidence:
                # Found a 'club-head' with higher confidence
                club_head_bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                # If you want to take the highest confidence detection, update max_confidence here
                # max_confidence = confidence 
                # For now, we take the first one that meets the threshold or the last one if multiple meet it.
                # To take the one with highest confidence, you'd sort or keep track of max_conf.
                # For simplicity, let's take the first high-confidence detection of class 0
                return club_head_bbox 
                
        return club_head_bbox # Returns None if no detection met the criteria

    def process_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        self.club_head_trace.clear() # Clear trace for new video

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            
            # 1. Convert the BGR frame to RGB for MediaPipe
            image_rgb_for_mediapipe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 2. Process pose with MediaPipe
            pose_results = self.pose.process(image_rgb_for_mediapipe)
            
            # 3. Convert the original frame (or a copy) back to BGR for OpenCV drawing & YOLO
            #    (YOLO expects BGR, and OpenCV drawing functions expect BGR)
            #    If MediaPipe processing modified image_rgb_for_mediapipe in place, 
            #    we should use 'frame' for YOLO and then draw on 'frame' or a copy.
            #    Let's use 'frame' for YOLO and draw on 'frame'.
            #    The variable 'image' will be used for drawing.
            image = frame.copy() # Use a copy to avoid drawing on the original frame if it's reused

            # 4. Detect club head using YOLO on the BGR frame
            club_bbox = self.detect_club_head(frame) # detect_club_head expects BGR

            # 5. If club_bbox exists (club head detected)
            if club_bbox is not None:
                x, y, w, h = club_bbox
                club_center = (int(x + w/2), int(y + h/2))
                self.club_head_trace.append(club_center)
                
                # Draw a dot at the club_center
                cv2.circle(image, club_center, radius=5, color=(0, 255, 0), thickness=-1)
                
                # Draw club head bounding box
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Calculate and display club head speed
                speed = self.calculate_club_speed(self.club_head_trace, fps)
                cv2.putText(image, f'Club Speed: {int(speed)} px/s',
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # 6. Draw the club head trace (path)
            if len(self.club_head_trace) > 1:
                for i in range(1, len(self.club_head_trace)):
                    cv2.line(image, self.club_head_trace[i-1], self.club_head_trace[i],
                           (0, 0, 255), 2) # Red line

            # 7. Draw MediaPipe pose landmarks on the BGR image
            if pose_results.pose_landmarks:
                # Draw pose landmarks
                self.mp_drawing.draw_landmarks(
                    image, # Draw on the BGR image
                    pose_results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )

                # Get landmarks
                landmarks = pose_results.pose_landmarks.landmark

                # Calculate spine angle (between shoulders and hips)
                left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * width,
                               landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * height]
                right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * width,
                                landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * height]
                left_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x * width,
                           landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y * height]
                right_hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x * width,
                            landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y * height]

                # Calculate mid points
                mid_shoulder = [(left_shoulder[0] + right_shoulder[0])/2, (left_shoulder[1] + right_shoulder[1])/2]
                mid_hip = [(left_hip[0] + right_hip[0])/2, (left_hip[1] + right_hip[1])/2]

                # Calculate spine angle
                spine_angle = math.degrees(math.atan2(mid_shoulder[1] - mid_hip[1], 
                                                    mid_shoulder[0] - mid_hip[0]))
                
                # Calculate knee flex
                left_knee_angle = self.calculate_angle(
                    [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x * width,
                     landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y * height],
                    [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x * width,
                     landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y * height],
                    [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x * width,
                     landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y * height]
                )

                # Add measurements to frame
                cv2.putText(image, f'Spine Angle: {int(spine_angle)}deg',
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(image, f'Knee Flex: {int(left_knee_angle)}deg',
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # 8. Write the frame to output
            out.write(image)

        cap.release()
        out.release()

def main():
    analyzer = GolfSwingAnalyzer()
    # Replace these paths with your input and output video paths
    input_video = "input/input_swing.mp4"  
    output_video = "output/analyzed_swing.mp4"
    analyzer.process_video(input_video, output_video)

if __name__ == "__main__":
    main() 