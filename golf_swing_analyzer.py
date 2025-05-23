import cv2
import mediapipe as mp
import numpy as np
import math

class GolfSwingAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize tracking variables
        self.tracker = None
        self.club_bbox = None
        self.tracking_initialized = False
        self.club_positions = []
        self.use_simple_tracking = True  # Default to simple tracking
        
        # Try to initialize advanced tracker
        try:
            self.tracker = cv2.TrackerKCF_create()
            self.use_simple_tracking = False
        except AttributeError:
            print("Advanced tracking not available, using simple color tracking instead.")

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
        """Detect club head using color and shape."""
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define range for silver/metallic color (adjust these values based on your club)
        lower = np.array([0, 0, 150])
        upper = np.array([180, 30, 255])
        
        # Create mask for metallic colors
        mask = cv2.inRange(hsv, lower, upper)
        
        # Apply morphological operations to remove noise
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 100:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(largest_contour)
                return (x, y, w, h)
        
        return None

    def process_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            
            # Convert the BGR image to RGB for MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image)
            
            # Convert back to BGR for OpenCV
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Track or detect club head
            if self.use_simple_tracking:
                # Always use color detection for simple tracking
                club_bbox = self.detect_club_head(frame)
                if club_bbox is not None:
                    self.club_bbox = club_bbox
                    x, y, w, h = club_bbox
                    club_center = (int(x + w/2), int(y + h/2))
                    self.club_positions.append(club_center)
                    
                    # Draw club head tracking
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Calculate and display club head speed
                    speed = self.calculate_club_speed(self.club_positions, fps)
                    cv2.putText(image, f'Club Speed: {int(speed)} px/s',
                               (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    # Draw club path
                    if len(self.club_positions) > 1:
                        for i in range(1, len(self.club_positions)):
                            cv2.line(image, self.club_positions[i-1], self.club_positions[i],
                                   (0, 0, 255), 2)
            else:
                # Use advanced tracking
                if not self.tracking_initialized:
                    club_bbox = self.detect_club_head(frame)
                    if club_bbox is not None:
                        self.club_bbox = club_bbox
                        self.tracker.init(frame, club_bbox)
                        self.tracking_initialized = True
                else:
                    success, bbox = self.tracker.update(frame)
                    if success:
                        self.club_bbox = tuple(map(int, bbox))
                        club_center = (int(bbox[0] + bbox[2]/2), int(bbox[1] + bbox[3]/2))
                        self.club_positions.append(club_center)
                        
                        # Draw club head tracking
                        x, y, w, h = self.club_bbox
                        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        
                        # Calculate and display club head speed
                        speed = self.calculate_club_speed(self.club_positions, fps)
                        cv2.putText(image, f'Club Speed: {int(speed)} px/s',
                                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        
                        # Draw club path
                        if len(self.club_positions) > 1:
                            for i in range(1, len(self.club_positions)):
                                cv2.line(image, self.club_positions[i-1], self.club_positions[i],
                                       (0, 0, 255), 2)
                    else:
                        self.tracking_initialized = False

            if results.pose_landmarks:
                # Draw pose landmarks
                self.mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )

                # Get landmarks
                landmarks = results.pose_landmarks.landmark

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