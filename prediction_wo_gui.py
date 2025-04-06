import math
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from keras.models import load_model
import traceback
from typing import Tuple, List, Dict

class SignLanguageRecognizer:
    def __init__(self):
        try:
            # Initialize model
            self.model = load_model('cnn8grps_rad1_model.h5')
            
            # Initialize camera
            self.capture = cv2.VideoCapture(0)
            
            # Initialize hand detectors with specific parameters
            self.hd = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, 
                                 detectionCon=0.5, minTrackCon=0.5)
            
            # Create white background
            self.white = np.ones((400, 400), np.uint8) * 255
            cv2.imwrite("white.jpg", self.white)
            
            # Constants
            self.OFFSET = 29
            self.predictions_dict = {}
            self.predictions_list = []
            
        except Exception as e:
            print(f"Initialization error: {str(e)}")
            raise

    @staticmethod
    def calculate_distance(point1: List[float], point2: List[float]) -> float:
        return math.sqrt(((point1[0] - point2[0]) ** 2) + ((point1[1] - point2[1]) ** 2))

    def draw_hand_connections(self, white_img: np.ndarray, pts: List[List[int]], offset_x: int, offset_y: int) -> np.ndarray:
        try:
            # Draw finger connections
            finger_ranges = [(0, 4), (5, 8), (9, 12), (13, 16), (17, 20)]
            for start, end in finger_ranges:
                for t in range(start, end, 1):
                    pt1 = (pts[t][0] + offset_x, pts[t][1] + offset_y)
                    pt2 = (pts[t + 1][0] + offset_x, pts[t + 1][1] + offset_y)
                    cv2.line(white_img, pt1, pt2, (0, 255, 0), 3)
            
            # Draw palm connections
            palm_connections = [(5, 9), (9, 13), (13, 17), (0, 5), (0, 17)]
            for start, end in palm_connections:
                pt1 = (pts[start][0] + offset_x, pts[start][1] + offset_y)
                pt2 = (pts[end][0] + offset_x, pts[end][1] + offset_y)
                cv2.line(white_img, pt1, pt2, (0, 255, 0), 3)
            
            # Draw landmarks
            for i in range(21):
                cv2.circle(white_img, 
                          (pts[i][0] + offset_x, pts[i][1] + offset_y),
                          2, (0, 0, 255), 1)
                
            return white_img
        except Exception as e:
            print(f"Error in draw_hand_connections: {str(e)}")
            return white_img

    def process_sign(self, pts: List[List[int]]) -> str:
        try:
            # Basic sign detection based on finger positions
            thumb_tip = pts[4]
            index_tip = pts[8]
            middle_tip = pts[12]
            ring_tip = pts[16]
            pinky_tip = pts[20]
            
            # Get vertical positions of fingertips
            fingers_up = []
            
            # Thumb
            if thumb_tip[0] < pts[3][0]:  # Left side of hand
                fingers_up.append(1)
            else:
                fingers_up.append(0)
                
            # Other fingers
            for tip_id in [8, 12, 16, 20]:  # Index, Middle, Ring, Pinky
                if pts[tip_id][1] < pts[tip_id - 2][1]:  # If tip is above the second joint
                    fingers_up.append(1)
                else:
                    fingers_up.append(0)
            
            # Basic gesture recognition
            total_fingers = sum(fingers_up)
            
            # Simple mapping of finger counts to letters
            gesture_map = {
                0: "A",
                1: "B",
                2: "C",
                3: "D",
                4: "E",
                5: "F"
            }
            
            return gesture_map.get(total_fingers, "Unknown")
            
        except Exception as e:
            print(f"Error in process_sign: {str(e)}")
            return "Unknown"

    def run(self):
        try:
            while True:
                # Read frame
                success, frame = self.capture.read()
                if not success:
                    print("Failed to capture frame")
                    continue

                # Flip frame horizontally for more intuitive interaction
                frame = cv2.flip(frame, 1)
                
                # Find hands
                hands, frame = self.hd.findHands(frame)  # Enable drawing for visualization
                
                if hands:
                    # Get the first hand detected
                    hand = hands[0]
                    landmarks = hand["lmList"]  # Get the landmark list
                    
                    # Process the hand gesture
                    sign = self.process_sign(landmarks)
                    
                    # Draw the prediction on the frame
                    cv2.putText(frame, f"Sign: {sign}", (10, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Create white background for visualization
                    white_img = np.ones((400, 400, 3), dtype=np.uint8) * 255
                    
                    # Draw hand skeleton on white background
                    if len(landmarks) >= 21:  # Ensure we have all landmarks
                        os_x = 100  # Fixed offset for visualization
                        os_y = 100
                        white_img = self.draw_hand_connections(white_img, landmarks, os_x, os_y)
                        cv2.imshow("Hand Skeleton", white_img)

                # Show the original frame
                cv2.imshow("Sign Language Recognition", frame)
                
                # Break loop with 'Esc' key
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        except Exception as e:
            print(f"Error in main loop: {str(e)}")
            print(traceback.format_exc())
        finally:
            self.cleanup()

    def cleanup(self):
        self.capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        recognizer = SignLanguageRecognizer()
        recognizer.run()
    except Exception as e:
        print(f"Application error: {str(e)}")
        print(traceback.format_exc())