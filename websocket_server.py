from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from keras.models import load_model
import asyncio
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import warnings
warnings.filterwarnings("ignore")


# Load your pre-trained model
model = load_model('C:/Users/Anudip/OneDrive/Desktop/Sign-Language-To-Text-and-Speech-Conversion/cnn8grps_rad1_model.h5')
detector = HandDetector(maxHands=1)

# FastAPI app setup
app = FastAPI()

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # This allows all domains to make requests
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Preprocessing parameters
offset = 29
img_size = 400  # Match this with your model input size

def preprocess_hand_image(hand_image):
    """Resize and normalize the hand image for the model."""
    hand_image = cv2.resize(hand_image, (img_size, img_size))
    hand_image = hand_image.reshape(1, img_size, img_size, 3) / 255.0  # Normalize
    return hand_image

def predict_hand_sign(hand_image):
    """Run the prediction and decode the class."""
    preprocessed_image = preprocess_hand_image(hand_image)
    predictions = model.predict(preprocessed_image)[0]
    predicted_class = np.argmax(predictions)
    class_map = {
        0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
        9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q',
        17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
        25: 'Z'
    }
    # Return the predicted sign language symbol (A-Z)
    return class_map.get(predicted_class, '')

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint that processes webcam frames and sends predictions."""
    await websocket.accept()
    cap = cv2.VideoCapture(0)  # Open the webcam
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                await websocket.send_json({
                    "type": "error", 
                    "message": "Failed to capture frame"
                })
                break

            # Detect hand in the frame
            hands = detector.findHands(frame, draw=False)
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                cropped_image = frame[y - offset:y + h + offset, x - offset:x + w + offset]

                # Predict hand sign
                if cropped_image.size > 0:
                    try:
                        symbol = predict_hand_sign(cropped_image)
                        await websocket.send_json({
                            "type": "prediction", 
                            "symbol": symbol
                        })
                    except Exception as e:
                        await websocket.send_json({
                            "type": "error", 
                            "message": str(e)
                        })
            else:
                await websocket.send_json({
                    "type": "prediction", 
                    "symbol": ""
                })

            await asyncio.sleep(0.1)  # Control the frame rate
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.send_json({
            "type": "error", 
            "message": str(e)
        })
    finally:
        cap.release()  # Release the webcam
        await websocket.close()  # Close the WebSocket connection

