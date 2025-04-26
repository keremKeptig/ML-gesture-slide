import pickle
import cv2
import mediapipe as mp
import numpy as np
import os
import sys
import time
import asyncio
import websockets
from collections import deque, Counter
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the NeuralNetwork class from ml/model.py.
try:
    from framework.model import NeuralNetwork
except ImportError:
    raise ImportError("Could not import NeuralNetwork from ml/model.py. Make sure ml/__init__.py exists.")

# WebSocket server URI
WEBSOCKET_URI = "ws://localhost:8000/events"

# Function to send gestures to the Sanic WebSocket server
async def send_gesture_to_server(gesture: str):
    """Send detected gesture to the WebSocket server."""
    print("Sending gesture to server...:", gesture)
    try:
        async with websockets.connect(WEBSOCKET_URI) as ws:
            await ws.send(gesture)
            print(f"Sent gesture: {gesture}")
    except Exception as e:
        print(f"Error sending gesture: {e}")

# -------------------------------
# Load the Trained Model
# -------------------------------
def load_trained_model():
    model_path = os.path.join("models", "best_model_optional.npz")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    params = np.load(model_path)
    
    num_layers = sum(1 for key in params.keys() if key.startswith("W"))
    n_layers = num_layers - 1
    
    first_weight = params["W1"]
    input_dim = first_weight.shape[0]
    hidden_dim = first_weight.shape[1]
    output_dim = params[f"W{num_layers}"].shape[1]
    print(input_dim, hidden_dim, output_dim, n_layers)
    model = NeuralNetwork(input_dim, hidden_dim, output_dim, n_layers=n_layers)
    
    model.weights = []
    model.biases = []
    for i in range(1, num_layers + 1):
         model.weights.append(params[f"W{i}"])
         model.biases.append(params[f"b{i}"])
    
    print("Model loaded successfully.")
    return model

# -------------------------------
# Compute Normalization Parameters from Training Data
# -------------------------------
def compute_norm_params(training_csv, gesture_col="ground_truth"):
    df = pd.read_csv(training_csv)
    feature_columns = [col for col in df.columns if col not in [gesture_col, "timestamp"]]
    X = df[feature_columns].values.astype(np.float32)
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + 1e-8
    return X_mean, X_std

# -------------------------------
# Feature Extraction from Pose Landmarks
# -------------------------------
def extract_pose_features(pose_landmarks):
    features = []
    for lm in pose_landmarks.landmark:
        features.extend([lm.x, lm.y, lm.z, lm.visibility])
    return np.array(features, dtype=np.float32).reshape(1, -1)

def construct_feature_vector(pose_landmarks, expected_dim):
    if pose_landmarks is None:
        return None
    pose_features = extract_pose_features(pose_landmarks)
    
    current_dim = pose_features.shape[1]
    if current_dim < expected_dim:
        padding = np.zeros((1, expected_dim - current_dim), dtype=pose_features.dtype)
        feature_vector = np.concatenate([pose_features, padding], axis=1)
    elif current_dim > expected_dim:
        feature_vector = pose_features[:, :expected_dim]
    else:
        feature_vector = pose_features

    return feature_vector

# -------------------------------
# Temporal Smoothing Utility
# -------------------------------
def majority_vote(queue):
    if not queue:
        return None
    count = Counter(queue)
    return count.most_common(1)[0][0]

# -------------------------------
# Main Live Performance Test Functionality
# -------------------------------
import threading

def send_gesture_nonblocking(gesture: str):
    # Launch the async send in a background thread.
    threading.Thread(
        target=lambda: asyncio.run(send_gesture_to_server(gesture)),
        daemon=True  # Use daemon=True to not block program exit.
    ).start()

def performance_test_live():
    model = load_trained_model()
    expected_dim = model.weights[0].shape[0]

    training_csv = os.path.join("demo_data/mandatory/train", "demo_video_csv_with_ground_truth_rotate.csv")
    if os.path.exists(training_csv):
        X_mean, X_std = compute_norm_params(training_csv)
        print("Normalization parameters computed from training CSV.")
    else:
        X_mean = 0
        X_std = 1

    # Load PCA parameters if they exist.
    pca_params_path = os.path.join("models", "pca_params_optional.pkl")
    if os.path.exists(pca_params_path):
        with open(pca_params_path, "rb") as f:
            pca_mean, pca_components = pickle.load(f)
        print("Loaded PCA parameters for live inference.")
    else:
        pca_mean, pca_components = None, None
    
    idx_to_label = {
        0: "left",
        1: "right",
        2: "rotate",
        3: "idle",
        4: "flip_table",
        5: "point",
        6: "zoom_in",
        7: "zoom_out"
    }
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    window_size = 5
    prediction_queue = deque(maxlen=window_size)
    last_displayed = None

    print("Starting live performance test. Press ESC to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = pose.process(rgb_frame)
        rgb_frame.flags.writeable = True
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            raw_feature_vector = construct_feature_vector(results.pose_landmarks, expected_dim=132)
            if raw_feature_vector is not None:
                # Normalize features if applicable.
                features_norm = (raw_feature_vector - X_mean) / X_std if not (np.all(X_mean == 0) and np.all(X_std == 1)) else raw_feature_vector
                
                # Apply PCA transformation if PCA parameters exist.
                if pca_mean is not None and pca_components is not None:
                    features_norm = np.dot(features_norm - pca_mean, pca_components)
                
                # Now features_norm should have shape (1, expected_dim)
                probs = model.forward(features_norm)
                pred_idx = np.argmax(probs, axis=1)[0]
                gesture = idx_to_label.get(pred_idx, "unknown")
                confidence = probs[0][pred_idx]
                print(f"Predicted gesture: {gesture} (confidence: {confidence:.2f})")

                # Update prediction queue.
                if confidence > 0.8 and gesture != "idle":
                    prediction_queue.append(gesture)
                else:
                    prediction_queue.append("uncertain")

                smoothed_prediction = majority_vote(prediction_queue)

                if smoothed_prediction != last_displayed:
                    last_displayed = smoothed_prediction
                    if smoothed_prediction != "uncertain":
                        send_gesture_nonblocking(smoothed_prediction)

                cv2.putText(frame, f"Gesture: {smoothed_prediction}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow("Live Gesture Prediction", frame)


        if cv2.waitKey(5) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
   performance_test_live()  
