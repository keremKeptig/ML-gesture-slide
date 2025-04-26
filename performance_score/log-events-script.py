import os
import argparse
import sys
import numpy as np
import pandas as pd
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework.model import NeuralNetwork

# Example parameters:
#   --input_csv=demo_data/demo_video_csv_with_ground_truth_rotate.csv
#   --output_csv_name=performance_results.csv
#   --model_path=models/model.npz
#   --pca_path=models/pca_params.pkl

# Available gestures as defined in your code
AVAILABLE_GESTURES = ["idle", "swipe_left", "swipe_right", "rotate"]

class GestureDetectionApp:
    def __init__(self, model_path="../models/best_model_mandatory.npz", pca_path="models/pca_params_mandatory.pkl"):
        """
        Initialize the gesture detection application.
        
        Args:
            model_path: Path to the trained model weights
            pca_path: Path to the saved PCA parameters (if PCA was used)
        """
        # Load the trained model if available
        self.model = self._load_model(model_path)
        # Load PCA parameters if available
        self.pca_params = self._load_pca_params(pca_path)
        
        # Gesture mapping consistent with preprocess.py
        self.gesture_map = {"swipe_left": 0, "swipe_right": 1, "rotate": 2, "idle": 3}
        self.gesture_map_inv = {v: k for k, v in self.gesture_map.items()}
        
        # Tracking state for gesture detection
        self.last_detected_gesture = "idle"
        self.gesture_already_emitted = False
        self.cooldown_frames = 30  # Frames to wait after detecting a gesture
        self.cooldown_counter = 0
        
        # Window for feature extraction
        self.window_size = 15  # Number of frames to consider for gesture detection
        self.past_frames = []

    def _load_model(self, model_path):
        """
        Dynamically load the trained model with multiple layers.
        """
        try:
            model_data = np.load(model_path)
            # Get all weight keys sorted by layer order (e.g., "W1", "W2", ...)
            weight_keys = sorted(
                [key for key in model_data.keys() if key.startswith('W')],
                key=lambda x: int(x[1:])
            )
            n_weights = len(weight_keys)
            # n_layers for NeuralNetwork is defined as number_of_weights - 1 (since the last weight is the output layer)
            n_layers = n_weights - 1

            # Infer dimensions from the first and last weight matrices.
            input_dim = model_data[weight_keys[0]].shape[0]
            hidden_dim = model_data[weight_keys[0]].shape[1]
            output_dim = model_data[weight_keys[-1]].shape[1]
            print(f"Model dimensions: input={input_dim}, hidden={hidden_dim}, output={output_dim}, n_layers={n_layers}")

            # Initialize the model with the dynamic number of hidden layers.
            model = NeuralNetwork(input_dim, hidden_dim, output_dim, n_layers=n_layers)

            # Load weights for each layer.
            for i, key in enumerate(weight_keys):
                model.weights[i] = model_data[key]

            # Similarly, load biases.
            bias_keys = sorted(
                [key for key in model_data.keys() if key.startswith('b')],
                key=lambda x: int(x[1:])
            )
            for i, key in enumerate(bias_keys):
                model.biases[i] = model_data[key]

            print(f"Loaded model from {model_path}")
            return model

        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using random features for demonstration purposes.")
            return None


    def _load_pca_params(self, pca_path):
        """
        Load PCA parameters if available.
        """
        try:
            with open(pca_path, 'rb') as f:
                pca_params = pickle.load(f)
            print(f"Loaded PCA parameters from {pca_path}")
            return pca_params
        except Exception as e:
            print(f"No PCA parameters found or error loading them: {e}")
            return None

    def _extract_features(self, frame):
        """
        Extract features from the current frame.
        We assume the frame has the same structure as the training data.
        """
        # Extract feature columns (excluding timestamp, events, ground_truth)
        feature_columns = [col for col in frame.index if col not in 
                          ["timestamp", "events", "ground_truth"]]
        
        # Convert to numpy array
        features = frame[feature_columns].values.astype(np.float32)
        
        # Apply PCA transformation if available
        if self.pca_params is not None:
            mean_X, components = self.pca_params
            features = np.dot(features - mean_X, components)
            
        return features.reshape(1, -1)  # Reshape for a single sample

    def _predict_gesture(self, features):
        """
        Predict the gesture using the trained model.
        """
        if self.model is None:
            # If no model is loaded, use random prediction for demonstration
            return np.random.choice(AVAILABLE_GESTURES, p=[0.94, 0.02, 0.02, 0.02])
        
        # Make prediction
        prediction = self.model.predict(features)[0]
        return self.gesture_map_inv.get(prediction, "idle")

    def detect_gesture(self, frame):
        """
        Detect a gesture from the current frame.
        
        Args:
            frame: Current frame data (pandas Series)
            
        Returns:
            String: Detected gesture or "idle"
        """
        # If we're in cooldown, decrement counter and return idle
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return "idle"
        
        try:
            # Extract features from the current frame
            features = self._extract_features(frame)
            
            # Make prediction
            detected_gesture = self._predict_gesture(features)
            
            # Handle gesture state tracking
            if detected_gesture != "idle" and not self.gesture_already_emitted:
                # When a new gesture is detected, emit it and start cooldown
                self.gesture_already_emitted = True
                self.cooldown_counter = self.cooldown_frames
                self.last_detected_gesture = detected_gesture
                return detected_gesture
            elif detected_gesture == "idle" and self.last_detected_gesture != "idle":
                # Reset tracking when returning to idle
                self.gesture_already_emitted = False
                self.last_detected_gesture = "idle"
            
            return "idle"
        except Exception as e:
            print(f"Error in gesture detection: {e}")
            return "idle"

    def compute_events(self, frames):
        """
        Process all frames and detect events.
        
        Args:
            frames: DataFrame containing all frames
            
        Returns:
            List of detected events for each frame
        """
        events = []
        
        # Process frames one by one (simulating live processing)
        for idx, frame in frames.iterrows():
            # Detect gesture for this frame
            event = self.detect_gesture(frame)
            events.append(event)
            
            # Add current frame to past frames for feature extraction
            self.past_frames.append(frame)
            
            # Limit the number of past frames
            if len(self.past_frames) > self.window_size:
                self.past_frames.pop(0)
        
        return events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", help="CSV file containing the video transcription from OpenPose", required=True)
    parser.add_argument("--output_csv_name", help="output CSV file containing the events", default="performance_results.csv")
    parser.add_argument("--model_path", help="Path to the trained model", default="../models/best_model_mandatory.npz")
    parser.add_argument("--pca_path", help="Path to the saved PCA parameters", default="../models/pca_params_mandatory.pkl")
    
    args = parser.parse_known_args()[0]
    
    input_path = args.input_csv
    
    # Fix the output path to use correct path separators and avoid absolute path issues
    output_directory = os.path.dirname(os.path.abspath(input_path))
    output_path = os.path.join(output_directory, args.output_csv_name)
    
    # Debug output
    print(f"Input CSV: {input_path}")
    print(f"Output will be written to: {output_path}")
    
    # Read the input CSV
    try:
        print(f"Reading input CSV: {input_path}")
        frames = pd.read_csv(input_path)
        print(f"Successfully read {len(frames)} frames")
        
        # Set timestamp as index if it exists as a column
        if 'timestamp' in frames.columns:
            frames = frames.set_index("timestamp")
            frames.index = frames.index.astype(int)
            print("Set timestamp as index")
    except Exception as e:
        print(f"Error reading input CSV: {e}")
        return
    
    # ================================= your application =============================
    # Initialize the gesture detection application with your trained model
    print("Initializing GestureDetectionApp...")
    my_model = GestureDetectionApp(model_path=args.model_path, pca_path=args.pca_path)
    
    # ================================================================================
    
    # Compute events
    print("Computing events...")
    frames["events"] = my_model.compute_events(frames)
    print("Events computed successfully")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # the CSV has to have the columns "timestamp" and "events"
    # but may also contain additional columns, which will be ignored during the score evaluation
    print(f"Writing events to {output_path}")
    try:
        if 'timestamp' in frames.columns:
            frames[["events"]].to_csv(output_path, index=False)
        else:
            frames[["events"]].to_csv(output_path, index=True)
        print(f"Events exported to {output_path}")
    except Exception as e:
        print(f"Error writing CSV: {e}")
        # Try writing to current directory as fallback
        fallback_path = "performance_results.csv"
        print(f"Trying fallback path: {fallback_path}")
        frames[["events"]].to_csv(fallback_path, index=True)
        print(f"Events exported to fallback location: {fallback_path}")

if __name__ == "__main__":
    print("Current working directory:", os.getcwd())
    main()