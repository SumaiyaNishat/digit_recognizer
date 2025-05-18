# src/predict.py
import os
import sys
import joblib
import numpy as np
from PIL import Image
import cv2 # For resizing
from skimage.feature import hog # For HOG features

# Assuming paths are relative to the project root or use absolute paths
MODEL_DIR = '../models/' # Relative to src/
MODEL_FILENAME = os.path.join(MODEL_DIR, 'digit_classifier_model.joblib')
FEATURE_PARAMS_FILENAME = os.path.join(MODEL_DIR, 'feature_extractor_params.joblib')

# Copy preprocess_image from digit_detector.py
def preprocess_image(image_path, target_size):
    try:
        pil_image = Image.open(image_path).convert('L')
        img = np.array(pil_image)
        img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        img_normalized = img_resized / 255.0
        return img_normalized
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def predict_digit(image_path_to_predict):
    if not os.path.exists(MODEL_FILENAME) or not os.path.exists(FEATURE_PARAMS_FILENAME):
        print("Error: Model or feature parameters file not found. Train the model first using digit_detector.py.")
        return None

    model = joblib.load(MODEL_FILENAME)
    feature_params = joblib.load(FEATURE_PARAMS_FILENAME)

    _image_size = tuple(feature_params['image_size'])
    _use_hog_for_prediction = feature_params['used_hog']

    preprocessed_img_array = preprocess_image(image_path_to_predict, _image_size)
    if preprocessed_img_array is None:
        return None

    if _use_hog_for_prediction:
        img_features = hog(preprocessed_img_array,
                           orientations=feature_params.get('hog_orientations', 9),
                           pixels_per_cell=tuple(feature_params.get('hog_pixels_per_cell', (_image_size[0]//4, _image_size[1]//4))),
                           cells_per_block=tuple(feature_params.get('hog_cells_per_block', (2,2))),
                           visualize=False,
                           block_norm=feature_params.get('hog_block_norm', 'L2-Hys'))
    else:
        img_features = preprocessed_img_array.flatten()
    
    img_features_reshaped = img_features.reshape(1, -1)
    prediction = model.predict(img_features_reshaped)
    proba = model.predict_proba(img_features_reshaped)
    
    print(f"Predicted Digit for '{image_path_to_predict}': {prediction[0]}")
    print(f"Probabilities: {dict(zip(model.classes_, proba[0]))}")
    return prediction[0]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <path_to_image>")
        # Example using a dummy image if no arg provided and in dev
        # This part is for quick testing, remove or adapt for production
        dummy_image_path = "../test_digit_example.png" # Path from src/ to project root
        if not os.path.exists(dummy_image_path):
             print(f"Dummy image {dummy_image_path} not found. Please provide an image path.")
        else:
             print(f"No image path provided. Using dummy image: {dummy_image_path}")
             predict_digit(dummy_image_path)
    else:
        image_to_predict = sys.argv[1]
        if not os.path.exists(image_to_predict):
            print(f"Error: Image not found at {image_to_predict}")
        else:
            predict_digit(image_to_predict)