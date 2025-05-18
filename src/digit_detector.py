import os
import re # Regular expressions for filename parsing
import cv2
import numpy as np
from PIL import Image
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # Good general-purpose classifier
from sklearn.metrics import classification_report, accuracy_score
from skimage.feature import hog # Histogram of Oriented Gradients
import joblib
import pandas as pd

# --- Configuration ---
# --- PATHS (IMPORTANT: Adjust these relative to where you RUN the script from, or use absolute paths) ---
# Assuming you run the script from the `digit_recognizer` root folder:
IMAGE_FOLDER = 'images/'
ARFF_FILE_PATH = 'data/your_digit_data.arff' # Path to your ARFF file if you intend to use it
MODEL_SAVE_DIR = 'models/'
MODEL_FILENAME = os.path.join(MODEL_SAVE_DIR, 'digit_classifier_model.joblib')
FEATURE_PARAMS_FILENAME = os.path.join(MODEL_SAVE_DIR, 'feature_extractor_params.joblib')

# --- IMAGE PROCESSING & FEATURE EXTRACTION ---
IMAGE_SIZE = (28, 28) # Resize images to this (e.g., MNIST-like)
USE_HOG_FEATURES = True # True for HOG, False for raw pixel values

# --- DATA LOADING MODE ---
# "FROM_FILENAMES": Labels from image filenames, features from images.
# "FROM_ARFF_FEATURES": Labels AND features from ARFF (images in IMAGE_FOLDER are ignored for training).
# Choose one:
DATA_LOADING_MODE = "FROM_FILENAMES"
# DATA_LOADING_MODE = "FROM_ARFF_FEATURES"


# --- Helper Functions ---
def ensure_dir(directory):
    """Creates a directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def parse_label_from_filename(filename):
    """
    Extracts the digit label from filenames like 'G1-B0.jpg' or 'G2_E6.png'.
    It looks for the last digit before the extension.
    """
    # Remove extension
    name_part = os.path.splitext(filename)[0]
    
    # Regex to find a digit that is either at the end of the string (after a letter)
    # or preceded by a separator and a letter.
    # This regex looks for [_-] followed by an optional letter, then a digit.
    # Or, a more general one: just the last digit in the string.
    match = re.search(r'(\d)$', name_part) # Finds the last digit in the name part
    
    if match:
        return int(match.group(1))
    else:
        # Fallback: try to get the character just before the (potential) letter before the digit
        # e.g. G1-B0 -> 0, X_Y7 -> 7
        # This is trickier if format varies a lot. The regex above is safer.
        # For simplicity, we'll rely on the regex. If it fails, we return None.
        print(f"Warning: Could not parse label from filename: {filename}")
        return None

def preprocess_image(image_path, target_size):
    """Loads an image, converts to grayscale, resizes, and normalizes."""
    try:
        pil_image = Image.open(image_path).convert('L') # 'L' for grayscale
        img = np.array(pil_image)
        img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        img_normalized = img_resized / 255.0
        return img_normalized
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def extract_features(image_array):
    """Extracts features from a preprocessed image array."""
    if USE_HOG_FEATURES:
        # These HOG parameters are common defaults for digits.
        # pixels_per_cell should ideally be a divisor of IMAGE_SIZE dimensions.
        ppc_x = IMAGE_SIZE[0] // 4 # e.g., 28//4 = 7
        ppc_y = IMAGE_SIZE[1] // 4 # e.g., 28//4 = 7
        features = hog(image_array, orientations=9, pixels_per_cell=(ppc_x, ppc_y),
                       cells_per_block=(2, 2), visualize=False, block_norm='L2-Hys')
    else:
        features = image_array.flatten() # Raw pixel values
    return features

def load_data_from_image_filenames(image_folder_path, target_size):
    """Loads images, extracts features, and gets labels from filenames."""
    features_list = []
    labels_list = []
    print(f"Loading images from: {image_folder_path}")
    if not os.path.isdir(image_folder_path):
        print(f"Error: Image folder not found at {image_folder_path}")
        return None, None
        
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(image_folder_path) if f.lower().endswith(valid_extensions)]

    if not image_files:
        print(f"No images found in {image_folder_path}")
        return None, None

    for img_name in image_files:
        label = parse_label_from_filename(img_name)
        if label is None:
            print(f"Skipping {img_name} due to label parsing error.")
            continue

        img_path = os.path.join(image_folder_path, img_name)
        processed_img_array = preprocess_image(img_path, target_size)
        
        if processed_img_array is not None:
            img_features = extract_features(processed_img_array)
            features_list.append(img_features)
            labels_list.append(label)
        else:
            print(f"Skipping {img_name} due to image processing error.")

    if not features_list:
        print("No features were extracted. Check images and processing steps.")
        return None, None
        
    return np.array(features_list), np.array(labels_list)

def load_data_from_arff_features(arff_file_path):
    """Loads features and labels directly from an ARFF file."""
    print(f"Loading data from ARFF file: {arff_file_path}")
    if not os.path.exists(arff_file_path):
        print(f"Error: ARFF file not found at {arff_file_path}")
        return None, None
    try:
        data, meta = arff.loadarff(arff_file_path)
        df = pd.DataFrame(data)
        
        # Assume class attribute is the last one
        class_column_name = meta.names()[-1]
        
        # Decode if labels are byte strings and convert to int
        if df[class_column_name].dtype == object:
            try:
                df[class_column_name] = df[class_column_name].str.decode('utf-8').astype(int)
            except AttributeError: # Already strings
                df[class_column_name] = df[class_column_name].astype(int)
            except Exception as e_decode:
                print(f"Warning: Could not decode/convert class column '{class_column_name}' to int: {e_decode}")
                # Fallback: if they are already '0', '1' as strings.
                try:
                    df[class_column_name] = df[class_column_name].astype(int)
                except Exception as e_convert:
                     print(f"Error: Class column '{class_column_name}' could not be converted to int: {e_convert}")
                     return None, None


        labels = df[class_column_name].values
        
        # Features are all columns except the class column
        feature_columns = [col for col in df.columns if col != class_column_name]
        features = df[feature_columns].values.astype(np.float32)

        print(f"  ARFF Data Loaded: {features.shape[0]} samples, {features.shape[1]} features.")
        # IMPORTANT: If using ARFF features, the USE_HOG_FEATURES flag becomes less relevant for training.
        # However, for PREDICTING new raw images, you'd need a way to extract features
        # consistent with what's in the ARFF. This script assumes if ARFF features are used for training,
        # new image prediction will use the HOG/raw pixel settings defined globally.
        # This is a mismatch if ARFF features are something else entirely.
        # A more advanced system would store metadata about ARFF feature type.
        print("  WARNING: If training on ARFF features, ensure the 'extract_features' function used for NEW image prediction is consistent with ARFF feature type or that ARFF features are HOG/raw pixels matching current settings.")

        return features, labels
    except Exception as e:
        print(f"Error loading ARFF file {arff_file_path}: {e}")
        return None, None

# --- Main Script ---
def main():
    ensure_dir(MODEL_SAVE_DIR) # Create models directory if it doesn't exist

    X = None
    y = None

    print(f"--- Digit Recognition Training ---")
    print(f"Data Loading Mode: {DATA_LOADING_MODE}")

    if DATA_LOADING_MODE == "FROM_FILENAMES":
        print(f"Using HOG features: {USE_HOG_FEATURES}")
        X, y = load_data_from_image_filenames(IMAGE_FOLDER, IMAGE_SIZE)
    elif DATA_LOADING_MODE == "FROM_ARFF_FEATURES":
        X, y = load_data_from_arff_features(ARFF_FILE_PATH)
        # Note: If ARFF is used, USE_HOG_FEATURES is for *new image prediction consistency*,
        # not for training data loading here.
        print(f"For predicting new images, HOG features will be: {USE_HOG_FEATURES}")
    else:
        print(f"Error: Invalid DATA_LOADING_MODE: {DATA_LOADING_MODE}")
        return

    if X is None or y is None or len(X) == 0:
        print("Failed to load data. Exiting.")
        return

    print(f"\nData loaded: {X.shape[0]} samples, {X.shape[1]} features per sample.")
    print(f"Unique labels found: {np.unique(y)}")
    print(f"Label distribution: {pd.Series(y).value_counts().sort_index()}")

    if len(np.unique(y)) < 2:
        print("Error: Not enough classes found in the data for training. Need at least 2.")
        return
    if X.shape[0] < 10 : # Arbitrary small number
        print("Warning: Very few samples loaded. Model performance might be poor.")


    # 1. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"\nTraining samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    # 2. Train Model
    print("\nTraining model (RandomForestClassifier)...")
    # model = SVC(kernel='rbf', probability=True, random_state=42) # Alternative: SVM
    model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1, class_weight='balanced')
    model.fit(X_train, y_train)
    print("Model training complete.")

    # 3. Save Model and Feature Parameters
    joblib.dump(model, MODEL_FILENAME)
    print(f"Model saved to: {MODEL_FILENAME}")

    # Save parameters used for feature extraction, crucial for consistent prediction
    feature_params = {
        'image_size': IMAGE_SIZE,
        'used_hog': USE_HOG_FEATURES, # This reflects how new images should be processed
        'data_loading_mode_for_training': DATA_LOADING_MODE
    }
    if USE_HOG_FEATURES: # Store HOG params only if HOG was intended for new images
        feature_params.update({
            'hog_orientations': 9,
            'hog_pixels_per_cell': (IMAGE_SIZE[0] // 4, IMAGE_SIZE[1] // 4),
            'hog_cells_per_block': (2, 2),
            'hog_block_norm': 'L2-Hys'
        })
    joblib.dump(feature_params, FEATURE_PARAMS_FILENAME)
    print(f"Feature extractor parameters saved to: {FEATURE_PARAMS_FILENAME}")

    # 4. Evaluate Model
    print("\nEvaluating model on the test set...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Set Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("\n--- Training complete ---")


def predict_single_image(image_path_to_predict):
    """Loads the trained model and predicts the digit for a single image."""
    print(f"\n--- Predicting Digit for: {image_path_to_predict} ---")
    if not os.path.exists(MODEL_FILENAME) or not os.path.exists(FEATURE_PARAMS_FILENAME):
        print("Error: Model or feature parameters file not found. Train the model first.")
        return

    # Load model and feature parameters
    model = joblib.load(MODEL_FILENAME)
    feature_params = joblib.load(FEATURE_PARAMS_FILENAME)

    _image_size = tuple(feature_params['image_size'])
    _use_hog_for_prediction = feature_params['used_hog'] # This is key!

    # Preprocess the new image
    preprocessed_img_array = preprocess_image(image_path_to_predict, _image_size)
    if preprocessed_img_array is None:
        print("Failed to preprocess the image.")
        return

    # Extract features (CONSISTENTLY with how model was trained or intended for prediction)
    if _use_hog_for_prediction:
        # If HOG params were saved, use them explicitly. Otherwise, use global settings.
        img_features = hog(preprocessed_img_array,
                           orientations=feature_params.get('hog_orientations', 9),
                           pixels_per_cell=tuple(feature_params.get('hog_pixels_per_cell', (_image_size[0]//4, _image_size[1]//4))),
                           cells_per_block=tuple(feature_params.get('hog_cells_per_block', (2,2))),
                           visualize=False,
                           block_norm=feature_params.get('hog_block_norm', 'L2-Hys'))
    else:
        img_features = preprocessed_img_array.flatten()
    
    # Reshape for single prediction
    img_features_reshaped = img_features.reshape(1, -1)

    # Predict
    prediction = model.predict(img_features_reshaped)
    proba = model.predict_proba(img_features_reshaped)

    print(f"Predicted Digit: {prediction[0]}")
    print(f"Prediction Probabilities: {dict(zip(model.classes_, proba[0]))}")
    return prediction[0]


if __name__ == "__main__":
    # To train the model:
    main() # This trains on YOUR images first

    # --- Prediction on a specific image from your dataset (after training) ---
    # 1. Make sure your model is trained (the main() above does this).
    # 2. Change 'your_image_filename.jpg' to an actual image file in your IMAGE_FOLDER
    #    or any other image path you want to test.
    
    print("\n--- Prediction on a specific image ---")
    if os.path.exists(MODEL_FILENAME): # Check if model exists (it should after main())
        # ===> PUT THE FILENAME OF YOUR TEST IMAGE HERE <===
        # This image should be in your IMAGE_FOLDER or provide the full path
        test_image_filename = 'nishat.jpg' # <--- !!! EXAMPLE: CHANGE THIS !!!
        # For example, if you have an image 'my_test_digit_3.png' in 'images/' folder:
        # test_image_filename = 'my_test_digit_3.png' 
        # Or if it's somewhere else:
        # image_path_to_test = '/path/to/your/other_image.png'

        image_path_to_test = os.path.join(IMAGE_FOLDER, test_image_filename) # Constructs path if in IMAGE_FOLDER

        if os.path.exists(image_path_to_test):
            print(f"Attempting to predict: {image_path_to_test}")
            predicted_digit = predict_single_image(image_path_to_test)
            if predicted_digit is not None:
                print(f"The model predicted the digit in '{image_path_to_test}' is: {predicted_digit}")
            else:
                print(f"Could not get a prediction for '{image_path_to_test}'.")
        else:
            print(f"Test image not found: {image_path_to_test}")
            print(f"Make sure '{test_image_filename}' exists in '{IMAGE_FOLDER}' or you provided a correct full path.")
    else:
        print("Model not found. Train the model first by running the script.")
