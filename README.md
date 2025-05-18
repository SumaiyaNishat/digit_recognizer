# Digit Recognizer Project

This project detects digits (0-9) from images.

## Features

*   Loads images from a single folder.
*   Extracts digit labels directly from image filenames (e.g., `G1-B0.jpg` -> `0`).
*   Option to load pre-extracted features from an ARFF file.
*   Uses HOG (Histogram of Oriented Gradients) or raw pixel values as features.
*   Trains a RandomForestClassifier model.
*   Saves the trained model for later use.
*   Provides a function/script for predicting digits from new images.

## Project Structure

```txt
digit_recognizer/
├── images/                     # Contains all training/raw digit images
├── data/                       # For ARFF files or other data
├── src/                        # Source code
│   ├── digit_detector.py       # Main script for training and example prediction
│   └── predict.py              # Standalone script for prediction
├── models/                     # Trained models are saved here
└── requirements.txt            # Python package dependencies
└── README.md                   # This file
```

## Setup

1.  **Clone/Download the repository.**
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # macOS/Linux
    # venv\Scripts\activate    # Windows
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Place your images:**
    Put all your digit image files (e.g., `G1-B0.jpg`, `anyname_1.png`) into the `images/` folder.
5.  **(Optional) Place ARFF file:**
    If using `DATA_LOADING_MODE = "FROM_ARFF_FEATURES"`, place your ARFF file in the `data/` folder and update the `ARFF_FILE_PATH` in `src/digit_detector.py`.

## Configuration

Edit `src/digit_detector.py` to configure:
*   `IMAGE_FOLDER`: Path to the image directory.
*   `ARFF_FILE_PATH`: Path to the ARFF file (if used).
*   `DATA_LOADING_MODE`: `"FROM_FILENAMES"` or `"FROM_ARFF_FEATURES"`.
*   `USE_HOG_FEATURES`: `True` or `False`.
*   `IMAGE_SIZE`: Target size for image resizing.

## Usage

### 1. Training the Model
Navigate to the project root directory in your terminal and run:
```bash
python src/digit_detector.py
```