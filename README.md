# AI-Based Waste Classification System

A beginner-friendly AI/ML project prototype aligned with **SDG 12: Responsible Consumption and Production**. It classifies one main waste item in an image into categories represented in the training dataset.

## Features
- Five intended categories: organic, paper, plastic, glass, and metal.
- Train a CNN image classifier using a labelled local dataset.
- Upload an image through a Streamlit interface.
- Display predicted category, estimated model confidence, and general disposal guidance.
- Warn users when a prediction has low confidence.

**Important:** The source code is provided, but a trained model and labelled dataset are not included. The model must be trained on suitable images before the app can make predictions. Do not claim accuracy or real-world impact until you measure it.

## 1. Requirements
Recommended: Python 3.10 or 3.11. Training may be slow on a computer without a GPU.

Open a terminal in this project folder and create a virtual environment:

### Windows
```bash
py -3.11 -m venv .venv
.venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / WSL / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If TensorFlow installation fails, check the TensorFlow compatibility information for your operating system and Python version, then use a supported Python version.

## 2. Prepare the dataset
Create this folder structure and place labelled JPG, JPEG, or PNG images inside each folder:

```text
AI_Waste_Classification_Project/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
└── dataset/
    ├── organic/
    ├── paper/
    ├── plastic/
    ├── glass/
    └── metal/
```

Use images that you have permission to use. Ensure each image's folder label is correct. For a useful initial experiment, aim for at least 100 varied images per category if possible; more and more diverse examples are generally better. Include different backgrounds, lighting, orientations, and item appearances. This is a practical guideline, not a guarantee of accuracy.

You may use a suitable public dataset only after checking its licence, category definitions, and whether its labels match these folders. Do not mix images from the same source/photo sequence across training and evaluation in a way that leaks near-duplicates.

## 3. Train the model
With the virtual environment active, run:

```bash
python train_model.py
```

The script creates:
- `model/waste_classifier.keras` — trained model
- `model/class_names.json` — category names in the model's class order

It uses an 80/20 training/validation split and early stopping. The printed training/validation accuracy is not an independent test result. For a formal project report, reserve a separate test set and evaluate it only after training.

## 4. Run the web app
After training finishes:

```bash
streamlit run app.py
```

Streamlit will show a local address in the terminal (usually `http://localhost:8501`). Open that address in your browser and upload a clear image of one item.

## 5. How the system works
1. The user uploads a photo.
2. The image is converted to RGB and resized to 128 × 128 pixels.
3. The CNN predicts class probabilities.
4. The app displays the highest-scoring category and a general disposal tip.
5. Low-confidence predictions receive a warning.

## 6. Suggested evaluation for your report
- Create a separate test folder that is not used in training or validation.
- Report overall test accuracy and per-class precision/recall.
- Include a confusion matrix and a few examples of incorrect predictions.
- Test under varied backgrounds, lighting, distances, and item orientations.
- Record the dataset name, source, licence, number of images per class, hardware, Python version, and training settings.
- Report the measured results honestly; do not invent an accuracy percentage.

## 7. Limitations and responsible use
- The classifier only knows the categories present in its training data.
- Dirty, crushed, partly hidden, mixed-material, or unusual items may be misclassified.
- A category prediction does not establish whether an item is recyclable in a particular municipality.
- Hazardous, medical, and electronic waste require appropriate specialist/local guidance.
- Avoid collecting personal images and respect dataset licences and privacy.

## 8. Suggested project demonstration
1. Introduce the waste-sorting problem and SDG 12.
2. Show the dataset folder structure and training command.
3. Open the Streamlit app and upload several test images.
4. Show predictions and at least one limitation/error.
5. Present actual test metrics and future improvements.

## Troubleshooting
- **“The trained model is not available yet”**: prepare the dataset and run `python train_model.py`.
- **No images found**: check that each category folder contains supported image files.
- **Poor predictions**: add diverse correctly labelled images, check class balance, and evaluate on images not used for training.
- **Package errors**: use a supported Python/TensorFlow combination and install requirements inside the active virtual environment.
