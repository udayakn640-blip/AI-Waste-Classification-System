import json
import numpy as np
import tensorflow as tf
from pathlib import Path

# File locations
MODEL_PATH = Path("model/waste_classifier.keras")
LABELS_PATH = Path("model/class_names.json")

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Load category names
class_names = json.loads(
    LABELS_PATH.read_text(encoding="utf-8")
)

# Ask for a photo
image_path = input("Enter the image path: ").strip().strip('"')

# Load and resize the image
image = tf.keras.utils.load_img(
    image_path,
    target_size=(160, 160)
)

# Convert image into an array
image_array = tf.keras.utils.img_to_array(image)
image_array = np.expand_dims(image_array, axis=0)

# Predict the category
predictions = model.predict(image_array, verbose=0)[0]

predicted_index = int(np.argmax(predictions))
predicted_class = class_names[predicted_index]
confidence = float(predictions[predicted_index]) * 100

print("\n===== WASTE CLASSIFICATION RESULT =====")
print("Predicted category:", predicted_class)
print(f"Model confidence: {confidence:.2f}%")