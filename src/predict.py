# src/predict.py
import numpy as np
from PIL import Image
import sys

# Use keras 3.x directly (models were saved with Keras 3.x)
try:
    import keras
    from keras.models import load_model
    print("Using standalone Keras 3.x")
except ImportError:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import load_model
    print("Using TensorFlow Keras")

# === LOAD MODELS WITHOUT COMPILING (AVOIDS METRIC ERROR) ===
CLASSIFIER = load_model("./models/classifier.h5", compile=False)
REGRESSOR  = load_model("./models/regressor.h5", compile=False)

# === CLASS NAMES (MUST MATCH FOLDER ORDER) ===
CLASS_NAMES = ['biryani', 'dal', 'halwa', 'poha', 'roti']

def load_and_preprocess_image(path):
    img = Image.open(path).convert('RGB').resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

if len(sys.argv) < 2:
    print("Usage: python predict.py <image_path>")
    sys.exit(1)

img = load_and_preprocess_image(sys.argv[1])

# === PREDICT DISH ===
pred_class = np.argmax(CLASSIFIER.predict(img, verbose=0))
dish = CLASS_NAMES[pred_class]

# === PREDICT CARBS ===
pred_carbs = REGRESSOR.predict(img, verbose=0)[0][0]

print(f"Dish: {dish}")
print(f"Net carbs: {pred_carbs:.1f} g")