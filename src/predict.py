# src/predict.py
"""
Enhanced Prediction Script for Type 1 Diabetes Carb Estimation
- Dish classification
- Net carbohydrate prediction
- Glycemic load calculation
- Insulin dosing recommendations
"""
import numpy as np
from PIL import Image
import sys
import pandas as pd
import os

# Use keras 3.x directly (models were saved with Keras 3.x)
try:
    import keras
    from keras.models import load_model
except ImportError:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import load_model

# === LOAD MODELS WITHOUT COMPILING (AVOIDS METRIC ERROR) ===
MODEL_DIR = "./models"
CSV_PATH = "./data/nutrition.csv"

try:
    CLASSIFIER = load_model(os.path.join(MODEL_DIR, "classifier.h5"), compile=False)
    REGRESSOR = load_model(os.path.join(MODEL_DIR, "regressor.h5"), compile=False)
except Exception as e:
    print(f"Error loading models: {e}")
    print("Please train the models first using:")
    print("  python src/train_classifier.py")
    print("  python src/train_regressor.py")
    sys.exit(1)

# === LOAD NUTRITION DATA ===
try:
    nutrition_df = pd.read_csv(CSV_PATH)
    class_to_gi = dict(zip(nutrition_df['dish'], nutrition_df['glycemic_index']))
except Exception as e:
    print(f"Warning: Could not load nutrition data: {e}")
    class_to_gi = {}

CLASS_NAMES = ['biryani', 'dal', 'halwa', 'poha', 'roti']

def load_and_preprocess_image(path):
    """Load and preprocess image for model input"""
    try:
        img = Image.open(path).convert('RGB').resize((224, 224))
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

def calculate_glycemic_load(net_carbs, glycemic_index):
    """Calculate glycemic load from net carbs and GI"""
    return (glycemic_index * net_carbs) / 100

def get_gl_classification(gl):
    """Classify glycemic load"""
    if gl >= 20:
        return "High GL — Rapid blood sugar rise ⚠️"
    elif gl <= 10:
        return "Low GL — Gradual blood sugar rise ✅"
    else:
        return "Medium GL — Moderate blood sugar rise ⚠️"

def calculate_insulin_dose(net_carbs, ratio=12):
    """
    Calculate insulin dose recommendation
    Default: 1 unit per 12g carbs (adjustable)
    """
    dose = net_carbs / ratio
    return max(0, round(dose, 1))

def format_output(dish, net_carbs, confidence, gi=None, gl=None, insulin_dose=None):
    """Format and display prediction results"""
    print("\n" + "="*70)
    print("TYPE 1 DIABETES CARB ESTIMATION - PREDICTION RESULTS")
    print("="*70)
    print(f"\n🍽️  Identified Dish: {dish.upper()}")
    print(f"   Confidence: {confidence:.1f}%")
    print(f"\n📊 Net Carbohydrates: {net_carbs:.1f}g")
    
    if gi is not None:
        print(f"📈 Glycemic Index (GI): {gi}")
    
    if gl is not None:
        print(f"📉 Glycemic Load (GL): {gl:.1f}")
        print(f"   {get_gl_classification(gl)}")
    
    if insulin_dose is not None:
        print(f"\n💉 Insulin Recommendation:")
        print(f"   Suggested dose: {insulin_dose} units")
        print(f"   (Based on 1 unit per 12g carbs)")
        print(f"   ⚠️  Always consult with your healthcare provider")
        print(f"   ⚠️  Individual insulin ratios may vary (10-15g per unit)")
    
    print("\n" + "="*70)
    print("CLINICAL NOTES:")
    print("="*70)
    print("• Target accuracy: ±10g for safe insulin dosing")
    print("• Manual carb counting typically has ±20-30% errors")
    print("• 5g carb error ≈ 25-50 mg/dL blood glucose change")
    print("• This is a v1 model - use as decision support tool")
    print("="*70 + "\n")

if len(sys.argv) < 2:
    print("Usage: python predict.py <image_path> [--insulin-ratio RATIO]")
    print("\nExample:")
    print("  python predict.py data/images/biryani/0d81432b55.jpg")
    print("  python predict.py data/images/biryani/0d81432b55.jpg --insulin-ratio 10")
    sys.exit(1)

# Parse insulin ratio if provided
insulin_ratio = 12  # Default: 1 unit per 12g carbs
if len(sys.argv) > 2 and sys.argv[2] == "--insulin-ratio":
    try:
        insulin_ratio = float(sys.argv[3])
    except (IndexError, ValueError):
        print("Warning: Invalid insulin ratio, using default (12g per unit)")

img = load_and_preprocess_image(sys.argv[1])

class_probs = CLASSIFIER.predict(img, verbose=0)[0]
pred_class = np.argmax(class_probs)
dish = CLASS_NAMES[pred_class]
confidence = class_probs[pred_class] * 100

pred_carbs = REGRESSOR.predict(img, verbose=0)[0][0]
pred_carbs = max(0, pred_carbs)  # Ensure non-negative

# === CALCULATE GLYCEMIC LOAD ===
gi = class_to_gi.get(dish, None)
gl = None
if gi is not None:
    gl = calculate_glycemic_load(pred_carbs, gi)

# === CALCULATE INSULIN DOSE ===
insulin_dose = calculate_insulin_dose(pred_carbs, insulin_ratio)

# === DISPLAY RESULTS ===
format_output(dish, pred_carbs, confidence, gi, gl, insulin_dose)