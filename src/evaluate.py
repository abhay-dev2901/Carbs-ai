# src/evaluate.py
"""
Comprehensive Model Evaluation Script
Evaluates both classifier and regressor on test data
"""
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
import os

DATA_DIR = "./data/images"
MODEL_DIR = "./models"
CSV_PATH = "./data/nutrition.csv"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def evaluate_models():
    """Evaluate both classifier and regressor models"""

    print("Loading models...")
    try:
        classifier = load_model(os.path.join(MODEL_DIR, "classifier.h5"), compile=False)
        regressor = load_model(os.path.join(MODEL_DIR, "regressor.h5"), compile=False)
    except Exception as e:
        print(f"Error loading models: {e}")
        return
    
    nutrition_df = pd.read_csv(CSV_PATH)
    class_to_carbs = dict(zip(nutrition_df['dish'], nutrition_df['net_carbs_g']))

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    val_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        subset='validation',
        shuffle=False,
        class_mode='categorical'
    )
    
    class_names = list(val_gen.class_indices.keys())
    class_indices = val_gen.class_indices
    index_to_carbs = {idx: class_to_carbs[class_name] for class_name, idx in class_indices.items()}
    
    print(f"\n{'='*70}")
    print("MODEL EVALUATION REPORT")
    print(f"{'='*70}\n")
    
    # === CLASSIFIER EVALUATION ===
    print("1. CLASSIFIER EVALUATION")
    print("-" * 70)
    
    y_true_class = []
    y_pred_class = []
    
    val_gen.reset()
    for i in range(len(val_gen)):
        x_batch, y_batch = val_gen[i]
        pred_batch = classifier.predict(x_batch, verbose=0)
        y_true_class.extend(np.argmax(y_batch, axis=1))
        y_pred_class.extend(np.argmax(pred_batch, axis=1))
        if (i + 1) * BATCH_SIZE >= val_gen.samples:
            break
    
    y_true_class = np.array(y_true_class[:val_gen.samples])
    y_pred_class = np.array(y_pred_class[:val_gen.samples])
    
    accuracy = np.mean(y_true_class == y_pred_class) * 100
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"\nClassification Report:")
    print(classification_report(y_true_class, y_pred_class, target_names=class_names))
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_true_class, y_pred_class)
    print(f"{'':<12}", end="")
    for name in class_names:
        print(f"{name[:8]:<12}", end="")
    print()
    for i, name in enumerate(class_names):
        print(f"{name[:8]:<12}", end="")
        for j in range(len(class_names)):
            print(f"{cm[i, j]:<12}", end="")
        print()
    
    # === REGRESSOR EVALUATION ===
    print(f"\n\n2. REGRESSOR EVALUATION")
    print("-" * 70)
    
    y_true_carbs = []
    y_pred_carbs = []
    
    val_gen.reset()
    for i in range(len(val_gen)):
        x_batch, y_batch = val_gen[i]
        class_indices_batch = np.argmax(y_batch, axis=1)
        carb_values = np.array([index_to_carbs[int(idx)] for idx in class_indices_batch])
        
        pred_batch = regressor.predict(x_batch, verbose=0)
        
        y_true_carbs.extend(carb_values)
        y_pred_carbs.extend(pred_batch.flatten())
        
        if (i + 1) * BATCH_SIZE >= val_gen.samples:
            break
    
    y_true_carbs = np.array(y_true_carbs[:val_gen.samples])
    y_pred_carbs = np.array(y_pred_carbs[:val_gen.samples])
    
    # Calculate metrics
    mae = mean_absolute_error(y_true_carbs, y_pred_carbs)
    rmse = np.sqrt(mean_squared_error(y_true_carbs, y_pred_carbs))
    r2 = r2_score(y_true_carbs, y_pred_carbs)
    mape = np.mean(np.abs((y_true_carbs - y_pred_carbs) / y_true_carbs)) * 100
    
    # Accuracy within thresholds
    within_5g = np.sum(np.abs(y_pred_carbs - y_true_carbs) <= 5) / len(y_true_carbs) * 100
    within_10g = np.sum(np.abs(y_pred_carbs - y_true_carbs) <= 10) / len(y_true_carbs) * 100
    within_15g = np.sum(np.abs(y_pred_carbs - y_true_carbs) <= 15) / len(y_true_carbs) * 100
    
    print(f"Mean Absolute Error (MAE): {mae:.2f}g")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}g")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    
    print(f"\nAccuracy within thresholds:")
    print(f"  ±5g:  {within_5g:.1f}%")
    print(f"  ±10g: {within_10g:.1f}% (Target: >80%)")
    print(f"  ±15g: {within_15g:.1f}%")
    
    # Error distribution
    errors = np.abs(y_pred_carbs - y_true_carbs)
    print(f"\nError Distribution:")
    print(f"  Min:    {np.min(errors):.2f}g")
    print(f"  Max:    {np.max(errors):.2f}g")
    print(f"  Median: {np.median(errors):.2f}g")
    print(f"  Mean:   {np.mean(errors):.2f}g")
    print(f"  Std:    {np.std(errors):.2f}g")
    print(f"  75th percentile: {np.percentile(errors, 75):.2f}g")
    print(f"  95th percentile: {np.percentile(errors, 95):.2f}g")
    
    # Per-class analysis
    print(f"\nPer-Class Carb Prediction Analysis:")
    print(f"{'Dish':<12} {'True':<8} {'Pred':<8} {'Error':<8} {'MAE':<8}")
    print("-" * 50)
    for class_name in class_names:
        class_idx = class_indices[class_name]
        mask = y_true_class == class_idx
        if np.sum(mask) > 0:
            true_vals = y_true_carbs[mask]
            pred_vals = y_pred_carbs[mask]
            class_mae = mean_absolute_error(true_vals, pred_vals)
            avg_true = np.mean(true_vals)
            avg_pred = np.mean(pred_vals)
            avg_error = np.mean(np.abs(pred_vals - true_vals))
            print(f"{class_name[:11]:<12} {avg_true:<8.1f} {avg_pred:<8.1f} {avg_error:<8.1f} {class_mae:<8.1f}")
    
    # Clinical significance
    print(f"\n\n3. CLINICAL SIGNIFICANCE")
    print("-" * 70)
    print(f"Target accuracy: ±10g for safe insulin dosing")
    print(f"Current performance: {within_10g:.1f}% within ±10g")
    
    if within_10g >= 80:
        print("Model meets v1 target accuracy threshold")
    else:
        print("Model below v1 target - consider additional training/data")
    
    avg_error = np.mean(errors)
    bg_impact_low = avg_error * 5  
    bg_impact_high = avg_error * 10  
    print(f"\nEstimated blood glucose impact of average error ({avg_error:.1f}g):")
    print(f"  {bg_impact_low:.0f}-{bg_impact_high:.0f} mg/dL")
    
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    evaluate_models()

