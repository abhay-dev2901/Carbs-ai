# Type 1 Diabetes Carb Estimation System - v1

**AI/ML system for gram-precise net carbohydrate estimation and glycemic load prediction from meal photos for Type 1 diabetics.**

## Problem Statement

Type 1 diabetics require precise carbohydrate counting for insulin dosing (1 unit per 10-15g carbs). Manual carb counting has ±20-30% errors, causing blood glucose fluctuations. This system aims to achieve ±10g accuracy (v1 target) to support safer insulin dosing decisions.

## Dataset

- **5 Indian dishes**: biryani, dal, halwa, poha, roti
- **250 images total** (50 per dish)
- **80% training** (200 images)
- **20% validation** (50 images)
- **Nutrition data**: Net carbs and glycemic index for each dish

## Enhanced Model Architecture

### Classifier
- **Base**: MobileNetV2 (ImageNet pretrained, frozen)
- **Head**: Global Average Pooling → BatchNorm → Dense(256) → Dropout(0.5) → Dense(128) → Dropout(0.3) → Softmax
- **Training**: Enhanced data augmentation, early stopping, learning rate scheduling

### Regressor
- **Feature Extractor**: Frozen classifier features (GAP layer)
- **Head**: BatchNorm → Dense(256) → Dropout(0.4) → Dense(128) → Dropout(0.3) → Dense(64) → Dropout(0.2) → Linear
- **Loss**: Huber loss (robust to outliers)
- **Training**: Comprehensive evaluation metrics, early stopping

## Key Improvements (v1 Enhancements)

✅ **Fixed critical bug** in regressor label mapping  
✅ **Enhanced architecture** with deeper networks and regularization  
✅ **Improved data augmentation** (rotation, shear, brightness, zoom)  
✅ **Early stopping & learning rate scheduling** for better convergence  
✅ **Comprehensive evaluation** (MAE, RMSE, R², error distribution, ±10g accuracy)  
✅ **Glycemic load prediction** from GI and net carbs  
✅ **Insulin dosing recommendations** with customizable ratios  
✅ **Clinical context** in predictions (blood glucose impact estimates)

## Performance Metrics

After retraining with enhanced models, evaluate using:

```bash
python src/evaluate.py
```

**Target Metrics:**
- Classifier accuracy: >90%
- Regressor MAE: <10g (v1 target)
- Within ±10g accuracy: >80%

## Usage

### 1. Train Models

```bash
# Train classifier first
python src/train_classifier.py

# Train regressor (uses classifier features)
python src/train_regressor.py
```

### 2. Make Predictions

```bash
# Basic prediction
python src/predict.py data/images/biryani/0d81432b55.jpg

# With custom insulin ratio (e.g., 1 unit per 10g carbs)
python src/predict.py data/images/biryani/0d81432b55.jpg --insulin-ratio 10
```

**Output includes:**
- Identified dish with confidence
- Net carbohydrates (grams)
- Glycemic Index (GI)
- Glycemic Load (GL) with classification
- Insulin dose recommendation
- Clinical safety notes

### 3. Evaluate Models

```bash
python src/evaluate.py
```

Provides comprehensive metrics including:
- Classification accuracy and confusion matrix
- Regression metrics (MAE, RMSE, R²)
- Error distribution analysis
- Per-class performance
- Clinical significance assessment

## Clinical Context

- **Target accuracy**: ±10g for safe insulin dosing
- **Manual counting error**: ±20-30% typically
- **Blood glucose impact**: 5g carb error ≈ 25-50 mg/dL change
- **Insulin ratio**: Typically 1 unit per 10-15g carbs (individual varies)

## Project Structure

```
Carbs-ai/
├── data/
│   ├── images/          # Training images by dish
│   └── nutrition.csv    # Net carbs and GI data
├── models/              # Trained model files
│   ├── classifier.h5
│   └── regressor.h5
└── src/
    ├── train_classifier.py
    ├── train_regressor.py
    ├── predict.py
    ├── evaluate.py
    └── utils/
        └── diabetic_check.py
```

## Requirements

```
tensorflow==2.15.0
pandas==2.2.2
numpy==1.26.4
Pillow==10.4.0
scikit-learn==1.5.1
```

## Important Notes

⚠️ **This is a v1 model for research/decision support**  
⚠️ **Always consult healthcare providers for insulin dosing**  
⚠️ **Individual insulin ratios vary (10-15g per unit)**  
⚠️ **Model trained on limited dataset (5 dishes, 250 images)**  
⚠️ **For production use, expand dataset and validate clinically**

## Future Enhancements

- [ ] Expand dataset with more dishes and variations
- [ ] Portion size estimation
- [ ] Multi-dish meal support
- [ ] Real-time mobile app
- [ ] Clinical validation study
- [ ] Fine-tuning for individual user patterns
