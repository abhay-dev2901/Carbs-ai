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
python src/train_classifier.py

python src/train_regressor.py
```

### 2. Make Predictions

```bash

python src/predict.py data/images/biryani/0d81432b55.jpg
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


