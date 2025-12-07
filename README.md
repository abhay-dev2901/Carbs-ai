# 🍽️ Type 1 Diabetes Carb Estimation System

**An AI/ML-powered system for gram-precise net carbohydrate estimation and glycemic load prediction from meal photos, designed to support safer insulin dosing decisions for Type 1 diabetics.**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Key Features](#-key-features)
- [Performance Metrics](#-performance-metrics)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Guide](#-usage-guide)
- [Project Structure](#-project-structure)
- [Clinical Context](#-clinical-context)
- [Safety & Disclaimers](#-safety--disclaimers)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Problem Statement

### The Challenge

Type 1 diabetes requires precise insulin dosing based on carbohydrate intake. The standard rule is:
- **1 unit of insulin per 10-15g of carbohydrates**

However, manual carbohydrate counting has significant limitations:
- **±20-30% error rate** in typical manual counting
- Each **5g error** can cause **25-50 mg/dL blood glucose fluctuations**
- Poor estimation increases risks of:
  - Hypoglycemia (low blood sugar)
  - Hyperglycemia (high blood sugar)
  - Long-term diabetic complications

### Our Solution

This system uses **computer vision and deep learning** to:
- **Identify Indian dishes** from meal photos
- **Estimate net carbohydrates** with **±10g accuracy** (v1 target)
- **Calculate glycemic load** for better blood sugar management
- **Provide insulin dosing recommendations** as a decision support tool

---

## ✨ Key Features

### 🎯 Core Capabilities

- **Dish Classification**: Identifies 6 Indian dishes with 89% accuracy
  - Biryani, Dal, Halwa, Poha, Rasgulla, Roti
  
- **Carbohydrate Estimation**: Predicts net carbs with 82% accuracy within ±10g
  - Mean Absolute Error: **5.92g**
  - Median Error: **3.05g**
  
- **Glycemic Load Calculation**: 
  - Computes GL from GI and net carbs
  - Classifies as Low/Medium/High GL
  - Provides blood sugar impact warnings
  
- **Insulin Recommendations**:
  - Customizable insulin-to-carb ratios (default: 1 unit per 12g)
  - Individual ratio support (10-15g per unit)
  
- **Clinical Context**: 
  - Blood glucose impact estimates
  - Safety warnings and disclaimers
  - Comparison to manual counting accuracy

### 🔧 Technical Features

- **Enhanced Data Augmentation**: Rotation, shear, zoom, brightness adjustments
- **Robust Training**: Early stopping, learning rate scheduling, model checkpointing
- **Comprehensive Evaluation**: MAE, RMSE, R², error distribution, per-class analysis
- **Production-Ready**: Modular code structure, error handling, validation

---

## 📊 Performance Metrics

### Current Performance (v1)

#### Classifier Performance
- **Accuracy**: 89.0%
- **Top-K Accuracy**: 98.5%
- **Per-Class Performance**:
  - Biryani: 90% precision, 90% recall
  - Dal: 91% precision, 100% recall
  - Halwa: 100% precision, 100% recall
  - Poha: 89% precision, 80% recall
  - Rasgulla: 100% precision, 100% recall
  - Roti: 100% precision, 90% recall

#### Regressor Performance
- **Mean Absolute Error (MAE)**: **5.92g** ✅ (Target: <10g)
- **Root Mean Squared Error (RMSE)**: 8.97g
- **R² Score**: 0.77
- **Mean Absolute Percentage Error (MAPE)**: 18.61%

#### Accuracy Thresholds
- **Within ±5g**: 58.5%
- **Within ±10g**: **82.0%** ✅ (Target: >80%)
- **Within ±15g**: 89.5%

#### Error Distribution
- **Min Error**: 0.03g
- **Max Error**: 29.03g
- **Median Error**: 3.05g
- **75th Percentile**: 8.00g
- **95th Percentile**: 22.52g

#### Per-Class Carb Prediction
| Dish    | True Carbs | Predicted | Error | MAE  |
|---------|------------|-----------|-------|------|
| Biryani | 45.0g      | 39.9g     | 7.4g  | 7.4g |
| Dal     | 12.0g      | 13.3g     | 2.7g  | 2.7g |
| Halwa   | 60.0g      | 52.4g     | 10.6g | 10.6g|
| Poha    | 28.0g      | 30.1g     | 4.0g  | 4.0g |
| Rasgulla| 35.0g      | 38.4g     | 7.5g  | 7.5g |
| Roti    | 18.0g      | 19.2g     | 3.5g  | 3.5g |

### Clinical Significance

- **Target Accuracy**: ±10g for safe insulin dosing ✅ **ACHIEVED**
- **Blood Glucose Impact**: Average error (5.9g) ≈ 30-59 mg/dL change
- **Comparison**: Manual counting typically has ±20-30% errors
- **Status**: **Model meets v1 target accuracy threshold**

---

## 📦 Dataset

### Overview

- **Total Images**: 1,000 images
- **Classes**: 6 Indian dishes
- **Split**: 80% training (800 images), 20% validation (200 images)
- **Source**: Custom dataset + Indian Food Images dataset

### Distribution

| Dish     | Images | Status      |
|----------|--------|-------------|
| Biryani  | 150    | ⚠️ Need 50+ |
| Dal      | 250    | ✅ Good      |
| Halwa    | 250    | ✅ Good      |
| Poha     | 150    | ⚠️ Need 50+ |
| Rasgulla | 50     | ⚠️ Need 150+|
| Roti     | 150    | ⚠️ Need 50+ |

### Nutrition Data

Each dish has associated nutrition information:

| Dish     | Net Carbs (g) | Glycemic Index |
|----------|---------------|----------------|
| Biryani  | 45            | 70             |
| Dal      | 12            | 30             |
| Halwa    | 60            | 75             |
| Poha     | 28            | 65             |
| Rasgulla | 35            | 65             |
| Roti     | 18            | 55             |

### Data Quality

- **Diversity**: Various lighting, angles, backgrounds, portion sizes
- **Validation**: All images validated for readability and quality
- **Augmentation**: Enhanced augmentation during training

---

## 🏗️ Model Architecture

### Classifier Architecture

```
Input (224×224×3)
    ↓
MobileNetV2 (ImageNet pretrained, frozen)
    ↓
Global Average Pooling (GAP)
    ↓
Batch Normalization
    ↓
Dense(256) + ReLU + Dropout(0.5)
    ↓
Dense(128) + ReLU + Dropout(0.3)
    ↓
Dense(6) + Softmax
    ↓
Output: Dish Classification
```

**Training Details**:
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Categorical Crossentropy
- **Callbacks**: Early Stopping, ReduceLROnPlateau, ModelCheckpoint
- **Epochs**: 30 (with early stopping)
- **Data Augmentation**: Rotation, shear, zoom, brightness, horizontal flip

### Regressor Architecture

```
Input (224×224×3)
    ↓
Frozen Classifier Feature Extractor (GAP layer)
    ↓
Batch Normalization
    ↓
Dense(256) + ReLU + Dropout(0.4)
    ↓
Dense(128) + ReLU + Dropout(0.3)
    ↓
Dense(64) + ReLU + Dropout(0.2)
    ↓
Dense(1) + Linear
    ↓
Output: Net Carbohydrates (grams)
```

**Training Details**:
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Huber Loss (robust to outliers)
- **Metrics**: MAE, MSE
- **Callbacks**: Early Stopping, ReduceLROnPlateau, ModelCheckpoint
- **Epochs**: 50 (with early stopping)

### Key Design Decisions

1. **Transfer Learning**: MobileNetV2 pretrained on ImageNet for feature extraction
2. **Frozen Base**: Prevents overfitting on small dataset
3. **Huber Loss**: More robust to outliers than MSE
4. **Progressive Dropout**: Higher dropout in deeper layers
5. **Feature Reuse**: Regressor uses classifier features for efficiency

---

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- 4GB+ RAM (8GB recommended)
- GPU optional but recommended for faster training

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/Carbs-ai.git
cd Carbs-ai
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required Packages**:
```
tensorflow==2.15.0
pandas==2.2.2
numpy==1.26.4
Pillow==10.4.0
scikit-learn==1.5.1
```

### Step 3: Verify Installation

```bash
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"
python -c "import pandas as pd; print(f'Pandas {pd.__version__}')"
```

---

## 🏃 Quick Start

### 1. Train Models

**Train Classifier** (takes ~5-10 minutes):
```bash
python src/train_classifier.py
```

**Train Regressor** (takes ~5-10 minutes):
```bash
python src/train_regressor.py
```

### 2. Make a Prediction

```bash
python src/predict.py data/images/biryani/0d81432b55.jpg
```

**Output Example**:
```
======================================================================
TYPE 1 DIABETES CARB ESTIMATION - PREDICTION RESULTS
======================================================================

🍽️  Identified Dish: BIRYANI
   Confidence: 99.0%

📊 Net Carbohydrates: 30.9g
📈 Glycemic Index (GI): 70
📉 Glycemic Load (GL): 21.7
   High GL — Rapid blood sugar rise ⚠️

💉 Insulin Recommendation:
   Suggested dose: 2.6 units
   (Based on 1 unit per 12g carbs)
```

### 3. Evaluate Models

```bash
python src/evaluate.py
```

---

## 📖 Usage Guide

### Making Predictions

#### Basic Prediction
```bash
python src/predict.py <image_path>
```

#### Custom Insulin Ratio
```bash
python src/predict.py <image_path> --insulin-ratio 10
```
(Default is 12g per unit; adjust for individual needs)

#### Example Predictions
```bash
# Biryani
python src/predict.py data/images/biryani/0d81432b55.jpg

# Dal
python src/predict.py data/images/dal/006aeeea8d.jpg

# Rasgulla
python src/predict.py data/images/rasgulla/<image_name>.jpg
```

### Training Models

#### Classifier Training
```bash
python src/train_classifier.py
```

**What it does**:
- Loads images from `data/images/`
- Applies data augmentation
- Trains MobileNetV2-based classifier
- Saves model to `models/classifier.h5`
- Shows validation accuracy

#### Regressor Training
```bash
python src/train_regressor.py
```

**What it does**:
- Loads trained classifier
- Extracts features from GAP layer
- Trains regression head for carb prediction
- Saves model to `models/regressor.h5`
- Shows comprehensive evaluation metrics

### Evaluating Models

```bash
python src/evaluate.py
```

**Provides**:
- Classification accuracy and confusion matrix
- Regression metrics (MAE, RMSE, R², MAPE)
- Accuracy within thresholds (±5g, ±10g, ±15g)
- Error distribution statistics
- Per-class performance analysis
- Clinical significance assessment

### Expanding Dataset

See `DATASET_GUIDE.md` for detailed instructions on:
- Finding compatible datasets
- Downloading from Kaggle
- Organizing new images
- Validating dataset quality

**Quick expansion**:
```bash
# Organize images from downloaded dataset
python scripts/expand_dataset.py <source_folder> --target data/images

# Check current dataset status
python scripts/expand_dataset.py data/images --count-only
```

---

## 📁 Project Structure

```
Carbs-ai/
├── data/
│   ├── images/              # Training images organized by dish
│   │   ├── biryani/
│   │   ├── dal/
│   │   ├── halwa/
│   │   ├── poha/
│   │   ├── rasgulla/
│   │   └── roti/
│   ├── nutrition.csv        # Net carbs and GI data
│   └── raw/                 # Raw downloaded datasets
│
├── models/                  # Trained model files
│   ├── classifier.h5
│   ├── classifier_best.h5
│   ├── regressor.h5
│   └── regressor_best.h5
│
├── src/
│   ├── train_classifier.py # Classifier training script
│   ├── train_regressor.py  # Regressor training script
│   ├── predict.py          # Prediction script
│   ├── evaluate.py        # Model evaluation script
│   └── utils/
│       ├── __init__.py
│       └── diabetic_check.py  # Utility functions
│
├── scripts/
│   └── expand_dataset.py   # Dataset expansion utility
│
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── DATASET_GUIDE.md        # Dataset expansion guide
```

---

## 🏥 Clinical Context

### Why Accuracy Matters

**Insulin Dosing Rule**:
- 1 unit of insulin per 10-15g carbohydrates (individual varies)

**Impact of Errors**:
- **5g error** → 25-50 mg/dL blood glucose change
- **10g error** → 50-100 mg/dL blood glucose change
- **15g error** → 75-150 mg/dL blood glucose change

**Current System Performance**:
- Average error: **5.9g** (30-59 mg/dL impact)
- 82% of predictions within ±10g
- Better than manual counting (±20-30% errors)

### Glycemic Load (GL) Classification

- **Low GL (≤10)**: Gradual blood sugar rise ✅ Safe
- **Medium GL (11-19)**: Moderate blood sugar rise ⚠️ Caution
- **High GL (≥20)**: Rapid blood sugar rise ⚠️ High caution

**Formula**: GL = (GI × Net Carbs) / 100

### Clinical Recommendations

1. **Use as Decision Support**: Always verify with healthcare provider
2. **Individual Ratios**: Adjust insulin ratio based on personal needs
3. **Monitor Blood Sugar**: Track actual vs. predicted impact
4. **Continuous Learning**: System improves with more data

---

## ⚠️ Safety & Disclaimers

### Important Medical Disclaimer

**⚠️ THIS IS NOT A MEDICAL DEVICE**

- This system is a **research/decision support tool** only
- **NOT FDA approved** for medical use
- **NOT a substitute** for professional medical advice
- **Always consult** with healthcare providers for insulin dosing
- Individual insulin ratios vary (10-15g per unit)
- Blood sugar responses vary by person

### Limitations

1. **Limited Dataset**: Trained on 6 dishes, 1000 images
2. **Portion Size**: Assumes standard serving sizes
3. **Variability**: Different preparations may vary in carbs
4. **Accuracy**: 82% within ±10g (not 100% accurate)
5. **Clinical Validation**: Not clinically validated

### Best Practices

- ✅ Use as a **starting point** for carb estimation
- ✅ **Verify** predictions with nutrition labels when available
- ✅ **Monitor** blood sugar after meals
- ✅ **Adjust** based on personal experience
- ❌ **Don't rely solely** on predictions
- ❌ **Don't use** without healthcare provider consultation

---

## 🔮 Future Enhancements

### Short-term (v1.1 - v1.5)

- [ ] Expand dataset to 200+ images per dish
- [ ] Add more Indian dishes (10+ total)
- [ ] Portion size estimation
- [ ] Multi-dish meal support
- [ ] Real-time mobile app

### Medium-term (v2.0)

- [ ] Clinical validation study
- [ ] User feedback loop for continuous improvement
- [ ] Personalization based on user history
- [ ] Integration with CGM (Continuous Glucose Monitor)
- [ ] Meal planning recommendations

### Long-term (v3.0+)

- [ ] Multi-cuisine support (beyond Indian food)
- [ ] Real-time video analysis
- [ ] Community dataset contributions
- [ ] FDA approval pathway exploration
- [ ] Integration with insulin pumps

---

## 🤝 Contributing

We welcome contributions! Areas where help is needed:

1. **Dataset Expansion**: More images, more dishes
2. **Model Improvements**: Better architectures, hyperparameter tuning
3. **Code Quality**: Bug fixes, documentation, testing
4. **Clinical Validation**: Real-world testing and feedback
5. **Feature Development**: New features from roadmap

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Guidelines

- Follow PEP 8 style guide
- Add docstrings to new functions
- Include tests for new features
- Update documentation as needed

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset**: Indian Food Images dataset from Kaggle
- **Framework**: TensorFlow/Keras for deep learning
- **Architecture**: MobileNetV2 for efficient feature extraction
- **Community**: Open source ML community for tools and inspiration

---

## 📞 Contact & Support

- **Issues**: Report bugs or request features via [GitHub Issues](https://github.com/yourusername/Carbs-ai/issues)
- **Discussions**: Join discussions in [GitHub Discussions](https://github.com/yourusername/Carbs-ai/discussions)

---

## 📚 Additional Resources

- [DATASET_GUIDE.md](DATASET_GUIDE.md) - Comprehensive guide for dataset expansion
- [Type 1 Diabetes Foundation](https://www.jdrf.org/) - Resources for Type 1 diabetes
- [Carb Counting Guide](https://www.diabetes.org/) - Manual carb counting resources

---

**Made with ❤️ for the Type 1 diabetes community**

*Remember: This tool is for decision support only. Always consult with healthcare providers for medical decisions.*
Contributor fix test
