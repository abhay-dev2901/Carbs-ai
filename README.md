# Indian Food Carb Estimator – Milestone 2

## Dataset

- **5 Indian dishes**: biryani, dal, halwa, poha, roti
- **250 images total**
- **80% training** (200 images)
- **20% validation** (50 images)

## Model Architecture

- **Classifier**: MobileNetV2 + Global Average Pooling + Dense
- **Regressor**: Features from classifier → Dense(64) → 1 output

## Performance

| Model           | Metric                  | Result    |
| --------------- | ----------------------- | --------- |
| Dish Classifier | **Validation Accuracy** | **94.0%** |
| Carb Regressor  | **Validation MAE**      | **16.3g** |

> Model correctly identifies dish in **94%** of cases  
> Carb prediction error: **±16.3g** on average

## Run

```bash
python src/train_classifier.py
python src/train_regressor.py
python src/predict.py data/images/biryani/0d81432b55.jpg
```
