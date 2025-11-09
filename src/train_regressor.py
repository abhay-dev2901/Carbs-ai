# src/train_regressor.py
"""
Enhanced Regressor Training for Net Carbohydrate Estimation
- Fixed label mapping bug
- Improved architecture with regularization
- Better training pipeline with callbacks
- Comprehensive evaluation metrics
"""
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

DATA_DIR = "./data/images"
CSV_PATH = "./data/nutrition.csv"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50

# === 1. REBUILD CLASSIFIER USING FUNCTIONAL API ===
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
base.trainable = False

inputs = Input(shape=(*IMG_SIZE, 3))
x = base(inputs, training=False)
x = GlobalAveragePooling2D(name='gap_layer')(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
outputs = Dense(5, activation='softmax')(x)  # 5 classes

clf = Model(inputs, outputs)

try:
    clf.load_weights("./models/classifier.h5")
    print("Loaded classifier weights successfully")
except Exception as e:
    print(f"Warning: Could not load classifier weights: {e}")
    print("Training regressor from scratch...")


gap_layer = clf.get_layer('gap_layer')
feature_model = Model(inputs=clf.input, outputs=gap_layer.output)
feature_model.trainable = False  # Freeze feature extractor initially


datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.9, 1.1]
)

train_gen = datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    subset='training', shuffle=True, class_mode='categorical'
)

val_gen = datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    subset='validation', shuffle=False, class_mode='categorical'
)

labels_df = pd.read_csv(CSV_PATH)
class_to_carbs = dict(zip(labels_df['dish'], labels_df['net_carbs_g']))

class_indices = train_gen.class_indices
index_to_carbs = {idx: class_to_carbs[class_name] for class_name, idx in class_indices.items()}
print(f"Class to carbs mapping: {class_to_carbs}")
print(f"Index to carbs mapping: {index_to_carbs}")


def carb_generator(gen):
    """Properly map each image to its class's carb value"""
    while True:
        batch_x, batch_y = next(gen)
        # batch_y is one-hot encoded, convert to class indices
        class_indices_batch = np.argmax(batch_y, axis=1)
        # Map each class index to its carb value
        carb_values = np.array([index_to_carbs[int(idx)] for idx in class_indices_batch])
        yield batch_x, carb_values.astype(np.float32)

train_carb_gen = carb_generator(train_gen)
val_carb_gen = carb_generator(val_gen)


inputs = Input(shape=(*IMG_SIZE, 3))
feats = feature_model(inputs, training=False)
x = BatchNormalization()(feats)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.2)(x)
out = Dense(1, activation='linear')(x) 
reg = Model(inputs, out)


reg.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='huber', 
    metrics=['mae', 'mse']
)


callbacks = [
    EarlyStopping(
        monitor='val_mae',
        patience=12,
        restore_best_weights=True,
        verbose=1,
        mode='min'
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=6,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        './models/regressor_best.h5',
        monitor='val_mae',
        save_best_only=True,
        verbose=1,
        mode='min'
    )
]

print(f"\n{'='*60}")
print(f"Training regressor on {train_gen.samples} samples")
print(f"Validating on {val_gen.samples} samples")
print(f"{'='*60}\n")


history = reg.fit(
    train_carb_gen,
    steps_per_epoch=max(1, train_gen.samples // BATCH_SIZE),
    validation_data=val_carb_gen,
    validation_steps=max(1, val_gen.samples // BATCH_SIZE),
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)


print(f"\n{'='*60}")
print("EVALUATION METRICS")
print(f"{'='*60}")

val_predictions = []
val_true = []
steps = 0
val_carb_gen_eval = carb_generator(val_gen)
for _ in range(val_gen.samples // BATCH_SIZE + 1):
    x_batch, y_batch = next(val_carb_gen_eval)
    pred_batch = reg.predict(x_batch, verbose=0)
    val_predictions.extend(pred_batch.flatten())
    val_true.extend(y_batch)
    steps += 1
    if steps * BATCH_SIZE >= val_gen.samples:
        break

val_predictions = np.array(val_predictions[:val_gen.samples])
val_true = np.array(val_true[:val_gen.samples])


mae = mean_absolute_error(val_true, val_predictions)
rmse = np.sqrt(mean_squared_error(val_true, val_predictions))
r2 = r2_score(val_true, val_predictions)
mape = np.mean(np.abs((val_true - val_predictions) / val_true)) * 100


within_10g = np.sum(np.abs(val_predictions - val_true) <= 10) / len(val_true) * 100
within_5g = np.sum(np.abs(val_predictions - val_true) <= 5) / len(val_true) * 100

print(f"Mean Absolute Error (MAE): {mae:.2f}g")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}g")
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"\nAccuracy Metrics:")
print(f"  Within ±5g: {within_5g:.1f}%")
print(f"  Within ±10g: {within_10g:.1f}% (Target: >80%)")
print(f"\nError Distribution:")
errors = np.abs(val_predictions - val_true)
print(f"  Min error: {np.min(errors):.2f}g")
print(f"  Max error: {np.max(errors):.2f}g")
print(f"  Median error: {np.median(errors):.2f}g")
print(f"  75th percentile: {np.percentile(errors, 75):.2f}g")
print(f"  95th percentile: {np.percentile(errors, 95):.2f}g")
print(f"{'='*60}\n")

os.makedirs("./models", exist_ok=True)
reg.save("./models/regressor.h5")
print("Regressor saved!")