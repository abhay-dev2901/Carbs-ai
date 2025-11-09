# src/train_regressor.py
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.models import Model

DATA_DIR = "./data/images"
CSV_PATH = "./data/nutrition.csv"
IMG_SIZE = (224, 224)

# === 1. REBUILD CLASSIFIER USING FUNCTIONAL API ===
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
base.trainable = False

inputs = Input(shape=(*IMG_SIZE, 3))
x = base(inputs, training=False)
x = GlobalAveragePooling2D(name='gap_layer')(x)
x = Dense(128, activation='relu')(x)
outputs = Dense(5, activation='softmax')(x)  # 5 classes

clf = Model(inputs, outputs)

# === 2. Load trained weights ===
clf.load_weights("./models/classifier.h5")  # Only weights, not full model

# === 3. Extract feature model ===
gap_layer = clf.get_layer('gap_layer')
feature_model = Model(inputs=clf.input, outputs=gap_layer.output)

# === 4. Data generators ===
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, validation_split=0.2
)
train_gen = datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=32, subset='training', shuffle=False
)
val_gen = datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=32, subset='validation', shuffle=False
)

# === 5. Map carbs ===
labels_df = pd.read_csv(CSV_PATH)
class_to_carbs = dict(zip(labels_df['dish'], labels_df['net_carbs_g']))
ordered_carbs = [class_to_carbs[c] for c in train_gen.class_indices.keys()]

# === 6. Generator with carb labels ===
def carb_generator(gen):
    i = 0
    while True:
        x, _ = next(gen)
        y = np.array([ordered_carbs[i % len(ordered_carbs)] for i in range(i, i + len(x))])
        i += len(x)
        yield x, y

train_carb_gen = carb_generator(train_gen)
val_carb_gen = carb_generator(val_gen)

# === 7. Regressor ===
inputs = Input(shape=(224, 224, 3))
feats = feature_model(inputs)
x = Dense(64, activation='relu')(feats)
out = Dense(1)(x)
reg = Model(inputs, out)

reg.compile(optimizer='adam', loss='mse', metrics=['mae'])

# === 8. Train ===
reg.fit(
    train_carb_gen,
    steps_per_epoch=max(1, train_gen.samples // 32),
    validation_data=val_carb_gen,
    validation_steps=max(1, val_gen.samples // 32),
    epochs=8
)

# After reg.fit(...)
val_mae = reg.evaluate(val_carb_gen, steps=val_gen.samples // 32 + 1, verbose=0)[1]
print(f"Final Carb MAE: {val_mae:.1f}g")

reg.save("./models/regressor.h5")
print("Regressor saved!")