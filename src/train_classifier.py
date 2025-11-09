# src/train_classifier.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential
import os

DATA_DIR = "./data/images"
IMG_SIZE = (224, 224)
BATCH = 32
EPOCHS = 5

# Data
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH,
    class_mode='categorical', subset='training'
)

val_gen = datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH,
    class_mode='categorical', subset='validation'
)

# Model
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
base.trainable = False

model = Sequential([
    base,
    GlobalAveragePooling2D(name='gap_layer'),  # NAMED
    Dense(128, activation='relu'),
    Dense(train_gen.num_classes, activation='softmax')
])

# CRITICAL: Build model with input shape
model.build((None, 224, 224, 3))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen)

# After model.fit(...)
val_loss, val_accuracy = model.evaluate(val_gen)
print(f"Final Validation Accuracy: {val_accuracy:.2%}")

os.makedirs("./models", exist_ok=True)
model.save("./models/classifier.h5")
print("Classifier saved with build() and named GAP layer!")