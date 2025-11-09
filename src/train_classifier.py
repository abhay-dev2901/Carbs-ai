# src/train_classifier.py
"""
Enhanced Classifier Training for Type 1 Diabetes Carb Estimation
- Improved data augmentation
- Better regularization
- Early stopping and learning rate scheduling
"""
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os

DATA_DIR = "./data/images"
IMG_SIZE = (224, 224)
BATCH = 32
EPOCHS = 30  

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

train_gen = datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH,
    class_mode='categorical', subset='training', shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH,
    class_mode='categorical', subset='validation', shuffle=False
)

base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
base.trainable = False  # Freeze base initially

model = Sequential([
    base,
    GlobalAveragePooling2D(name='gap_layer'),
    BatchNormalization(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(train_gen.num_classes, activation='softmax')
])

model.build((None, 224, 224, 3))

# Compile with better optimizer settings
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy', 'top_k_categorical_accuracy']
)

# Callbacks for better training
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        './models/classifier_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

print(f"Training classifier on {train_gen.samples} samples")
print(f"Validating on {val_gen.samples} samples")
print(f"Classes: {train_gen.class_indices}")

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks,
    verbose=1
)

# Final evaluation
val_loss, val_accuracy, val_top_k = model.evaluate(val_gen, verbose=0)
print(f"\n{'='*60}")
print(f"Final Validation Accuracy: {val_accuracy:.2%}")
print(f"Top-K Accuracy: {val_top_k:.2%}")
print(f"{'='*60}")

os.makedirs("./models", exist_ok=True)
model.save("./models/classifier.h5")
print("Classifier saved!")