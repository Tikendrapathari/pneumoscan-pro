# train_model.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.utils import class_weight

# -------------------------------
# CONFIG
# -------------------------------
DATA_DIR = 'E:/dataset2/Pneumonia/chest_xray'
IMG_SIZE = 150
BATCH_SIZE = 32
EPOCHS_INITIAL = 15
EPOCHS_FINE_TUNE = 10

# Reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# -------------------------------
# DATA GENERATORS
# -------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'train'),
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'val'),
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_gen = test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'test'),
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# -------------------------------
# CLASS WEIGHTS
# -------------------------------
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights = {0: class_weights[0], 1: class_weights[1]}

# -------------------------------
# MODEL: VGG16 + Custom Head
# -------------------------------
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Convert grayscale → RGB
def grayscale_to_rgb(x):
    return tf.repeat(x, 3, axis=-1)

model = Sequential([
    tf.keras.layers.Lambda(grayscale_to_rgb, input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Freeze base
base_model.trainable = False

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# -------------------------------
# INITIAL TRAINING
# -------------------------------
print("Starting initial training...")
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_INITIAL,
    class_weight=class_weights,
    callbacks=[reduce_lr, early_stop]
)

# -------------------------------
# FINE-TUNING (Unfreeze last 4 layers)
# -------------------------------
for layer in base_model.layers[-4:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

print("Fine-tuning last layers...")
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FINE_TUNE,
    class_weight=class_weights,
    callbacks=[reduce_lr, early_stop]
)

# -------------------------------
# SAVE MODEL
# -------------------------------
model.save('pneumonia_model.h5')
print("Model saved as 'pneumonia_model.h5'")

# -------------------------------
# FINAL EVALUATION
# -------------------------------
test_loss, test_acc, test_prec, test_rec = model.evaluate(test_gen)
print(f"\nFINAL TEST RESULTS:")
print(f"Accuracy: {test_acc:.4f}")
print(f"Precision: {test_prec:.4f}")
print(f"Recall: {test_rec:.4f}")