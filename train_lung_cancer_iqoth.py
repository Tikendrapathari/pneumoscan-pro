"""
Lung Cancer Detection - FINAL HIGH ACCURACY VERSION (3 CLASS)
Classes: Benign, Malignant, Normal
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
import warnings
warnings.filterwarnings('ignore')

# ==============================================
# CONFIG
# ==============================================

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 40
CLASS_NAMES = ['Benign', 'Malignant', 'Normal']

DATA_DIR = Path(r"E:/dataset2/lung_cancer/The IQ-OTHNCCD lung cancer dataset/The IQ-OTHNCCD lung cancer dataset")

# ==============================================
# CLAHE PREPROCESSING
# ==============================================

def enhance_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)

    img = cv2.merge((l,a,b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

    return img

# ==============================================
# LOAD DATASET (3 CLASS)
# ==============================================

def load_dataset():
    X = []
    y = []

    classes = ["Bengin cases", "Malignant cases", "Normal cases"]

    for label, class_name in enumerate(classes):
        folder = DATA_DIR / class_name

        for img_path in folder.glob("*"):
            try:
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                img = enhance_image(img)

                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img.astype(np.float32) / 255.0

                X.append(img)
                y.append(label)

            except:
                pass

    X = np.array(X)
    y = np.array(y)

    print(f"Total Images: {len(X)}")
    print(f"Benign: {np.sum(y==0)}, Malignant: {np.sum(y==1)}, Normal: {np.sum(y==2)}")

    return X, y

# ==============================================
# MODEL
# ==============================================

def create_model():
    base_model = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    # Freeze initial layers
    for layer in base_model.layers[:200]:
        layer.trainable = False

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.6)(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(3, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    return model, base_model

# ==============================================
# TRAIN FUNCTION
# ==============================================

def train():
    X, y = load_dataset()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Class weights
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(enumerate(class_weights))

    print("Class Weights:", class_weights)

    # Augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    model, base_model = create_model()

    # Phase 1
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        class_weight=class_weights,
        verbose=1
    )

    # Phase 2 - Fine tuning
    for layer in base_model.layers:
        layer.trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=15,
        class_weight=class_weights,
        verbose=1
    )

    # Evaluation
    y_pred = np.argmax(model.predict(X_val), axis=1)

    acc = accuracy_score(y_val, y_pred)
    print(f"\n FINAL ACCURACY: {acc*100:.2f}%")

    print("\nClassification Report:\n")
    print(classification_report(y_val, y_pred, target_names=CLASS_NAMES))

    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES)
    plt.show()

    # Save model
    model.save("lung_cancer_model_3class.keras")
    print(" Model Saved!")

    return model

# ==============================================

if __name__ == "__main__":
    train()