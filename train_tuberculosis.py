"""
Tuberculosis Detection - FIXED VERSION
Using DenseNet121 + Strong Class Weights
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ==============================================
# CONFIGURATION
# ==============================================

IMG_SIZE = 224
BATCH_SIZE = 32  # Smaller batch for imbalanced data
INITIAL_EPOCHS = 20
FINE_TUNE_EPOCHS = 15
NUM_CLASSES = 2
CLASS_NAMES = ['Normal', 'Tuberculosis']

# ✅ CORRECT DATASET PATH
DATA_DIR = r"E:/dataset2/Tuberculosis/TB_Chest_Radiography_Database"


# ==============================================
# DATA GENERATOR WITH AUGMENTATION
# ==============================================

def create_data_generators(data_dir, img_size, batch_size):
    """Create data generators with strong augmentation for TB class"""
    
    print(f"\n[INFO] Looking for dataset in: {data_dir}")
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"[ERROR] Dataset directory not found: {data_dir}")
    
    # Show folder counts
    print("\n[INFO] Available folders:")
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            image_count = len([f for f in os.listdir(item_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
            print(f"   - {item}: {image_count} images")
    
    # ✅ Strong augmentation for TB class
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        validation_split=0.2
    )
    
    print("\n[INFO] Creating data generators...")
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        classes=CLASS_NAMES,
        subset='training',
        shuffle=True,
        seed=42
    )
    
    # Validation generator
    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        classes=CLASS_NAMES,
        subset='validation',
        shuffle=False,
        seed=42
    )
    
    print(f"\n[INFO] Dataset Summary:")
    print(f"   Train samples: {train_generator.samples}")
    print(f"   Validation samples: {validation_generator.samples}")
    
    return train_generator, validation_generator


# ==============================================
# MODEL CREATION - DenseNet121
# ==============================================

def create_model(img_size):
    """Create DenseNet121 model for TB detection"""
    
    base_model = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(img_size, img_size, 3)
    )
    
    base_model.trainable = False
    
    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs, name='TB_DenseNet121')
    
    return model, base_model


# ==============================================
# MAIN TRAINING FUNCTION
# ==============================================

def train_tuberculosis_model(data_dir=None, epochs=INITIAL_EPOCHS, 
                             batch_size=BATCH_SIZE, fine_tune_epochs=FINE_TUNE_EPOCHS):
    
    print("\n" + "="*70)
    print("TUBERCULOSIS DETECTION - FIXED DENSENET121 TRAINING")
    print("="*70)
    
    if data_dir is None:
        data_dir = DATA_DIR
    
    print(f"\n[INFO] Data directory: {data_dir}")
    
    # Create directories
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    reports_dir = Path('reports')
    reports_dir.mkdir(exist_ok=True)
    
    # Create data generators
    print("\n[INFO] Creating data generators...")
    train_gen, val_gen = create_data_generators(data_dir, IMG_SIZE, batch_size)
    
    if train_gen.samples == 0:
        print("\n[ERROR] No images found!")
        return None, None
    
    # ✅ STRONG CLASS WEIGHTS for TB class
    y_train = train_gen.classes
    unique_classes = np.unique(y_train)
    
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=unique_classes,
        y=y_train
    )
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    print(f"\n[INFO] Class weights (higher for TB): {class_weight_dict}")
    
    # Create model
    print("\n[INFO] Creating DenseNet121 model...")
    model, base_model = create_model(IMG_SIZE)
    model.summary()
    
    # Callbacks
    callbacks = [
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(models_dir / 'tuberculosis_model_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Phase 1
    print("\n[INFO] Phase 1: Training top layers...")
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.AUC(name='auc')]
    )
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    # Phase 2
    print("\n[INFO] Phase 2: Fine-tuning...")
    
    base_model.trainable = True
    
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall', 'auc']
    )
    
    fine_tune_history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=fine_tune_epochs,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    # Combine histories
    for key in fine_tune_history.history:
        if key in history.history:
            history.history[key].extend(fine_tune_history.history[key])
    
    # Final evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION - TUBERCULOSIS DETECTION")
    print("="*70)
    
    y_pred_probs = model.predict(val_gen)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    y_true = val_gen.classes
    
    from sklearn.metrics import confusion_matrix, classification_report
    
    val_loss, val_acc, val_prec, val_rec, val_auc = model.evaluate(val_gen, verbose=0)
    
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n OVERALL ACCURACY: {val_acc*100:.2f}%")
    
    print(f"\n CONFUSION MATRIX:")
    print(f"   {'':<15} {'Normal':<12} {'TB':<10}")
    print(f"   {'Normal':<15} {cm[0,0]:<12} {cm[0,1]:<10}")
    print(f"   {'TB':<15} {cm[1,0]:<12} {cm[1,1]:<10}")
    
    print(f"\n CLASSIFICATION REPORT:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    
    # Save model
    final_model_path = models_dir / 'tuberculosis_model.h5'
    model.save(str(final_model_path))
    print(f"\n[INFO] Model saved to: {final_model_path}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"   Best Model: {models_dir / 'tuberculosis_model_best.h5'}")
    print(f"   Final Model: {final_model_path}")
    print(f"   Overall Accuracy: {val_acc*100:.2f}%")
    print("="*70)
    
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=DATA_DIR)
    parser.add_argument('--epochs', type=int, default=INITIAL_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--fine_tune_epochs', type=int, default=FINE_TUNE_EPOCHS)
    
    args = parser.parse_args()
    
    train_tuberculosis_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        fine_tune_epochs=args.fine_tune_epochs
    )