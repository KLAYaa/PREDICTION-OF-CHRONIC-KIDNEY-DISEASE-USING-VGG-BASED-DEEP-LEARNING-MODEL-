"""
Chronic Kidney Disease (CKD) Classification using VGG16
Based on: "Prediction of Chronic Kidney Disease using VGG-Based Deep Learning Model"

Classes: Normal, Cyst, Tumor, Stone
Dataset split: 80% train, 10% val, 10% test
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ──────────────────────────────────────────────
# 1. CONFIGURATION
# ──────────────────────────────────────────────

IMG_SIZE    = (224, 224)       # VGG16 standard input size
BATCH_SIZE  = 32
EPOCHS      = 40
NUM_CLASSES = 4
LR          = 1e-4
CLASS_NAMES = ['Cyst', 'Normal', 'Stone', 'Tumor']

# Update these paths to point to your dataset folder.
# Expected structure:
#   dataset/
#     train/   Cyst/  Normal/  Stone/  Tumor/
#     val/     Cyst/  Normal/  Stone/  Tumor/
#     test/    Cyst/  Normal/  Stone/  Tumor/
DATASET_DIR = 'dataset'
TRAIN_DIR   = os.path.join(DATASET_DIR, 'train')
VAL_DIR     = os.path.join(DATASET_DIR, 'val')
TEST_DIR    = os.path.join(DATASET_DIR, 'test')

MODEL_SAVE_PATH = 'ckd_vgg16_best.keras'


# ──────────────────────────────────────────────
# 2. DATA AUGMENTATION & PREPROCESSING
# ──────────────────────────────────────────────

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASS_NAMES,
    shuffle=True
)

val_gen = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASS_NAMES,
    shuffle=False
)

test_gen = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASS_NAMES,
    shuffle=False
)

print(f"\nClass indices: {train_gen.class_indices}")
print(f"Training samples  : {train_gen.samples}")
print(f"Validation samples: {val_gen.samples}")
print(f"Test samples      : {test_gen.samples}\n")


# ──────────────────────────────────────────────
# 3. MODEL — VGG16 WITH TRANSFER LEARNING
# ──────────────────────────────────────────────

def build_vgg16_model(num_classes: int, learning_rate: float) -> tf.keras.Model:
    """
    VGG16 pre-trained on ImageNet; top layers replaced for CKD classification.
    Base layers frozen initially (fine-tuning can be enabled after warm-up).
    """
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )

    # Freeze base during warm-up phase
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    return model, base_model


model, base_model = build_vgg16_model(NUM_CLASSES, LR)


# ──────────────────────────────────────────────
# 4. CALLBACKS
# ──────────────────────────────────────────────

callbacks = [
    ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
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
    )
]


# ──────────────────────────────────────────────
# 5. PHASE 1 — WARM-UP (frozen base, train head)
# ──────────────────────────────────────────────

print("\n── Phase 1: Warm-up (base frozen) ──")
history_warmup = model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen,
    callbacks=callbacks
)


# ──────────────────────────────────────────────
# 6. PHASE 2 — FINE-TUNING (unfreeze last block)
# ──────────────────────────────────────────────

print("\n── Phase 2: Fine-tuning (last conv block unfrozen) ──")

# Unfreeze only the last convolutional block (block5)
base_model.trainable = True
for layer in base_model.layers:
    if layer.name.startswith('block5'):
        layer.trainable = True
    else:
        layer.trainable = False

# Recompile with a lower learning rate
model.compile(
    optimizer=optimizers.Adam(learning_rate=LR / 10),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_finetune = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks
)


# ──────────────────────────────────────────────
# 7. EVALUATION ON TEST SET
# ──────────────────────────────────────────────

print("\n── Test Set Evaluation ──")
best_model = tf.keras.models.load_model(MODEL_SAVE_PATH)
test_loss, test_acc = best_model.evaluate(test_gen, verbose=1)
print(f"\nTest Accuracy : {test_acc * 100:.2f}%")
print(f"Test Loss     : {test_loss:.4f}")

# Per-class metrics
test_gen.reset()
y_pred_probs = best_model.predict(test_gen, verbose=1)
y_pred  = np.argmax(y_pred_probs, axis=1)
y_true  = test_gen.classes

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))


# ──────────────────────────────────────────────
# 8. PLOTS
# ──────────────────────────────────────────────

def merge_histories(h1, h2):
    merged = {}
    for key in h1.history:
        merged[key] = h1.history[key] + h2.history[key]
    return merged

history = merge_histories(history_warmup, history_finetune)


def plot_accuracy_loss(history: dict, save_path: str = 'training_curves.png'):
    epochs_range = range(1, len(history['accuracy']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(epochs_range, history['accuracy'],     label='Train Accuracy',     linewidth=2)
    axes[0].plot(epochs_range, history['val_accuracy'], label='Validation Accuracy', linewidth=2, linestyle='--')
    axes[0].set_title('VGG16 Accuracy Curves', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(epochs_range, history['loss'],     label='Train Loss',     linewidth=2)
    axes[1].plot(epochs_range, history['val_loss'], label='Validation Loss', linewidth=2, linestyle='--')
    axes[1].set_title('VGG16 Loss Curves', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved training curves → {save_path}")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title('Confusion Matrix — VGG16 CKD Classification', fontsize=13)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved confusion matrix → {save_path}")
    plt.show()


plot_accuracy_loss(history)
plot_confusion_matrix(y_true, y_pred, CLASS_NAMES)


# ──────────────────────────────────────────────
# 9. SINGLE-IMAGE INFERENCE (utility)
# ──────────────────────────────────────────────

def predict_image(image_path: str, model: tf.keras.Model) -> None:
    """
    Run inference on a single kidney image and print the result.
    Usage: predict_image('path/to/kidney.jpg', best_model)
    """
    from tensorflow.keras.preprocessing.image import load_img, img_to_array

    img   = load_img(image_path, target_size=IMG_SIZE)
    arr   = img_to_array(img) / 255.0
    arr   = np.expand_dims(arr, axis=0)

    probs = model.predict(arr)[0]
    idx   = np.argmax(probs)

    print(f"\nImage : {image_path}")
    print(f"Prediction : {CLASS_NAMES[idx]}  ({probs[idx] * 100:.1f}% confidence)")
    for i, cls in enumerate(CLASS_NAMES):
        bar = '█' * int(probs[i] * 30)
        print(f"  {cls:<8} {bar} {probs[i] * 100:.1f}%")


# Example (uncomment to use):
# predict_image('sample_kidney.jpg', best_model)
