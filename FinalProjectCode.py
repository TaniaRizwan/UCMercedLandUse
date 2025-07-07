# ENEL 525 Final Project
# Author: Tania Rizwan (30115533)
# Date: December 18, 2024
# The objective of this file is to classify images from the UCMerced_LandUse Dataset

# Sources used:
# https://www.tensorflow.org/api_docs/python/tf/keras/applications/InceptionResNetV2
# https://www.tensorflow.org/tutorials/images/data_augmentation
# https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.confusion_matrix.html
# https://www.geeksforgeeks.org/os-module-python-examples/

# Imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, GlobalAveragePooling2D, RandomFlip, RandomRotation, RandomZoom, RandomContrast, RandomTranslation
)
from tensorflow.keras.utils import to_categorical # For one-hot encoding
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping # Metrics for callbacks
from tensorflow.keras.applications import InceptionResNetV2 # Transfer learning 
from tensorflow.keras.optimizers import Adam

import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import random
import matplotlib.pyplot as plt
import seaborn as sns

# For reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)


# Import data
data_dir = "UCMerced_LandUse/Images"
classes = sorted(os.listdir(data_dir))

X = [] # Features
Y = [] # Labels 
for index, class_name in enumerate(classes):
    class_path = os.path.join(data_dir, class_name) # Example: UCMerced_LandUse/Images/agricultural
    images = os.listdir(class_path) # Load all images
    for image_name in images:
        image_path = os.path.join(class_path, image_name)
        with Image.open(image_path) as img:
            img = img.resize((256, 256)).convert('RGB') # for generalizability and consistency
            image_arr = np.array(img) / 255.0 # Normalize vals to [0, 1]
            if image_arr.shape == (256, 256, 3):
                X.append(image_arr)
                Y.append(index)

# Convert X and Y to numpy arrays
X = np.array(X) # Tensor 
Y = np.array(Y)


# Train, Validation and Test Split: 70%, 15%, 15%, respectively
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y) # Temp will be split into val and test
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42, stratify=Y_temp) # 15% of overall data for each

# Convert labels to one-hot
Y_train_onehot = to_categorical(Y_train, num_classes=len(classes))
Y_val_onehot = to_categorical(Y_val, num_classes=len(classes))
Y_test_onehot = to_categorical(Y_test, num_classes=len(classes))

print(f"Training data: {X_train.shape}, Labels: {Y_train.shape}")
print(f"Validation data: {X_val.shape}, Labels: {Y_val.shape}")
print(f"Testing data: {X_test.shape}, Labels: {Y_test.shape}")

# Data Augmentation to Improve Model
data_augmentation = Sequential([
    RandomFlip('horizontal'),
    RandomRotation(0.2),
    RandomZoom(0.2),
    RandomContrast(0.1),
    RandomTranslation(0.1, 0.1),
])

# Base model
# InceptionResNetV2 used as it performs well on complex image classification tasks
base_model = InceptionResNetV2(
    weights='imagenet', # Load with pretrained imagenet weights
    include_top=False, # Only load the convolution (feature extraction) layers. Not the dense layers (those are for the model-specific output, we want to customize them)
    input_shape=(256, 256, 3)
)
base_model.trainable = False  # Freeze layers so they don't upate during training. Want to keep the integrity of the model, not update it for our dataset.

# Model definition
model = Sequential([
    data_augmentation, # Add augmentation and base model
    base_model,
    GlobalAveragePooling2D(), # Better than flattening. Reduces dimensions from convolutional layers before input into dense layers
    Dense(512, activation='relu'), 
    Dropout(0.5), # Randomly drop half of neurons during training to reduce dependance on specifics
    Dense(21, activation='softmax')
])

# Compile model
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy', # For one-hot encoded classification
    metrics=['accuracy', tf.keras.metrics.F1Score(threshold=0.5), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()] # Commonly used ML metrics. Need for generating the classification report
)

# Callbacks: Early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Callbacks: Reduce LR based on performance 
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    verbose=1,
    min_lr=1e-6
)

# Training 
history = model.fit(
    X_train, Y_train_onehot,
    validation_data=(X_val, Y_val_onehot),
    batch_size=16, # Number of samples trained before updating weights
    epochs=20, # Iterations over the dataset
    callbacks=[reduce_lr, early_stopping],
    verbose=2 # Show progress for each epoch
)

# Print model summary
model.summary()

# Evaluate on test set
results = model.evaluate(X_test, Y_test_onehot, verbose=1)

# Unpack all returned values
test_loss, test_accuracy, test_f1, test_precision, test_recall = results

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
print(f"Test F1 Score: {test_f1}")
print(f"Test Precision: {test_precision}")
print(f"Test Recall: {test_recall}")

# Save the model
model.save("final_model.h5")
print("Model saved as final_model.h5")

# Model Performance 
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

print(f"Model Accuracy: {acc}")
print(f"Model Accuracy (Val): {val_acc}")
print(f"Model Loss: {loss}")
print(f"Model Loss (Val): {val_loss}")

# Plot training & validation accuracy values
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Train Accuracy')
plt.plot(val_acc, label='Val Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()

plt.savefig('training_validation_plot.png', dpi=300)
plt.show()
plt.clf()

# Confusion Matrix
Y_pred = model.predict(X_test)
Y_pred_classes = np.argmax(Y_pred, axis=1)
cm = confusion_matrix(Y_test, Y_pred_classes)
print("Confusion Matrix:\n", cm)

# Plot confusion matrix with labels
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# Save the confusion matrix plot
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()
plt.clf()

# Classification report
report = classification_report(Y_test, Y_pred_classes, target_names=classes)
print("Classification Report:\n", report)