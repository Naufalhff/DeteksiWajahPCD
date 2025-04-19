import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch
from sklearn.model_selection import KFold
import numpy as np
import os

# Fungsi untuk load image dan label dari directory
def load_dataset(directory):
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=(160, 160),
        label_mode='categorical',
        batch_size=None  # agar dapat dikonversi ke array
    )
    X = []
    y = []
    for img, label in dataset:
        X.append(img.numpy())
        y.append(label.numpy())
    return np.array(X), np.array(y)

# Load full training data dari augmented
X, y = load_dataset('splits/train_augmented')

# Load validation data (digunakan setelah K-Fold selesai)
val_dataset = tf.keras.utils.image_dataset_from_directory(
    'splits/val',
    image_size=(160, 160),
    label_mode='categorical'
).prefetch(buffer_size=tf.data.AUTOTUNE)

# Fungsi bangun model (HyperModel)
def build_model(hp):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(160, 160, 3))

    for layer in base_model.layers[:-10]:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(hp.Int('units', 64, 512, step=64), activation='relu')(x)
    predictions = Dense(y.shape[1], activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        optimizer=hp.Choice('optimizer', ['adam', 'sgd']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Tuner untuk cari hyperparameter terbaik
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=1,
    directory='tuner_dir',
    project_name='resnet_kfold_tuning'
)

# K-Fold Cross Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
fold_no = 1
accuracies = []
losses = []

# Training dengan K-Fold
for train_idx, val_idx in kfold.split(X):
    print(f"\nTraining Fold {fold_no}")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=5, verbose=0)

    best_hp = tuner.get_best_hyperparameters(1)[0]
    model = build_model(best_hp)

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, verbose=1)

    val_acc = history.history['val_accuracy'][-1]
    val_loss = history.history['val_loss'][-1]
    accuracies.append(val_acc)
    losses.append(val_loss)

    print(f"Fold {fold_no} - Val Accuracy: {val_acc:.4f}, Val Loss: {val_loss:.4f}")
    fold_no += 1

# Menampilkan rata-rata hasil K-Fold
print("\n=== K-Fold Hasil Akhir ===")
print(f"Rata-rata Akurasi: {np.mean(accuracies):.4f}")
print(f"Rata-rata Kerugian: {np.mean(losses):.4f}")

# Final training dengan seluruh data + validasi
best_hp = tuner.get_best_hyperparameters(1)[0]
final_model = build_model(best_hp)
final_model.fit(X, y, epochs=10, validation_data=val_dataset)
final_model.save("final_ethnicity_model.h5")
print("✅ Model berhasil disimpan sebagai final_ethnicity_model.h5")
