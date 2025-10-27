# -*- coding: utf-8 -*-

# ============================================
# 📚 STEP 1 — Import dependencies
# ============================================
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def extract_mel_spectrogram(file_path, n_mels=128, max_len=128):
    """Convert an audio file into a Mel-spectrogram array."""
    try:
        y, sr = librosa.load(file_path, sr=None)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Pad or truncate to a fixed length
        if mel_db.shape[1] < max_len:
            pad_width = max_len - mel_db.shape[1]
            mel_db = np.pad(mel_db, pad_width=((0,0),(0,pad_width)), mode='constant')
        else:
            mel_db = mel_db[:, :max_len]

        return mel_db
    except Exception as e:
        print(f"⚠️ Error processing {file_path}: {e}")
        return None

DATA_PATH = "/dataset"

train_dirs = {
    "clean": os.path.join(DATA_PATH, "clean_trainset_28spk_wav"),
    "noisy": os.path.join(DATA_PATH, "noisy_trainset_28spk_wav"),
}

test_dirs = {
    "clean": os.path.join(DATA_PATH, "clean_testset_wav"),
    "noisy": os.path.join(DATA_PATH, "noisy_testset_wav"),
}

def load_data(directories):
    features, labels = [], []
    for label, folder in directories.items():
        print(f"🔍 Loading {label} data from {folder}")
        for file in tqdm(os.listdir(folder)):
            if file.endswith(".wav"):
                path = os.path.join(folder, file)
                mel = extract_mel_spectrogram(path)
                if mel is not None:
                    features.append(mel)
                    labels.append(label)
    return np.array(features), np.array(labels)

X_train, y_train = load_data(train_dirs)
X_test, y_test = load_data(test_dirs)

print(f"✅ Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# Reshape to (samples, height, width, channels)
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Encode labels (clean = 0, noisy = 1)
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

# One-hot encode
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print("✅ Data shapes:", X_train.shape, y_train.shape)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,1)),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # 2 classes: clean & noisy
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {acc*100:.2f}%")

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

# Generate predictions on the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
labels = encoder.classes_

# Display confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix — Clean vs Noisy (CNN)")
plt.show()

