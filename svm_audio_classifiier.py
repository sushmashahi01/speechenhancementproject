

# ============================================
# 📚 STEP 1 — Import dependencies
# ============================================
import os
import librosa
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tqdm import tqdm
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import noisereduce as nr

def clean_audio(y, sr):
    """Perform basic audio cleaning."""
    # 1️⃣ Trim silence from start and end
    y, _ = librosa.effects.trim(y, top_db=20)

    # 2️⃣ Normalize volume
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    # 3️⃣ Optional: Light noise reduction
    if len(y) > 10000:  # avoid crashing on tiny clips
        y = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.8, stationary=True)

    # 4️⃣ Optional: High-pass filter to remove hum (<100 Hz)
    y = librosa.effects.preemphasis(y, coef=0.97)

    return y

def extract_features(file_path, n_mfcc=20):
    """Load, clean, and extract MFCC features from a WAV file."""
    try:
        y, sr = librosa.load(file_path, sr=None)
        y = clean_audio(y, sr)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"⚠️ Error processing {file_path}: {e}")
        return None

DATA_PATH = "./dataset"

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
                feat = extract_features(path)
                if feat is not None:
                    features.append(feat)
                    labels.append(label)
    return np.array(features), np.array(labels)

X_train, y_train = load_data(train_dirs)
X_test, y_test = load_data(test_dirs)
print(f"✅ Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# ============================================
# 🧮 STEP 5 — Train the SVM classifier
# ============================================
svm_clf = SVC(kernel='rbf', probability=True, C=10, gamma='scale', random_state=42)
svm_clf.fit(X_train, y_train)

y_pred = svm_clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"✅ Test Accuracy: {acc*100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ============================================
# 📊 STEP 6 — Confusion matrix visualization
# ============================================
cm = confusion_matrix(y_test, y_pred, labels=["clean", "noisy"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["clean", "noisy"], yticklabels=["clean", "noisy"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

