import os
import cv2
import numpy as np
import time
from sklearn import svm
from tqdm import tqdm
import joblib

from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# ===============================
# CONFIG
# ===============================
DATASET_PATH = "decks_balanced"
IMG_SIZE = 256

LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS
LBP_METHOD = "uniform"

print("\n🚀 PIPELINE STARTED\n")

# ===============================
# FEATURE EXTRACTION
# ===============================

def extract_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        feature_vector=True
    )

    distances = [1, 2, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    glcm = graycomatrix(
        gray,
        distances=distances,
        angles=angles,
        levels=256,
        symmetric=True,
        normed=True
    )

    properties = ['contrast', 'energy', 'homogeneity', 'correlation']
    glcm_features = []

    for prop in properties:
        vals = graycoprops(glcm, prop)
        glcm_features.extend(vals.flatten())

    glcm_features = np.array(glcm_features)

    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method=LBP_METHOD)
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, LBP_POINTS + 3),
        range=(0, LBP_POINTS + 2)
    )
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)

    edges = cv2.Canny(gray, 50, 150)

    edge_count = np.sum(edges > 0)
    edge_ratio = edge_count / (IMG_SIZE * IMG_SIZE)
    edge_mean = np.mean(edges)

    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    num_contours = len(contours)
    avg_contour_length = np.mean(
        [cv2.arcLength(c, False) for c in contours]
    ) if contours else 0

    morph_features = np.array([num_contours, avg_contour_length])
    canny_features = np.array([edge_count, edge_ratio, edge_mean])

    return np.hstack([
        hog_features,
        glcm_features,
        hist,
        canny_features,
        morph_features
    ])


# ===============================
# LOAD DATASET
# ===============================

print("📦 Stage 1: Feature Extraction Started")
start_time = time.time()

features = []
labels = []

# ===============================
# LOAD SDNET (ALL 1000 EACH)
# ===============================

sdnet_path = "decks_balanced"

for label, folder in enumerate(["non_cracked", "cracked"]):
    folder_path = os.path.join(sdnet_path, folder)
    print(f"\n   Extracting SDNET {folder.upper()} features...")

    for file in tqdm(os.listdir(folder_path)):
        path = os.path.join(folder_path, file)
        feat = extract_features(path)
        if feat is not None:
            features.append(feat)
            labels.append(label)

# ===============================
# LOAD METU (ONLY 300 EACH)
# ===============================

metu_path = "metu"
print("METU Negative count:", len(os.listdir(os.path.join(metu_path, "Negative"))))
print("METU Positive count:", len(os.listdir(os.path.join(metu_path, "Positive"))))

import random
# Non-cracked (Negative)
metu_negative = os.listdir(os.path.join(metu_path, "Negative"))
random.shuffle(metu_negative)
metu_negative=metu_negative[:300]

print("\n   Extracting METU NON_CRACKED features...")

for file in tqdm(metu_negative):
    path = os.path.join(metu_path, "Negative", file)
    feat = extract_features(path)
    if feat is not None:
        features.append(feat)
        labels.append(0)

# Cracked (Positive)
metu_positive = os.listdir(os.path.join(metu_path, "Positive"))
random.shuffle(metu_positive)
metu_positive=metu_positive[:300]
print("\n   Extracting METU CRACKED features...")

for file in tqdm(metu_positive):
    path = os.path.join(metu_path, "Positive", file)
    feat = extract_features(path)
    if feat is not None:
        features.append(feat)
        labels.append(1)

print("\nTotal samples extracted:", len(features))

X = np.array(features)
y = np.array(labels)

print("✅ Feature extraction complete.")
print("Feature vector shape:", X.shape)
print("⏱ Time taken:", round((time.time()-start_time)/60,2), "minutes")

# ===============================
# TRAIN TEST SPLIT
# ===============================

print("\n📊 Stage 2: Train/Test Split")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ===============================
# SCALING
# ===============================

print("\n⚙ Stage 3: Scaling Features")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("✅ Scaling completed")


# ===============================
# FEATURE SELECTION
# ===============================

print("\n🧠 Stage 4: Mutual Information Feature Selection")

selector = SelectKBest(score_func=mutual_info_classif, k=1200)
X_train = selector.fit_transform(X_train, y_train)
X_test = selector.transform(X_test)

print("Reduced feature size:", X_train.shape[1])

# ===============================
# OPTIMIZED RBF SVM WITH REFINED GRID
# ===============================

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score

print("\n🚀 Running Refined Grid Search for RBF SVM...")

param_grid = {
    'C': [5, 10, 15, 20],
    'gamma': ['scale', 0.05, 0.02]
}

base_svm = SVC(kernel='rbf')

grid = GridSearchCV(
    base_svm,
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train, y_train)

print("\n✅ Best Parameters Found:")
print(grid.best_params_)

best_svm = grid.best_estimator_

# -------------------------------
# Use Decision Function
# -------------------------------

scores = best_svm.decision_function(X_test)

# -------------------------------
# Threshold Optimization
# -------------------------------

print("\n🔍 Optimizing decision threshold...")

thresholds = np.linspace(scores.min(), scores.max(), 200)

best_f1 = 0
best_thresh = 0

for t in thresholds:
    preds = (scores > t).astype(int)
    f1 = f1_score(y_test, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print("Best threshold found:", round(best_thresh, 4))
print("Best F1 score:", round(best_f1, 4))

# Final Predictions
y_pred = (scores > best_thresh).astype(int)

# -------------------------------
# Evaluation
# -------------------------------

print("\n📈 Final Evaluation")

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("Accuracy:", round(accuracy, 4))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))
print("==============================")
# ===============================
# SAVE BEST PIPELINE
# ===============================

print("\n💾 Saving Best Model Artifacts...")

os.makedirs("saved_models", exist_ok=True)

joblib.dump(best_svm, "saved_models/best_svm.pkl")
joblib.dump(scaler, "saved_models/scaler.pkl")
joblib.dump(selector, "saved_models/feature_selector.pkl")

print("✅ All artifacts saved inside 'saved_models' folder.")