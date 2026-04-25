# ==============================
# 1. IMPORT LIBRARIES
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

# ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Clustering
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ==============================
# 2. LOAD DATA
# ==============================
df = pd.read_csv("Crop_recommendation.csv")

print("=== Dataset Overview ===")
print(df.head())
print("\nShape:", df.shape)
print("\nClass Distribution:")
print(df['label'].value_counts())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nBasic Statistics:")
print(df.describe())

# ==============================
# 3. VISUALIZATION (EDA)
# ==============================

# --- Figure 1: Correlation Heatmap (features only, NO label) ---
plt.figure(figsize=(9, 7))
feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
sns.heatmap(df[feature_cols].corr(numeric_only=True), annot=True,
            fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("Figure_1_Correlation_Heatmap.png", dpi=150)
plt.show()

# --- Figure 2: Feature Distributions ---
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for i, col in enumerate(feature_cols):
    axes[i].hist(df[col], bins=30, color='steelblue', edgecolor='white')
    axes[i].set_title(col)
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Frequency")
axes[-1].set_visible(False)
plt.suptitle("Feature Distributions", fontsize=14)
plt.tight_layout()
plt.savefig("Figure_2_Feature_Distributions.png", dpi=150)
plt.show()

# --- Figure 3: Class Distribution Bar Chart ---
plt.figure(figsize=(14, 5))
df['label'].value_counts().plot(kind='bar', color='coral', edgecolor='black')
plt.title("Crop Class Distribution")
plt.xlabel("Crop")
plt.ylabel("Count")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("Figure_3_Class_Distribution.png", dpi=150)
plt.show()

# ==============================
# 4. PREPROCESSING
# ==============================

# Encode target AFTER EDA plots are done
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

print("\nClass label mapping:")
for i, cls in enumerate(le.classes_):
    print(f"  {i}: {cls}")

# Features & target
X = df[feature_cols]
y = df['label_encoded']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (stratified to maintain class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing  samples: {X_test.shape[0]}")

# ==============================
# 5. MACHINE LEARNING MODELS
# ==============================

results = {}  # Store accuracy for final comparison

def evaluate_model(name, model, save_cm=True, fig_num=None):
    """Train, evaluate and optionally save confusion matrix."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print(f"\n{'='*50}")
    print(f"Model: {name}")
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(classification_report(y_test, y_pred,
                                target_names=le.classes_,
                                zero_division=0))

    if save_cm and fig_num:
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=le.classes_,
                    yticklabels=le.classes_)
        plt.title(f"{name} — Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.savefig(f"Figure_{fig_num}_{name.replace(' ', '_')}_CM.png", dpi=150)
        plt.show()

    return model, y_pred

# --- Logistic Regression ---
# FIX: max_iter increased to 1000 to ensure convergence with 22 classes
lr = LogisticRegression(max_iter=1000, random_state=42)
lr_model, _ = evaluate_model("Logistic Regression", lr, fig_num=4)

# --- Random Forest ---
# FIX: Added random_state for reproducibility
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model, _ = evaluate_model("Random Forest", rf, fig_num=5)

# --- SVM ---
# FIX: Added random_state for reproducibility
svm = SVC(random_state=42)
svm_model, _ = evaluate_model("SVM", svm, fig_num=6)

# ==============================
# 6. HYPERPARAMETER TUNING (Random Forest)
# ==============================
print("\n=== Hyperparameter Tuning (Random Forest) ===")
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth':    [5, 10, None]
}

grid = GridSearchCV(RandomForestClassifier(random_state=42),
                    param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best CV Accuracy: {:.4f}".format(grid.best_score_))

best_rf = grid.best_estimator_
tuned_rf_model, _ = evaluate_model("Tuned Random Forest", best_rf, fig_num=7)

# --- Feature Importance ---
importances = best_rf.feature_importances_
sorted_idx = np.argsort(importances)

plt.figure(figsize=(8, 5))
plt.barh(np.array(feature_cols)[sorted_idx], importances[sorted_idx],
         color='steelblue', edgecolor='white')
plt.xlabel("Importance Score")
plt.title("Feature Importance — Tuned Random Forest")
plt.tight_layout()
plt.savefig("Figure_8_Feature_Importance.png", dpi=150)
plt.show()

# ==============================
# 7. DEEP LEARNING MODEL
# ==============================
print("\n=== Deep Learning Model ===")

num_classes = len(le.classes_)

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    # FIX: use len(le.classes_) — more reliable than np.unique(y)
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_accuracy', patience=10,
                           restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# --- Training History Plot ---
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'],    label='Train Accuracy', color='steelblue')
plt.plot(history.history['val_accuracy'], label='Val Accuracy',   color='coral')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("DL Model — Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'],    label='Train Loss', color='steelblue')
plt.plot(history.history['val_loss'], label='Val Loss',   color='coral')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("DL Model — Loss")
plt.legend()

plt.tight_layout()
plt.savefig("Figure_9_DL_Training_History.png", dpi=150)
plt.show()

# --- Evaluate DL on Test Set ---
loss, dl_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nDeep Learning Test Accuracy: {dl_acc:.4f} ({dl_acc*100:.2f}%)")
print(f"Deep Learning Test Loss:     {loss:.4f}")

results['Deep Learning (MLP)'] = dl_acc

# DL Predictions & Classification Report
y_pred_dl = np.argmax(model.predict(X_test), axis=1)
print(classification_report(y_test, y_pred_dl,
                             target_names=le.classes_, zero_division=0))

# DL Confusion Matrix
cm_dl = confusion_matrix(y_test, y_pred_dl)
plt.figure(figsize=(14, 12))
sns.heatmap(cm_dl, annot=True, fmt='d', cmap='Purples',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Deep Learning (MLP) — Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig("Figure_10_DL_Confusion_Matrix.png", dpi=150)
plt.show()

# ==============================
# 8. MODEL COMPARISON TABLE
# ==============================
print("\n=== Model Comparison Summary ===")
comparison_df = pd.DataFrame(
    list(results.items()), columns=['Model', 'Test Accuracy']
).sort_values('Test Accuracy', ascending=False).reset_index(drop=True)

comparison_df['Test Accuracy (%)'] = (comparison_df['Test Accuracy'] * 100).round(2)
print(comparison_df.to_string(index=False))

# Bar chart comparison
plt.figure(figsize=(10, 5))
colors = ['gold' if acc == comparison_df['Test Accuracy'].max() else 'steelblue'
          for acc in comparison_df['Test Accuracy']]
plt.bar(comparison_df['Model'], comparison_df['Test Accuracy'] * 100,
        color=colors, edgecolor='black')
plt.ylim(80, 101)
plt.ylabel("Test Accuracy (%)")
plt.title("Model Performance Comparison")
plt.xticks(rotation=20, ha='right')
for i, v in enumerate(comparison_df['Test Accuracy']):
    plt.text(i, v * 100 + 0.3, f"{v*100:.2f}%", ha='center', fontsize=9)
plt.tight_layout()
plt.savefig("Figure_11_Model_Comparison.png", dpi=150)
plt.show()

# ==============================
# 9. CLUSTERING
# ==============================
print("\n=== Clustering Analysis ===")

# --- Elbow Method to justify K ---
# FIX: Previously K=3 was used without justification
inertias = []
sil_scores = []
K_range = range(2, 12)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(K_range, inertias, marker='o', color='steelblue')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia (WCSS)")
plt.title("Elbow Method")
plt.xticks(K_range)

plt.subplot(1, 2, 2)
plt.plot(K_range, sil_scores, marker='s', color='coral')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Analysis")
plt.xticks(K_range)

plt.tight_layout()
plt.savefig("Figure_12_Elbow_Silhouette.png", dpi=150)
plt.show()

best_k = K_range[np.argmax(sil_scores)]
print(f"\nBest K by silhouette score: {best_k} (score: {max(sil_scores):.4f})")

# --- K-Means with best K ---
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

# Visualize K-Means: N vs P (soil nutrients)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(df['N'], df['P'],
                      c=df['KMeans_Cluster'], cmap='tab10',
                      alpha=0.6, edgecolors='none', s=30)
plt.colorbar(scatter, label='Cluster')
plt.xlabel("Nitrogen (N)")
plt.ylabel("Phosphorus (P)")
plt.title(f"K-Means Clustering (K={best_k}) — Soil Nutrients: N vs P")
plt.tight_layout()
plt.savefig("Figure_13_KMeans_NP.png", dpi=150)
plt.show()

# Visualize K-Means: temperature vs humidity
plt.figure(figsize=(8, 6))
scatter = plt.scatter(df['temperature'], df['humidity'],
                      c=df['KMeans_Cluster'], cmap='tab10',
                      alpha=0.6, edgecolors='none', s=30)
plt.colorbar(scatter, label='Cluster')
plt.xlabel("Temperature (°C)")
plt.ylabel("Humidity (%)")
plt.title(f"K-Means Clustering (K={best_k}) — Temperature vs Humidity")
plt.tight_layout()
plt.savefig("Figure_14_KMeans_TempHumidity.png", dpi=150)
plt.show()

# --- Hierarchical Clustering ---
# FIX: Added visualization (was missing before)
hc = AgglomerativeClustering(n_clusters=best_k)
df['HC_Cluster'] = hc.fit_predict(X_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(df['N'], df['P'],
                      c=df['HC_Cluster'], cmap='Set2',
                      alpha=0.6, edgecolors='none', s=30)
plt.colorbar(scatter, label='HC Cluster')
plt.xlabel("Nitrogen (N)")
plt.ylabel("Phosphorus (P)")
plt.title(f"Hierarchical Clustering (K={best_k}) — Soil Nutrients: N vs P")
plt.tight_layout()
plt.savefig("Figure_15_HierarchicalClustering.png", dpi=150)
plt.show()

# --- Cluster Insights ---
print("\n=== K-Means Cluster Insights (Feature Means) ===")
cluster_insights = df[feature_cols + ['KMeans_Cluster']].groupby('KMeans_Cluster').mean().round(2)
print(cluster_insights.to_string())

# Heatmap of cluster profiles
plt.figure(figsize=(10, 5))
sns.heatmap(cluster_insights.T, annot=True, fmt=".1f", cmap='YlOrRd',
            linewidths=0.5)
plt.title("K-Means Cluster Profiles (Feature Mean Values)")
plt.xlabel("Cluster")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("Figure_16_Cluster_Profiles.png", dpi=150)
plt.show()

# Silhouette comparison: K-Means vs Hierarchical
sil_km = silhouette_score(X_scaled, df['KMeans_Cluster'])
sil_hc = silhouette_score(X_scaled, df['HC_Cluster'])
print(f"\nSilhouette Score — K-Means:              {sil_km:.4f}")
print(f"Silhouette Score — Hierarchical Clustering: {sil_hc:.4f}")

# ==============================
# 10. FINAL SUMMARY
# ==============================
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print("\n--- ML & DL Model Accuracy ---")
print(comparison_df[['Model', 'Test Accuracy (%)']].to_string(index=False))
best_model = comparison_df.iloc[0]
print(f"\nBest Model: {best_model['Model']} ({best_model['Test Accuracy (%)']:.2f}%)")

print(f"\n--- Clustering ---")
print(f"Optimal Clusters (K): {best_k}")
print(f"K-Means Silhouette Score:              {sil_km:.4f}")
print(f"Hierarchical Clustering Silhouette Score: {sil_hc:.4f}")