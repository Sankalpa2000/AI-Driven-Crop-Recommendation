# AI-Driven Crop Recommendation

A machine learning and deep learning project that recommends optimal crops based on soil and environmental conditions.

## 📊 Dataset

| Feature | Description |
|---------|-------------|
| N | Nitrogen content in soil |
| P | Phosphorus content in soil |
| K | Potassium content in soil |
| temperature | Temperature (°C) |
| humidity | Humidity (%) |
| ph | pH value of soil |
| rainfall | Rainfall (mm) |

**Target:** 22 different crop types

## 🏗️ Project Structure

```
AI-Driven-Crop-Recommendation/
├── model.py                    # Main ML/DL model code
├── Crop_recommendation.csv     # Dataset
├── Figure_1_Correlation_Heatmap.png
├── Figure_2_Feature_Distributions.png
├── Figure_3_Class_Distribution.png
├── Figure_4_Logistic_Regression_CM.png
├── Figure_5_Random_Forest_CM.png
├── Figure_6_SVM_CM.png
├── Figure_7_Tuned_Random_Forest_CM.png
├── Figure_8_Feature_Importance.png
├── Figure_9_DL_Training_History.png
├── Figure_10_DL_Confusion_Matrix.png
├── Figure_11_Model_Comparison.png
├── Figure_12_Elbow_Silhouette.png
├── Figure_13_KMeans_NP.png
├── Figure_14_KMeans_TempHumidity.png
├── Figure_15_HierarchicalClustering.png
├── Figure_16_Cluster_Profiles.png
└── README.md
```

## 🔬 Models Used

### Machine Learning
| Model | Description |
|-------|-------------|
| Logistic Regression | Baseline linear classifier |
| Random Forest | Ensemble tree-based classifier |
| SVM | Support Vector Machine with RBF kernel |
| Tuned Random Forest | Random Forest with GridSearchCV hyperparameter tuning |

### Deep Learning
- **Architecture:** 3-layer Dense network with Dropout
  - Input → Dense(128) → Dropout(0.3) → Dense(64) → Dropout(0.2) → Dense(32) → Output(22 classes)
- **Optimizer:** Adam
- **Loss:** Sparse Categorical Crossentropy
- **Early Stopping:** Monitors val_accuracy with patience=10

### Clustering (Unsupervised)
- K-Means Clustering
- Hierarchical Clustering
- Elbow Method & Silhouette Analysis for optimal clusters

## 📈 Results

| Model | Accuracy |
|-------|----------|
| Logistic Regression | ~95% |
| Random Forest | ~97% |
| SVM | ~96% |
| Tuned Random Forest | ~98% |
| Deep Learning | ~96% |

**Best Model:** Tuned Random Forest with ~98% accuracy

## 🛠️ Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

## 🚀 Usage

```bash
python model.py
```

## 📝 License

This project is for educational purposes.