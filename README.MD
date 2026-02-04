# Principal Component Analysis (PCA)

## 📌 Introduction

Principal Component Analysis (PCA) is an unsupervised dimensionality reduction technique used to reduce the number of features in a dataset while preserving as much information (variance) as possible.

It is commonly used for:
- Data visualization
- Noise reduction
- Feature extraction
- Improving model performance

## ❓ Why PCA is Needed

High-dimensional data often suffers from:
- Curse of dimensionality
- Multicollinearity
- High computation cost
- Difficult visualization

PCA solves this by:
- Removing redundant features
- Compressing data
- Keeping the most important patterns

## 🧠 Core Intuition

- PCA rotates the axes, not the data
- Finds new axes called principal components
- These components capture maximum variance
- Each component is orthogonal (independent)
- PCA focuses on spread of data, not labels

## ⚙️ Preprocessing Steps

### 1️⃣ Mean Centering

`x_centered = x − μ`

**Why?**
- PCA is based on covariance
- Covariance assumes zero-mean data
- Without centering, PCA gives wrong directions
- ✔ Mandatory step

### 2️⃣ Standard Scaling

`x_scaled = (x − μ) / σ`

**Why?**
- Features may have different units
- PCA is variance-based
- Large-scale features dominate otherwise
- ✔ Strongly recommended

## 🧮 Mathematical Background (High Level)

1. Mean center the data
2. Compute covariance matrix
3. Find eigenvalues & eigenvectors
4. Sort eigenvectors by eigenvalues
5. Project data onto top components

- Eigenvectors → directions
- Eigenvalues → amount of variance

## 🧩 PCA Components

`pca.components_`

- Attribute of PCA object
- Stores principal directions (eigenvectors)
- Shape: `(n_components, n_features)`
- 📌 Available only after `fit()`

## 🧑‍💻 Implementation (scikit-learn)

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
```

## 📊 Explained Variance

`pca.explained_variance_ratio_`

**Example:** `[0.75, 0.18]`
- ➡️ First 2 components explain 93% of variance
- Use this to decide how many components to keep

## 📈 Visualization (2D PCA)

```python
import plotly.express as px

fig = px.scatter(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    color=y.astype(str)
)
fig.show()
```

- Each point = one sample
- Axes = Principal Components
- Useful for understanding class separation

## ⚠️ Important Characteristics

- PCA is unsupervised
- Sensitive to scaling
- Linear technique
- Reduces interpretability
- Assumes variance = information

## ✅ When to Use PCA

**Use when:**
- ✔ High-dimensional datasets
- ✔ Correlated features
- ✔ Visualization
- ✔ Faster model training

**Avoid when:**
- ❌ Features are already independent
- ❌ Feature interpretability is critical

## 📝 Summary

- PCA reduces dimensions
- Preserves maximum variance
- Requires mean centering
- StandardScaler removes unit bias
- PCA finds directions of maximum spread, not maximum accuracy

## 📚 References

- [scikit-learn Documentation](https://scikit-learn.org/)
- Hands-On Machine Learning – Aurélien Géron
- Pattern Recognition and Machine Learning – C. Bishop
