# Task-6
# K-Nearest Neighbors (KNN) Classifier on Iris Dataset

This project demonstrates how to implement the **K-Nearest Neighbors (KNN)** algorithm from scratch using **Python and Scikit-learn**. We used the classic **Iris flower dataset**, which is ideal for learning classification algorithms.

---

## 🎯 Objective
To learn how KNN works by:
- Normalizing data
- Training a KNN model
- Tuning the value of K
- Evaluating accuracy
- Visualizing decision boundaries

---

## 🧰 Tools & Libraries Used
- Python
- Pandas
- NumPy
- Scikit-learn (sklearn)
- Matplotlib

---

## 📦 Dataset
The dataset used is `Iris.csv`, which contains:
- **150 records**
- **4 numerical features**: SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm
- **1 categorical label**: Species

---

## ✅ Steps Followed

### 🔹 Step 1: Import Libraries
We imported libraries for data manipulation (`pandas`, `numpy`), visualization (`matplotlib`), and machine learning (`sklearn`).

---

### 🔹 Step 2: Load Dataset
The dataset was loaded using `pandas.read_csv()` and the first few rows were printed to understand the structure.

---

### 🔹 Step 3: Drop Unnecessary Columns
The `Id` column was not useful for prediction, so it was removed to avoid confusion during training.

---

### 🔹 Step 4: Split Features and Labels
- `X`: contains feature values (all columns except `Species`)
- `y`: contains the label (`Species`)

---

### 🔹 Step 5: Normalize the Features
KNN is a distance-based algorithm, so it's important to normalize features using `StandardScaler`.

---

### 🔹 Step 6: Split Dataset into Training and Testing
Used `train_test_split()` from sklearn to split the data into 80% training and 20% testing.

---

### 🔹 Step 7: Train KNN for Different K Values
A loop was used to train the model for `K=1` to `K=10`. Accuracy for each K was printed to find the best-performing value.

---

### 🔹 Step 8: Evaluate Final Model
After identifying the best K (e.g., K=3), we retrained the model, predicted on test data, and printed:
- Accuracy
- Confusion Matrix

---

### 🔹 Step 9: Visualize Decision Boundaries
We selected only **2 features** to visualize how KNN separates the classes.
- Used `LabelEncoder` to convert species names into numbers.
- Plotted decision regions using colored backgrounds.
- Plotted actual data points with different colors.

---

## 📈 Accuracy Achieved
Achieved accuracy ~96% using `K=3` on test data.

---

   
