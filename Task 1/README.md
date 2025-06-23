# Task 1: Data Cleaning & Preprocessing – Titanic Dataset

## 🎯 Objective
The goal of this task is to clean and preprocess the Titanic dataset in preparation for machine learning. This includes handling missing values, encoding categorical features, scaling numerical values, and removing outliers.

---

## 📁 Files Included
- `titanic_dataprocessing.py` – Python script with all preprocessing steps
- `Titanic-Dataset.csv` – Dataset from [Kaggle Titanic Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset)

---

## 🛠️ Tools & Libraries Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

---

## 📌 Preprocessing Steps
1. **Loaded the dataset**
2. **Handled missing values** using median/mode
3. **Dropped irrelevant columns** (`Name`, `Ticket`, `PassengerId`)
4. **Encoded categorical variables** (`Sex`, `Embarked`) using label and one-hot encoding
5. **Standardized** numeric features: `Age`, `Fare`
6. **Visualized outliers** using boxplots
7. **Removed outliers** using the IQR method
8. Displayed the cleaned DataFrame info and stats

---

## ▶️ How to Run
Make sure `Titanic-Dataset.csv` is in the same directory, then run:
```bash
python titanic_dataprocessing.py