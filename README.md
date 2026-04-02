# Credit Card Default Prediction

Predicting whether a customer will default on their credit card payment using Logistic Regression and Random Forest.

Built this as a beginner ML project to get a proper feel for the full workflow rather than just copying code off the internet.

---

## Dataset

**UCI Credit Card Default Dataset** with 30,000 customer records and 24 features including credit limit, payment history, bill amounts, and demographic info.

Source: [Kaggle](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)

Target variable: `default.payment.next.month` (1 = defaulted, 0 = did not default)

---

## Workflow

1. Load and explore the data - shape, missing values, column types
2. EDA - class balance, basic statistics
3. Preprocessing - drop ID column, train/test split, feature scaling
4. Train models - Logistic Regression and Random Forest
5. Evaluate and compare - accuracy, F1-score, ROC-AUC

---

## Models Used

**Logistic Regression**
- Scaled features using `StandardScaler`
- `max_iter=1000` to ensure convergence

**Random Forest**
- 100 estimators
- No scaling needed

---

## Results

| Model | Accuracy | F1 Score (Default) | ROC-AUC |
|---|---|---|---|
| Logistic Regression | ~80% | -- | -- |
| Random Forest | ~80% | -- | -- |

Accuracy alone is not the best metric here since the dataset is imbalanced (around 78% non-default, 22% default). F1-score and ROC-AUC tell a more complete story.

---

## Libraries

- `pandas`, `numpy` - data handling
- `matplotlib`, `seaborn` - visualization
- `scikit-learn` - models and evaluation

---

## What I Learned

- Why scaling matters for Logistic Regression but not Random Forest
- Why accuracy can be misleading on imbalanced datasets
- The difference between precision, recall, F1-score, and ROC-AUC
- How to properly split data to avoid data leakage (fit scaler only on train set)
