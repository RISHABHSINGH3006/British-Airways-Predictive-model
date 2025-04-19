# ✈️ Customer Booking Prediction using Random Forest

This project builds a machine learning model to predict whether a customer will complete a booking or not based on travel-related features. The model is trained using a Random Forest classifier after extensive data preparation and feature engineering.

---

## 📁 Dataset

The dataset contains **50,000** customer records from flight bookings, including:
- Passenger details
- Booking preferences (baggage, meals, seat)
- Route and origin
- Flight timing details
- Booking completion status (target variable)

---

## 🧰 Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
```

---

## 📊 Exploratory Data Analysis (EDA)

- Used `.head()`, `.info()`, and `.describe()` to inspect structure and statistics.
- No missing values detected.
- Categorical columns like `flight_day`, `sales_channel`, and `trip_type` were encoded numerically.

**Example Transformation:**

```python
# Encode days of the week
day_mapping = {"Mon": 1, "Tue": 2, "Wed": 3, "Thu": 4, "Fri": 5, "Sat": 6, "Sun": 7}
df["flight_day"] = df["flight_day"].map(day_mapping)
```

---

## 🧪 Feature Engineering

We engineered a new feature to combine optional service preferences:

```python
df['total_services_opted'] = df['wants_extra_baggage'] + df['wants_preferred_seat'] + df['wants_in_flight_meals']
```

---

## 📈 Visualizations

- **Booking Completion Pie Chart**:
  ```python
  df['booking_complete'].value_counts().plot.pie(autopct='%1.1f%%')
  ```

- **Correlation Heatmap**:
  ```python
  sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
  ```

- **Feature vs. Target Correlation**:
  ```python
  corr_with_target = df.drop(columns=['booking_complete', 'route','booking_origin']).corrwith(df['booking_complete'])
  sns.heatmap(pd.DataFrame(corr_with_target))
  ```

---

## 🤖 Machine Learning Model

We used **RandomForestClassifier** to model `booking_complete`.

### Preprocessing:

```python
# Target and features
X = df.drop('booking_complete', axis=1)
y = df['booking_complete']

# One-hot encode categorical columns
X = pd.get_dummies(X, columns=['route', 'booking_origin'], drop_first=True)
```

### Model Training:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
```

### Accuracy:

```python
accuracy = accuracy_score(y_test, rf_classifier.predict(X_test))
print(f"Accuracy: {accuracy}")
# Output: ~85.3%
```

---

## 🧪 Model Evaluation

- **Cross-Validation** (5-fold):
  ```python
  kf = KFold(n_splits=5, shuffle=True, random_state=42)
  cv_scores = cross_val_score(rf_classifier, X, y, cv=kf, scoring='accuracy')
  ```

- **Confusion Matrix and Classification Report**:
  ```python
  print(confusion_matrix(y_test, y_pred))
  print(classification_report(y_test, y_pred))
  ```

---

## 📌 Feature Importance Plot

```python
importances = rf_classifier.feature_importances_
feature_names = X.columns

# Create DataFrame for visualisation
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(25, 16))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance in Random Forest Model')
plt.gca().invert_yaxis()
plt.show()
```

---

## 📊 Key Insights

- Features like `purchase_lead`, `flight_hour`, and `length_of_stay` contribute most to predictions.
- Optional services (baggage, meals, preferred seat) also positively correlate with booking completion.
- Complex interactions likely exist — no single dominant factor.

---

## ✅ Final Notes

- Model shows high performance in identifying booking completions.
- Could be improved using class balancing, ensemble techniques, or deep learning.
- Great base for customer behavior analysis and personalization engines.

---

## 📂 File Structure

```
.
├── customer_booking.csv
├── model_training.py
├── feature_importance_plot.png
└── README.md
```

---

**Made with ❤️ for predictive analytics!**
```
