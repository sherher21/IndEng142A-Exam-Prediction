import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("personalized_learning_dataset.csv")

df['Quiz_Efficiency'] = df['Quiz_Scores'] / df['Quiz_Attempts'].replace(0, 1)

numerical_cols = [
    'Age', 'Time_Spent_on_Videos', 'Quiz_Attempts',
    'Quiz_Scores', 'Forum_Participation',
    'Assignment_Completion_Rate', 'Feedback_Score',
    'Quiz_Efficiency' 
]

categorical_cols = [
    'Gender', 'Education_Level', 'Course_Name',
    'Learning_Style', 'Dropout_Likelihood'
]

df_encoded = pd.get_dummies(df[categorical_cols], drop_first=True)

X = pd.concat([df[numerical_cols], df_encoded], axis=1)
Y = df['Final_Exam_Score']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2', None]  
}

grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid,
                    cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Best Parameters:", grid.best_params_)
print("Tuned RMSE:", rmse)
print("Tuned R² Score:", r2)

importances = pd.Series(best_model.feature_importances_, index=X.columns)
print(importances.sort_values(ascending=False))

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Final Exam Scores")
plt.ylabel("Predicted Scores")
plt.title("Actual vs Predicted Scores")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.grid(True)
plt.tight_layout()
plt.show()
