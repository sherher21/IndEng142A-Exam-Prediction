import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA

df = pd.read_csv("personalized_learning_dataset.csv")

df['Quiz_Efficiency'] = df['Quiz_Scores'] / df['Quiz_Attempts'].replace(0, 1)
df['Video_Quiz_Ratio'] = df['Time_Spent_on_Videos'] / df['Quiz_Attempts'].replace(0, 1)
df['Active_Participation'] = df['Forum_Participation'] * df['Assignment_Completion_Rate']

numerical_cols = [
    'Age', 'Time_Spent_on_Videos', 'Quiz_Attempts', 'Quiz_Scores',
    'Forum_Participation', 'Assignment_Completion_Rate', 'Feedback_Score',
    'Quiz_Efficiency', 'Video_Quiz_Ratio', 'Active_Participation'
]
categorical_cols = [
    'Gender', 'Education_Level', 'Learning_Style', 'Dropout_Likelihood'
]

df_encoded = pd.get_dummies(df[categorical_cols], drop_first=True)

X = pd.concat([df[numerical_cols], df_encoded], axis=1)
y = df['Final_Exam_Score']

X_numeric_only = X.select_dtypes(include=[np.number])
Q1 = X_numeric_only.quantile(0.25)
Q3 = X_numeric_only.quantile(0.75)
IQR = Q3 - Q1
mask = ~((X_numeric_only < (Q1 - 1.5 * IQR)) | (X_numeric_only > (Q3 + 1.5 * IQR))).any(axis=1)
X = X[mask]
y = y[mask]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

param_grid = {
    'n_neighbors': list(range(3, 21, 2)),
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}
grid_search = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5, scoring='r2')
grid_search.fit(X_train_pca, y_train)

best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test_pca)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Best Parameters:", grid_search.best_params_)
print(f"Optimized RMSE (with PCA & enhanced features): {rmse:.2f}")
print(f"Optimized R² Score (with PCA & enhanced features): {r2:.2f}")
