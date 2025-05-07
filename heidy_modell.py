import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

file_path = '/Users/heidybarrera/Downloads/indeng_proj/heidy_model/csv/personalized_learning_dataset.csv'
data = pd.read_csv(file_path)

data['Time_per_Quiz'] = data['Time_Spent_on_Videos'] / (data['Quiz_Attempts'] + 1)

features = [
    'Quiz_Scores',
    'Assignment_Completion_Rate',
    'Time_per_Quiz',
    'Education_Level',
    'Gender',
    'Feedback_Score'
]

categorical = ['Education_Level', 'Gender']
target = 'Final_Exam_Score'

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical)
], remainder='passthrough')

X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=10,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        random_state=42
    )
}

results = {}
for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    score_range = y_test.max() - y_test.min()
    accuracy = max(0, 1 - rmse/score_range) * 100

    results[name] = {
        'RMSE': round(rmse, 2),
        'R²': round(r2, 4),
        'Accuracy (%)': round(accuracy, 2)
    }

print("="*50)
for model, metrics in results.items():
    print("{:<15} {:<10} {:<10} {:<15}".format(
        model,
        metrics['RMSE'],
        metrics['R²'],
        metrics['Accuracy (%)']
    ))
