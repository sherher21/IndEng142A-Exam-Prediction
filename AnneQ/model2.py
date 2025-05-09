import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv("personalized_learning_dataset.csv")

q_low = df['Final_Exam_Score'].quantile(0.01)
q_high = df['Final_Exam_Score'].quantile(0.99)
df = df[(df['Final_Exam_Score'] > q_low) & (df['Final_Exam_Score'] < q_high)]

bins = list(df['Final_Exam_Score'].quantile([0, 0.33, 0.66, 1.0]))
labels = [0, 1, 2]
df['Score_Class'] = pd.cut(df['Final_Exam_Score'], bins=bins, labels=labels, include_lowest=True)

df['Video_Quiz_Interaction'] = df['Time_Spent_on_Videos'] * df['Quiz_Scores']
df['Quiz_Efficiency'] = df['Quiz_Scores'] / df['Quiz_Attempts'].replace(0, 1)
df['Participation_Efficiency'] = df['Forum_Participation'] / df['Assignment_Completion_Rate'].replace(0, 1)


numerical_cols = [
    'Time_Spent_on_Videos',
    'Assignment_Completion_Rate',
    'Quiz_Efficiency',
    'Forum_Participation',
    'Quiz_Scores',
    'Age',
    'Feedback_Score',
    'Quiz_Attempts'
]
X = df[numerical_cols]

def score_to_label(score):
    if score < 60:
        return 0
    elif score < 75:
        return 1
    else:
        return 2

y_class = df['Final_Exam_Score'].apply(score_to_label)

X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [300, 500, 700],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced']
}


grid = GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=42), param_grid,
                    cv=3, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)
print("Best Parameters:", grid.best_params_)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["0-59", "60-74", "75-100"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix: Final Exam Score Classes")
plt.show()

