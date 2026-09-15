import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_and_preprocess_data():
    print("Loading Titanic dataset...")
    df = sns.load_dataset('titanic')

    features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
    target = 'survived'

    data = df[features + [target]].copy()

    data['age'] = data['age'].fillna(data['age'].median())
    data['embarked'] = data['embarked'].fillna(data['embarked'].mode()[0])

    data['sex'] = data['sex'].map({'male': 0, 'female': 1})
    data = pd.get_dummies(data, columns=['embarked'], drop_first=True)

    X = data.drop(columns=[target])
    y = data[target]

    return train_test_split(X, y, test_size=0.2, random_state=42)

def main():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    print("Training Decision Tree Classifier...")
    clf = DecisionTreeClassifier(random_state=42, max_depth=5)
    clf.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()