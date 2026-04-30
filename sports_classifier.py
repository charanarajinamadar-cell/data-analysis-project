import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Dataset
data = pd.DataFrame({
    "SPORTS": [
        "Cricket", "Football", "Basketball", "Swimming",
        "Tennis", "Badminton", "Hockey", "Boxing",
        "Rugby", "Golf", "Volleyball", "Table Tennis"
    ],
    "SPORT LOCATION": [
        "OUTDOOR", "OUTDOOR", "INDOOR", "INDOOR",
        "OUTDOOR", "INDOOR", "OUTDOOR", "INDOOR",
        "OUTDOOR", "OUTDOOR", "INDOOR", "INDOOR"
    ]
})

# Features & labels
X = data["SPORTS"]
y = data["SPORT LOCATION"]

# Convert text → numbers
vectorizer = CountVectorizer()
X_vector = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_vector, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", round(accuracy, 2))

# Predictions
test_sports = ["Cricket", "Chess", "Swimming"]

for sport in test_sports:
    test_vector = vectorizer.transform([sport])
    prediction = model.predict(test_vector)
    print(f"{sport} → {prediction[0]}")
