import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src import clean_text

# load dataset
data = pd.read_csv("dataset/mbti_dataset.csv")

# clean text
data["clean_posts"] = data["posts"].apply(clean_text)

X = data["clean_posts"]
y = data["type"]

# convert text to numbers
vectorizer = TfidfVectorizer(max_features=5000)

X_vectorized = vectorizer.fit_transform(X)

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

pickle.dump(model, open("Models/mbti_model.pkl", "wb"))
pickle.dump(vectorizer, open("Models/vectorizer.pkl", "wb"))

print("Model trained and saved successfully!")
