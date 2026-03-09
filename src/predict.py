import pickle
from preprocess.preprocess import clean_text

model = pickle.load(open("Models/mbti_model.pkl", "rb"))
vectorizer = pickle.load(open("Models/vectorizer.pkl", "rb"))

text = input("Enter text: ")
cleaned = clean_text(text)

vectorized = vectorizer.transform([cleaned])

prediction = model.predict(vectorized)

print("Predicted Personality Type:", prediction[0])
