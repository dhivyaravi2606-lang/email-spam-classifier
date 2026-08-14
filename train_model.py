import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


# 1. Load the dataset
df = pd.read_csv("spam.csv")

# Remove empty rows
df = df.dropna()

# 2. Get email messages and labels
messages = df["message"].astype(str)
labels = df["label"].astype(str).str.lower().str.strip()


# 3. Convert email text into numerical features
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(messages)


# 4. Create the Machine Learning model
model = MultinomialNB()

# 5. Train the model
model.fit(X, labels)


# 6. Save the trained model
with open("spam_model.pkl", "wb") as file:
    pickle.dump(model, file)


# 7. Save the text vectorizer
with open("vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)


print("====================================")
print("Email Spam Classifier Model Trained!")
print("====================================")
print("spam_model.pkl created successfully.")
print("vectorizer.pkl created successfully.")
print("====================================")