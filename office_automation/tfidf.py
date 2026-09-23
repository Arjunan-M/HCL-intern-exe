import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
nltk.download('movie_reviews', quiet=True)
from nltk.corpus import movie_reviews
documents = [
    {"text": movie_reviews.raw(fileid), "sentiment": category}
    for category in movie_reviews.categories()
    for fileid in movie_reviews.fileids(category)
]
df = pd.DataFrame(documents)
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["sentiment"], test_size=0.20, random_state=42, stratify=df["sentiment"]
)
model_pipeline = make_pipeline(
    TfidfVectorizer(lowercase=True, stop_words="english", max_features=10000, ngram_range=(1, 2)),
    MultinomialNB()
)
model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)
print("=== Model Performance ===")
print(classification_report(y_test, y_pred))
sample_reviews = [
    "This movie was absolutely fantastic and entertaining.",
    "The movie was boring, slow and disappointing.",
    "I really enjoyed the story and the excellent acting.",
    "The film was terrible and a complete waste of time."
]
print("=== Custom Predictions ===")
predictions = model_pipeline.predict(sample_reviews)
for review, pred in zip(sample_reviews, predictions):
    print(f"Review: '{review}' -> Predicted: {pred}")
