import time
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.util import mark_negation
import re
from joblib import Parallel, delayed


class ENTextNormalizer:
    def __init__(self, stop_words):
        nltk.download("punkt")
        nltk.download("stopwords")
        nltk.download("wordnet")
        self.stop_words = set(stop_words)
        self.lemmatizer = WordNetLemmatizer()

    def mark_negation(self, tokens):
        return mark_negation(tokens, double_neg_flip=True, shallow=True)

    def tokenize(self, txt):
        return word_tokenize(txt)

    def lemmatize(self, tokens):
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def remove_stop_words(self, tokens):
        return [token for token in tokens if token.lower() not in self.stop_words]

    def remove_noise(self, txt):
        txt = re.sub(r"<.*?>", "", txt)  # Remove HTML tags
        txt = txt.lower()  # Convert to lowercase
        txt = re.sub(r"([?.!,;:])([\w])", r"\1 \2", txt)
        txt = re.sub(r"[()\[\]{}<>]", "", txt)
        txt = re.sub(r"[^a-zA-z,.!?:;\s]", "", txt)
        return txt

    def remove_punctuation(self, tokens):
        return [
            token
            for token in tokens
            if token.isalnum()
            or (token.endswith("_NEG") and not token.startswith(","))
            or token in self.stop_words
        ]

    def process(self, index, txt):
        print(f"Processing {index}")
        txt = self.remove_noise(txt)
        tokens = self.tokenize(txt)
        tokens = self.remove_stop_words(tokens)
        tokens = self.lemmatize(tokens)
        tokens = self.mark_negation(tokens)
        tokens = self.remove_punctuation(tokens)
        return " ".join(tokens)


if __name__ == "__main__":
    data = pd.read_csv("IMDB-Dataset.csv")
    review, sentiment = data["review"], data["sentiment"]
    tn = ENTextNormalizer(stopwords.words("english"))
    start = time.time()
    results = Parallel(n_jobs=14, verbose=10)(
        delayed(tn.process)(k, row) for k, row in enumerate(review)
    )

    data["review"] = results
    data.to_csv("IMDB-Dataset-Processed.csv")
    end = time.time()
    print(f"Processed in {end-start} seconds.")
