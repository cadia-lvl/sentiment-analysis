import multiprocessing
import string
import time
import dask.dataframe as dd
from dask.distributed import Client
import pandas as pd
from reynir import Greynir
from nltk.tokenize import word_tokenize
import re
from joblib import Parallel, delayed
import logging


logging.basicConfig(level=logging.INFO)
stop_flag = 0


class TextNormalizer:
    def __init__(self):
        self.stop_words = None
        if __name__ == "__main__":
            file_path = "src/all_stop_words.txt"
        else:
            file_path = "all_stop_words.txt"
        with open(file_path) as f:
            self.stop_words = f.readlines()
            self.stop_words = [
                stop_word.replace("\n", "") for stop_word in self.stop_words
            ]

    def mark_negation(self, tokens):
        negation_words = [
            "ekki",
            "aldrei",
            "engin",
            "hvorugt",
            "hvorki",
            "enginn",
        ]
        negation_scope = False
        negated_txt = []
        for token in tokens:
            if token in negation_words:
                negation_scope = True
                negated_txt.append(token)
                continue

            if negation_scope:
                if re.match(r"[.?!;:,]", token):
                    negation_scope = False
                else:
                    token = token + "_NEG"
            negated_txt.append(token)
        # txt = re.sub(r" ([?.!,“;])", r"\1", " ".join(negated_txt))
        # txt = re.sub(r"([„])([\s])", r"\1", txt)
        # return re.sub("([?.!,;“„])", r"", txt)
        return " ".join(
            [token for token in negated_txt if token not in string.punctuation]
        )

    def tokenize(self, txt, lower_case=True):
        if lower_case:
            txt = txt.lower()
        return word_tokenize(txt)

    def lemmatize(self, tokens, index):
        g = Greynir()
        output = []
        for filtered_token in tokens:
            parsed_token = g.parse_single(filtered_token)
            if (
                not parsed_token
                or parsed_token.tree is None
                or parsed_token.lemmas is None
            ):
                output.append(filtered_token)
            else:
                for lemmas in parsed_token.lemmas:
                    output.append(lemmas)

        return output

    def remove_stop_words(self, txt):
        return " ".join([t for t in txt.split(" ") if t not in self.stop_words])

    def clean_html(self, txt):
        clean = re.compile("<.*?>")
        return re.sub(clean, "", txt)

    def remove_brackets(self, txt):
        return re.sub(r"[()\[\]{}<>]", "", txt)

    def lower_case(self, txt):
        return txt.lower()

    def fix_repeated_characters(self, txt):
        return re.sub(r"(.)\1{5,}", r"\1", txt)

    def remove_overly_long_words(self, txt):
        return " ".join([t for t in txt.split(" ") if len(t) < 30])

    def remove_special_characters(self, txt):
        # remove special characters and digits except icelandic characters

        pattern = r"[^a-zA-záðéíóúýþæö.?!;:,\s]"
        txt = re.sub(pattern, "", txt)
        return txt

    def remove_noise(self, txt):
        txt = self.clean_html(txt)
        txt = self.remove_brackets(txt)
        txt = self.lower_case(txt)
        txt = self.remove_special_characters(txt)
        txt = self.fix_repeated_characters(txt)
        txt = self.remove_overly_long_words(txt)
        return txt

    def process(self, k, txt, stop_flag):
        if stop_flag:
            return
        try:
            print(f"Processing {k} by thread {multiprocessing.current_process().name}")
            txt = self.remove_noise(txt)
            return self.mark_negation(
                self.lemmatize(self.tokenize(self.remove_stop_words(txt)), k)
            )
        except Exception as e:
            print(f"Failed to lemmatize {k} with error {e}")
            stop_flag = 1
            raise e


if __name__ == "__main__":
    # client = Client()
    # data = dd.read_csv("IMDB-Dataset copy.csv")
    # tn = TextNormalizer()
    # start = time.time()
    # data["review"] = data["review"].map_partitions(
    #     lambda partition: partition.apply(tn.process), meta=("review", "object")
    # )
    # data.compute().to_csv("IMDB-Dataset-MideindTranslate-Processed.csv", index=False)
    # client.close()
    # end = time.time()
    # print(f"Processed in {end-start} seconds.")

    data = pd.read_csv("IMDB-Dataset-GoogleTranslate.csv")
    review, sentiment = data["review"], data["sentiment"]
    tn = TextNormalizer()
    start = time.time()
    results = Parallel(n_jobs=14, verbose=10)(
        delayed(tn.process)(k, row, stop_flag) for k, row in enumerate(review)
    )

    data["review"] = results
    data.to_csv("IMDB-Dataset-GoogleTranslate-Processed3.csv")
    end = time.time()
    print(f"Processed in {end-start} seconds.")

    # tn = TextNormalizer()
    # print(
    #     tn.process(
    #         0,
    #         "Þetta er ekki góður texti aaaaaabbbbcdefghijklmnopqr.",
    #     )
    # )
