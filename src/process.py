import html
import multiprocessing
import os
import string
import sys
import time
from tkinter import Tk, filedialog
import pandas as pd
import tokenizer
import re
from joblib import Parallel, delayed
import subprocess

# from nefnir import Nefnir
import nefnir
from pathlib import Path

stop_flag = 0


class TextNormalizer:
    def __init__(self, icetagger=None):
        self.stop_words = None
        current_dir = os.path.dirname(os.path.abspath(__file__))
        stop_words_file_path = os.path.join(current_dir, "all_stop_words.txt")

        with open(stop_words_file_path) as f:
            self.stop_words = f.readlines()
            self.stop_words = [
                stop_word.replace("\n", "") for stop_word in self.stop_words
            ]

        self.icetagger = icetagger

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

        return " ".join(
            [token for token in negated_txt if token[0] not in string.punctuation]
        )

    def tokenize(self, txt, lower_case=True):
        if lower_case:
            txt = txt.lower()
        return [token.txt for token in tokenizer.tokenize(txt)]

    def send_word_to_script(self, word):
        try:
            result = subprocess.run(
                [
                    "cmd",
                    "/c",
                    self.icetagger,
                ],
                input=word,
                text=True,
                check=True,
                stdout=subprocess.PIPE,
                cwd=self.icetagger[
                    :-13
                ],  # remove icetagger.bat from path to get the directory
            )
            java_output = result.stdout.strip().split("\n")[-1]

            tokens = java_output.split(" ")
            tokens = [(tokens[i], tokens[i + 1]) for i in range(0, len(tokens), 2)]

            return tokens

        except subprocess.CalledProcessError as e:
            print(f"An error occurred: {str(e)}")

    def lemmatize(self, tokens, index):
        if self.icetagger is None:
            return tokens

        output = []
        tokens = self.send_word_to_script(" ".join(tokens))
        for token, tag in tokens:
            output.append(nefnir.lemmatize(token, tag))
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
        txt = html.unescape(txt)
        txt = self.clean_html(txt)
        txt = self.remove_brackets(txt)
        txt = self.lower_case(txt)
        txt = self.remove_special_characters(txt)
        txt = self.fix_repeated_characters(txt)
        txt = self.remove_overly_long_words(txt)
        return txt

    def process(self, k, txt, stop_flag=0):
        if stop_flag:
            return
        try:
            print(f"Processing {k} by thread {multiprocessing.current_process().name}")
            txt = self.remove_noise(txt)
            return self.mark_negation(
                self.lemmatize(self.tokenize(self.remove_stop_words(txt)), k)
            )
        except Exception as e:
            print(f"Failed to process {k} with error {e}")
            stop_flag = 1
            raise e


if __name__ == "__main__":
    root = Tk()
    root.withdraw()

    print("Select the icetagger.bat File")
    icetagger = filedialog.askopenfilename(
        title="Select the icetagger.bat File", filetypes=[("bat files", "*.bat")]
    )
    if not icetagger or not Path(icetagger).exists() or not Path(icetagger).is_file():
        print("Invalid icetagger.bat path")
        sys.exit()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    dataset_path = os.path.join(
        parent_dir, "Datasets/IMDB-Dataset-MideindTranslate.csv"
    )

    data = pd.read_csv(dataset_path)
    review = data["review"]
    tn = TextNormalizer(icetagger)
    start = time.time()
    results = Parallel(n_jobs=1, verbose=10)(
        delayed(tn.process)(k, row, stop_flag) for k, row in enumerate(review)
    )

    data["review"] = results
    data.to_csv("Datasets/IMDB-Dataset-MideindTranslate-processed-nefnir.csv")
    end = time.time()
    print(f"Processed in {end-start} seconds.")
