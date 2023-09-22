from reynir import Greynir
from nltk.tokenize import word_tokenize
import re

g = Greynir()


class TextNormalizer:
    def __init__(self):
        self.stop_words = None
        with open("./all_stop_words.txt") as f:
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
                if re.match(r"[.?!;,]", token):
                    negation_scope = False
                else:
                    token = token + "_NEG"
            negated_txt.append(token)
        txt = re.sub(r" ([?.!,“;])", r"\1", " ".join(negated_txt))
        txt = re.sub(r"([„])([\s])", r"\1", txt)
        return re.sub("([?.!,;“„])", r"", txt)

    def tokenize(self, txt, lower_case=True):
        if lower_case:
            txt = txt.lower()
        return word_tokenize(txt)

    def lemmatize(self, tokens):
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
                output.append("".join(parsed_token.lemmas))
        return output

    def remove_stop_words(self, txt):
        return " ".join([t for t in txt.split(" ") if t not in self.stop_words])

    def clean_html(self, txt):
        clean = re.compile("<.*?>")
        return re.sub(clean, "", txt)

    def remove_brackets(self, txt):
        return re.sub("\[[^]]*\]", "", txt)

    def lower_case(self, txt):
        return txt.lower()

    def remove_special_characters(self, txt):
        pattern = r"[^a-zA-z0-9\s]"
        txt = re.sub(pattern, "", txt)
        return txt

    def remove_noise(self, txt):
        txt = self.clean_html(txt)
        txt = self.remove_brackets(txt)
        txt = self.lower_case(txt)
        txt = self.remove_special_characters(txt)
        return txt

    def process(self, txt):
        return self.mark_negation(
            self.lemmatize(self.tokenize(self.remove_stop_words(txt)))
        )


if __name__ == "__main__":
    txt = "Ég er ekki sáttur með þetta. Þetta er ekki gott."
    tn = TextNormalizer()
    print(tn.process(txt))
