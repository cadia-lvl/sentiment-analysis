import nltk
from reynir import Greynir
from tokenizer import tokenize, TOK
import tokenizer

g = Greynir()


class TextNormalizer:
    def tokenize(self, txt, lower_case=True):
        if lower_case:
            txt = txt.lower()
        tokens = tokenizer.tokenize(txt.lower())
        return [token.txt for token in tokens if token.kind == TOK.WORD]

    def lemmatize(self, tokens):
        output = []
        for filtered_token in tokens:
            parsed_token = g.parse_single(filtered_token)
            if parsed_token.tree is None or parsed_token.lemmas is None:
                output.append(filtered_token)
            else:
                output.append(parsed_token.lemmas.pop())
        return " ".join(output)

    def remove_stop_words(self, txt):
        stop_words = None
        with open("all_stop_words.txt") as f:
            stop_words = f.readlines()
            stop_words = [stop_word.replace("\n", "") for stop_word in stop_words]
        return " ".join([t for t in txt.split(" ") if t not in stop_words])

    def process(self, txt):
        return self.lemmatize(self.tokenize(self.remove_stop_words(txt)))
