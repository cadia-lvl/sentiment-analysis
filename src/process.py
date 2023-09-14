
import nltk
from .definitions import *
from reynir import Greynir
from tokenizer import tokenize, TOK
import tokenizer

g = Greynir()



class TextNormalizer:

    # TODO: add stop_words

    def tokenize(txt: str, lower_case = True):
        if lower_case:
            txt = txt.lower()
        tokens = tokenizer.tokenize(txt.lower())
        return [ token.txt for token in tokens if token.kind == TOK.WORD ]

    def lemmatize(tokens):
        output = []
        for filtered_token in tokens:
            parsed_token = g.parse_single(filtered_token)
            if parsed_token.tree is None or parsed_token.lemmas is None:
                output.append(filtered_token)
            else:
                output.append(parsed_token.lemmas.pop())
        return ' '.join(output)

