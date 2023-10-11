import pandas as pd

data = pd.read_csv("Hannes-Movie-Reviews-proccessed-nefnir.csv")

data["sentiment"] = data["sentiment"].apply(
    lambda x: "positive" if x > 4 else "negative"
)

data.to_csv("Hannes-Movie-Reviews-proccessed-nefnir-sentiment.csv", index=False)
