import csv
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt


def read_reviews(csv_path):
    with open(csv_path, mode="r", newline="", encoding="utf-8") as file:
        csv_reader = csv.DictReader(file)
        return list(csv_reader)


def assign_sentiment(reviews, negative_threshold, positive_threshold):
    reviews_with_sentiment = []
    sentiments_count = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for row in reviews:
        rating = float(row["rating"])
        if rating >= positive_threshold:
            sentiment = "Positive"
        elif rating <= negative_threshold:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        sentiments_count[sentiment] += 1
        reviews_with_sentiment.append({**row, "sentiment": sentiment})
    return reviews_with_sentiment, sentiments_count


def write_reviews_with_sentiment(csv_path, reviews_with_sentiment):
    with open(csv_path, mode="w", newline="", encoding="utf-8") as file:
        fieldnames = ["id", "review", "rating", "sentiment"]
        csv_writer = csv.DictWriter(file, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(reviews_with_sentiment)


def print_statistics(ratings):
    ratings_counter = Counter(ratings)
    print("Rating distribution:", ratings_counter)
    mean_rating = np.mean(ratings)
    std_dev = np.std(ratings)
    print(f"Mean rating: {mean_rating:.2f}")
    print(f"Standard Deviation: {std_dev:.2f}")


def plot_rating_distribution(ratings):
    plt.hist(ratings, bins=range(1, 11), edgecolor="black", alpha=0.7, density=True)
    plt.xlabel("Rating")
    plt.ylabel("Density")
    plt.title("Hannes Rating Distribution")
    plt.grid(True)
    plt.show()


def main():
    original_csv_path = "../data/Hannes-Movie-Reviews.csv"
    new_csv_path = "../data/hannes-reviews-reviews-with-sentiment.csv"
    # 4, 7 for imdb split  | 
    # 5, 6 for five split  |
    # 6, 7 for median split|
    negative_threshold = 5
    positive_threshold = 6

    reviews = read_reviews(original_csv_path)
    ratings = [float(review["rating"]) for review in reviews]

    reviews_with_sentiment, sentiments_count = assign_sentiment(
        reviews, negative_threshold, positive_threshold
    )
    print(f"Sentiment counts: {sentiments_count}")

    write_reviews_with_sentiment(new_csv_path, reviews_with_sentiment)

    print_statistics(ratings)
    #plot_rating_distribution(ratings)


if __name__ == "__main__":
    main()
