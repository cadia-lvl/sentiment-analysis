import csv

# Define the file name
filename = "../data/Hannes-Movie-Reviews-no-noise.csv"

# Read the CSV and sort by 'rating'
with open(filename, newline="", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    sorted_list = sorted(reader, key=lambda row: float(row["rating"]))


# Function to save the classification
def save_classification(classifications):
    with open("classified_reviews.csv", mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "rating", "review", "classification"])
        for row in classifications:
            writer.writerow(row)


# Placeholder for the classified reviews
classified_reviews = []

# Classification shorthand options
options = {"p": "positive", "n": "negative", "nu": "neutral"}

# Iterate and input classification
for row in sorted_list:
    print(f"Review: {row['review']}")
    shorthand = input("Classify the review (p: positive, n: negative, nu: neutral): ")
    # Input validation
    while shorthand not in options:
        print("Invalid input. Please use the shorthand options.")
        shorthand = input(
            "Classify the review (p: positive, n: negative, nu: neutral): "
        )
    classification = options[shorthand]
    classified_reviews.append((row["id"], row["rating"], row["review"], classification))

# Save the classifications to a new CSV file
save_classification(classified_reviews)
