import csv
import os


def main():
    # Path to your CSV file
    csv_file_path = "data/kvikmyndaryni-reviews.csv"

    # Check if file exists to determine if headers are needed
    file_exists = os.path.isfile(csv_file_path)

    # Open the file in append mode
    with open(csv_file_path, mode="a", newline="", encoding="utf-8") as file:
        csv_writer = csv.writer(file)

        if not file_exists:
            csv_writer.writerow(["movie", "review", "rating"])

        num_of_reviews = sum(1 for line in open(csv_file_path, encoding="utf-8")) - 1

        while True:
            # Clear the screen
            os.system("cls" if os.name == "nt" else "clear")

            print(f"Number of reviews in the CSV file: {num_of_reviews}")
            print("Enter the movie name, review and rating. Type 'exit' to finish.")
            print("--------------------------------------------------------------")

            # Prompt user for input

            movie = input("Enter the movie name: ")
            if movie.lower() == "exit":
                break
            review = input("Enter your review: ")
            if review.lower() == "exit":
                break
            rating = input("Enter your rating: ")
            if rating.lower() == "exit":
                break

            # Write the user input to the CSV file
            csv_writer.writerow([movie, review, rating])
            num_of_reviews += 1


if __name__ == "__main__":
    main()
