import os
import pandas as pd
import csv
from googletrans import Translator
from concurrent.futures import ThreadPoolExecutor
import threading
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
translator = Translator()

write_lock = threading.Lock()
thread_local = threading.local()


# Create a thread-local variable to store the translator object for each thread
# this is to avoid thread deadlock issues with the googletrans library
def get_translate_pipeline():
    # Initialize the pipeline only if it has not been initialized for this thread
    if not hasattr(thread_local, "translate"):
        thread_name = threading.current_thread().name
        logging.info("Initializing translation pipeline for thread: " + thread_name)
        start = time.time()
        thread_local.translate = Translator()
        end = time.time()
        logging.info(
            f"Translation pipeline initialized for {thread_name} in {end - start} seconds"
        )
    return thread_local.translate


def google_translate_review(review, index, max_retries=10):
    translator = get_translate_pipeline()
    thread_name = threading.current_thread().name
    review = review.replace("<br />", "")
    retries = 0
    while retries < max_retries:
        try:
            return translator.translate(review, dest="is").text
        except AttributeError as e:
            logging.warning(
                f"Translation failed for index: {index} by: {thread_name} with error: {e}. Retry #{retries}"
            )
            retries += 1
            time.sleep(3)
        except TypeError as e:
            logging.warning(
                f"Translation failed for index: {index} by: {thread_name} with error: {e}"
            )
            return None
    logging.error(
        f"Max retries reached for index: {index} by: {thread_name}. Giving up."
    )
    return None


def save_review(index, review, sentiment, writer, failed_writer):
    thread_name = threading.current_thread().name
    start = time.time()

    translated_review = google_translate_review(review, index)

    with write_lock:
        if translated_review is None:
            failed_writer.writerow([review, sentiment])
        else:
            writer.writerow([translated_review, sentiment])
    end = time.time()
    logging.info(f"Processed index: {index} by: {thread_name} in {end-start} seconds.")


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    dataset = os.path.join(parent_dir, "Datasets/IMDB-Dataset.csv")

    if not dataset or not Path(dataset).exists() or not Path(dataset).is_file():
        print("Invalid dataset path")
        return

    try:
        df = pd.read_csv(dataset)
        reviews = df["review"]
        sentiments = df["sentiment"]
    except Exception:
        print("Invalid dataset")
        return

    with open(
        os.path.join(parent_dir, "Datasets/IMDB-Dataset-GoogleTranslate.csv"),
        "a",
        newline="",
        encoding="utf-8",
    ) as trans_file, open(
        os.path.join(parent_dir, "Datasets/failed-IMDB-Dataset-GoogleTranslate.csv"),
        "a",
        newline="",
        encoding="utf-8",
    ) as failed_file:
        writer = csv.writer(trans_file)
        writer.writerow(["review", "sentiment"])

        failed_writer = csv.writer(failed_file)
        failed_writer.writerow(["review", "sentiment"])

        start = time.time()
        with ThreadPoolExecutor() as executor:
            for index, (review, sentiment) in enumerate(zip(reviews, sentiments)):
                executor.submit(
                    save_review, index, review, sentiment, writer, failed_writer
                )
        end = time.time()
        print(f"Time taken: {end-start}")


if __name__ == "__main__":
    main()
