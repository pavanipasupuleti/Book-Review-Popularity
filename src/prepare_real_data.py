# src/prepare_real_data.py

import json
import gzip
import csv
from pathlib import Path

RAW_DATA_PATH = "real_data/goodreads_reviews_fantasy_paranormal.json.gz"
OUTPUT_CSV = "data/goodreads_reviews_sample.csv"

MAX_ROWS = 100_000  # keep it safe for your Mac


def extract_reviews():
    Path("data").mkdir(exist_ok=True)

    with gzip.open(RAW_DATA_PATH, "rt", encoding="utf-8") as f, \
         open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as out:

        writer = csv.writer(out)
        writer.writerow([
            "review_id",
            "book_id",
            "user_id",
            "rating",
            "review_text",
            "n_votes",
            "n_comments"
        ])

        for i, line in enumerate(f):
            if i >= MAX_ROWS:
                break

            review = json.loads(line)

            # Skip empty reviews
            if not review.get("review_text"):
                continue

            writer.writerow([
                review.get("review_id"),
                review.get("book_id"),
                review.get("user_id"),
                review.get("rating"),
                review.get("review_text"),
                review.get("n_votes", 0),
                review.get("n_comments", 0)
            ])

            if i % 10_000 == 0 and i > 0:
                print(f"Processed {i} reviews...")

    print(f"Saved cleaned data to {OUTPUT_CSV}")


if __name__ == "__main__":
    extract_reviews()
