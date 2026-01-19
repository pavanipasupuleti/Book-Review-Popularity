import pandas as pd
from features import add_popularity_label, add_text_features

INPUT_CSV = "data/goodreads_reviews_sample.csv"
OUTPUT_CSV = "data/goodreads_reviews_processed.csv"

if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV)

    df = add_popularity_label(df)
    df = add_text_features(df)

    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved processed data to {OUTPUT_CSV}")
