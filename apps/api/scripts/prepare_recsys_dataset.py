from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


BASE_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/"
    "amazon_2023/benchmark/5core/last_out_w_his"
)

URLS = {
    "train": f"{BASE_URL}/Arts_Crafts_and_Sewing.train.csv.gz",
    "valid": f"{BASE_URL}/Arts_Crafts_and_Sewing.valid.csv.gz",
    "test": f"{BASE_URL}/Arts_Crafts_and_Sewing.test.csv.gz",
}

COLUMNS = ["user_id", "parent_asin", "rating", "timestamp", "history"]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing = {"user_id", "parent_asin", "rating", "timestamp"} - set(df.columns)
    if missing:
        raise RuntimeError(f"Official split is missing expected columns: {sorted(missing)}")

    keep = [column for column in COLUMNS if column in df.columns]
    df = df[keep].copy()

    df["user_id"] = df["user_id"].astype(str)
    df["parent_asin"] = df["parent_asin"].astype(str)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

    return df.dropna(subset=["user_id", "parent_asin", "timestamp"])


def select_users(num_users: int, seed: int) -> set[str]:
    print("Reading validation split to sample users...")
    valid = pd.read_csv(URLS["valid"], compression="gzip")
    valid = normalize_columns(valid)

    unique_users = valid["user_id"].drop_duplicates()
    if num_users > len(unique_users):
        raise ValueError(
            f"Requested {num_users:,} users but validation contains only "
            f"{len(unique_users):,} unique users."
        )

    users = set(unique_users.sample(n=num_users, random_state=seed).tolist())
    print(f"Selected {len(users):,} users with seed={seed}.")
    return users


def filter_split(
    split: str,
    selected_users: set[str],
    output_dir: Path,
    chunk_size: int,
) -> pd.DataFrame:
    print(f"\nStreaming {split} split...")

    selected_chunks = []
    rows_scanned = 0
    rows_kept = 0

    for chunk in pd.read_csv(
        URLS[split],
        compression="gzip",
        chunksize=chunk_size,
    ):
        rows_scanned += len(chunk)
        chunk = normalize_columns(chunk)
        filtered = chunk[chunk["user_id"].isin(selected_users)].copy()

        if not filtered.empty:
            selected_chunks.append(filtered)
            rows_kept += len(filtered)

        if rows_scanned % (chunk_size * 5) == 0:
            print(f"  scanned={rows_scanned:,} kept={rows_kept:,}")

    if not selected_chunks:
        raise RuntimeError(f"No rows were selected from {split}.")

    result = pd.concat(selected_chunks, ignore_index=True)
    result = result.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    output_path = output_dir / f"{split}.csv"
    result.to_csv(output_path, index=False)

    print(
        f"Saved {split}: {len(result):,} rows, "
        f"{result['user_id'].nunique():,} users, "
        f"{result['parent_asin'].nunique():,} items -> {output_path}"
    )
    return result


def validate_splits(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    expected_users: int,
) -> None:
    train_users = set(train["user_id"])
    valid_users = set(valid["user_id"])
    test_users = set(test["user_id"])
    common_users = train_users & valid_users & test_users

    print("\nDataset validation")
    print("=" * 55)
    print(f"requested users:              {expected_users:,}")
    print(f"train users:                  {len(train_users):,}")
    print(f"validation users:             {len(valid_users):,}")
    print(f"test users:                   {len(test_users):,}")
    print(f"users in all 3 splits:        {len(common_users):,}")

    if len(valid_users) != expected_users:
        raise RuntimeError("Not every sampled user appeared in validation.")
    if len(test_users) != expected_users:
        raise RuntimeError("Not every sampled user appeared in test.")
    if len(common_users) < int(expected_users * 0.95):
        raise RuntimeError("Too few sampled users have train/valid/test coverage.")

    train_last = train.groupby("user_id")["timestamp"].max()
    valid_time = valid.set_index("user_id")["timestamp"]
    test_time = test.set_index("user_id")["timestamp"]

    common_tv = valid_time.index.intersection(train_last.index)
    common_vt = test_time.index.intersection(valid_time.index)

    ordered_valid = (valid_time.loc[common_tv] >= train_last.loc[common_tv]).mean()
    ordered_test = (test_time.loc[common_vt] >= valid_time.loc[common_vt]).mean()

    print(f"train -> valid chronological: {ordered_valid:.2%}")
    print(f"valid -> test chronological:  {ordered_test:.2%}")

    if ordered_valid < 0.99 or ordered_test < 0.99:
        raise RuntimeError("Chronological split validation failed.")


def print_summary(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame) -> None:
    all_rows = pd.concat([train, valid, test], ignore_index=True)
    train_counts = train.groupby("user_id").size()

    print("\nDense recommender subset")
    print("=" * 55)
    print(f"users:                        {all_rows['user_id'].nunique():,}")
    print(f"unique items:                 {all_rows['parent_asin'].nunique():,}")
    print(f"train interactions:           {len(train):,}")
    print(f"validation interactions:      {len(valid):,}")
    print(f"test interactions:            {len(test):,}")
    print(f"total interactions:           {len(all_rows):,}")
    print(f"mean train interactions/user: {train_counts.mean():.2f}")
    print(f"median train/user:            {train_counts.median():.1f}")
    print(f"min train/user:               {train_counts.min():,}")
    print(f"max train/user:               {train_counts.max():,}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/recsys/arts_crafts_5core"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_users = select_users(args.users, args.seed)

    users_path = args.output_dir / "selected_users.txt"
    users_path.write_text("\n".join(sorted(selected_users)), encoding="utf-8")
    print(f"Saved user cohort -> {users_path}")

    train = filter_split("train", selected_users, args.output_dir, args.chunk_size)
    valid = filter_split("valid", selected_users, args.output_dir, args.chunk_size)
    test = filter_split("test", selected_users, args.output_dir, args.chunk_size)

    validate_splits(train, valid, test, args.users)
    print_summary(train, valid, test)

    print("\nReady for two-tower training.")


if __name__ == "__main__":
    main()
