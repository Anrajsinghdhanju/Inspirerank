from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/recsys/arts_crafts_5core")


def main() -> None:
    frames = {
        split: pd.read_csv(DATA_DIR / f"{split}.csv")
        for split in ("train", "valid", "test")
    }

    train = frames["train"]
    valid = frames["valid"]
    test = frames["test"]
    combined = pd.concat(frames.values(), ignore_index=True)

    train_counts = train.groupby("user_id").size()
    item_counts = train.groupby("parent_asin").size()

    print("\nInspireRank recommender dataset")
    print("=" * 52)
    print(f"users                       {combined['user_id'].nunique():,}")
    print(f"items                       {combined['parent_asin'].nunique():,}")
    print(f"train interactions          {len(train):,}")
    print(f"valid interactions          {len(valid):,}")
    print(f"test interactions           {len(test):,}")
    print(f"mean train/user             {train_counts.mean():.2f}")
    print(f"median train/user           {train_counts.median():.1f}")
    print(f"users with >=3 train        {(train_counts >= 3).sum():,}")
    print(f"users with >=5 train        {(train_counts >= 5).sum():,}")
    print(f"users with >=10 train       {(train_counts >= 10).sum():,}")
    print(f"items with >=2 train        {(item_counts >= 2).sum():,}")
    print(f"items with >=5 train        {(item_counts >= 5).sum():,}")
    print(f"items with >=10 train       {(item_counts >= 10).sum():,}")

    train_items = set(train["parent_asin"])
    valid_cold = (~valid["parent_asin"].isin(train_items)).mean()
    test_cold = (~test["parent_asin"].isin(train_items)).mean()

    print(f"validation cold-item rate   {valid_cold:.2%}")
    print(f"test cold-item rate         {test_cold:.2%}")


if __name__ == "__main__":
    main()
