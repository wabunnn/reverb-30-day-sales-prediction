from pathlib import Path
import pandas as pd

path = Path("data/Reverb_30_Day_Sales_Master_Dataset.csv")
if not path.exists():
    raise SystemExit(f"Private dataset not found at {path}. See README.md")
df = pd.read_csv(path, low_memory=False)
print("Rows and columns:", df.shape)
print("Unique listing IDs:", df.listing_id.nunique())
print("Duplicate listing IDs:", df.listing_id.duplicated().sum())
print("Label status:\n", df.label_status.value_counts(dropna=False))
print("Outcome:\n", df.sold_within_30_days.value_counts(dropna=False))
print("Boundary reviews:", df.sale_date_boundary_review.fillna(False).sum())
print("Missing values in candidate features:\n", df[["condition", "product_type", "exported_listing_price", "exported_shipping_price"]].isna().sum())
print("CAUTION: exported price and shipping are not verified at publication.")
