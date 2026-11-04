# Reverb 30-Day Sale Prediction — KA1 Milestone 1

Research question: Can information available at publication predict whether a listing receives a qualifying completed order within 30 days?

## Data
Place the private `Reverb_30_Day_Sales_Master_Dataset.csv` under `data/` locally. Do not commit this file or any buyer/payment information to a public repository. The CSV was assembled from seller-owned listings, orders, and payments exports; 3,351 listings, 3,238 determined labels, 113 unresolved.

## Run
`pip install -r requirements.txt` then run `python audit_data.py`.

## Research safeguards
- One row per listing ID.
- Do not train on order, payment, label, or post-publication fields.
- Exclude unresolved labels; review near-boundary dates.
- Exported price and shipping are NOT verified as publication-time values. Do not treat them as leakage-safe until verified.
- Use chronological train/validation/test partitions with outcome-maturity safeguards.

## Next tasks
1. Verify order matching and 30-day outcome definitions.
2. Resolve date inconsistencies and boundary cases.
3. Determine which predictor fields were known at publication.
4. Fit and evaluate leakage-safe baselines.
