# KA1 Milestone 2 — Reverb 30-Day Sales Prediction

Research question: can available listing attributes predict a completed order within 30 days?

## Files
- `run_analysis.py`: executable preprocessing, chronological splitting, baseline/logistic/random-forest evaluation
- `audit.json`: data checks and split descriptions
- `model_metrics.csv`: validation/test scores and confusion matrices
- `KA1_Milestone_2_Research_Progress_Report.docx`: research progress report

## Reproduce
1. Install: `pip install -r requirements.txt`
2. Place the **private** `Reverb_30_Day_Sales_Master_Dataset.csv` beside `run_analysis.py`.
3. Run: `python run_analysis.py`

Do not commit the private CSV. Source exports include commercial transaction data.

## Key caveats
- Exported price/shipping may not be original publication-time values; `price_at_publication_verified` is false for all rows. Results are preliminary and may contain temporal feature contamination.
- No historical watcher-count snapshots available.
- No customer personal data are used as predictors.
- This is a single-seller study and should not be treated as a causal pricing model.

## Split
Eligible labeled records sorted by listing publication UTC, then listing ID: 70% train / 15% validation / 15% test. No tuning on test data.

## AI assistance
ChatGPT assisted with analysis code and reporting; review outputs and methods before submitting.
