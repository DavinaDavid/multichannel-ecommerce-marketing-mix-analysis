# Marketing Mix Modeling with Ridge Regression and SHAP

This is my marketing analytics project on a multi-channel eCommerce dataset. I wanted to figure out which ad channels actually drive purchases, what each one returns per dollar, and where the budget should move. I used Ridge regression as the main model and SHAP to sanity-check what the model was doing.

The full slide deck is in `reports/MA_Project_PPT.pdf`.

## The data

About 132,000 daily rows and 50 columns, covering roughly 2019 to 2024. The target is `ALL_PURCHASES`. Media channels are Google (Paid Search, Shopping, PMax, Display, Video), Meta (Facebook, Instagram, Other) and TikTok.

The raw file isn't in this repo, so you'll need to drop your own copy into `data/`.

A lot of the media columns are mostly empty. TikTok is missing about 97% of its values and Google Video about 94%. I filled these with 0 on the assumption that a blank means no ads were running that day. There's also one huge purchase spike around 2021 that I kept in rather than deleting it, since I couldn't tell if it was a real campaign or a data glitch.

## What I did

1. **Cleaned and sorted** the data by date and dropped rows with no purchase count.
2. **Built features** in three groups: media spend, control variables (territory, promotions, trend, month/quarter/weekday) and time variables (day of week, weekend flag).
3. **Transformed media spend** with adstock (decay of 0.6) so ads keep working after the day they run, and a Hill function so returns flatten at high spend.
4. **Split by time**, not randomly. The first 80% of dates were for training (97,425 rows) and the last 20% for validation (35,334 rows), so the model never sees the future.
5. **Fit two models.** `Ridge(alpha=1.0)` is the main one. `BayesianRidge()` was only a cross-check.
6. **Ran SHAP** on the Ridge model and compared it with the channel contributions from the MMM.
7. **Simulated four budget scenarios** to see which allocation strategy does best.

## Results

How the model did on the validation set:

| Metric | Value |
|---|---|
| R² | 0.0029 |
| RMSE | 1493.85 |
| MAPE | 384.35% |

To be upfront: this is a weak fit. The model doesn't predict daily purchases well, so I treat everything below as directional, not as hard numbers.

Estimated ROI by channel:

| Channel | ROI | Verdict |
|---|---|---|
| Meta Facebook | 79.74 | High efficiency, under-invested |
| Google Shopping | 39.66 | High efficiency, under-invested |
| TikTok | 8.03 | Medium |
| Google Paid Search | 7.27 | Medium |
| Google PMax | -24.85 | Low, over-invested |
| Meta Instagram | -65.41 | Low, over-invested |

SHAP and the MMM agree on Meta Facebook. They disagree on Instagram and PMax, which SHAP ranks as very important while the MMM gives them negative returns. Google Shopping goes the other way: great ROI but low SHAP importance. The SHAP dependency plots also suggest Meta and PMax are saturating while TikTok still responds to extra spend.

Scenario results:

| Scenario | Incremental purchases | ROI |
|---|---|---|
| Equal allocation (+20% everywhere) | 1.53M | 12.16 |
| mROI-optimized (+50% on best channel) | 5.26M | 79.74 |
| Saturation-aware (-30% on saturated channels) | 2.80M | 46.61 |
| Random (+10% on half the channels) | 0.87M | 23.51 |

## What I'd recommend

Spread-it-evenly budgeting loses to targeted budgeting. Put more into the channels with high marginal ROI (Meta Facebook, Google Shopping), cut back on the saturated or negative ones (PMax, Instagram), leave TikTok and Paid Search roughly where they are, and keep a small budget aside to test things like scaling up TikTok.

## Things to keep in mind

- The R² is close to zero, so don't take the exact ROI numbers literally.
- I didn't force coefficients to be positive. The negative ROI for Instagram and PMax could just be multicollinearity, since Google and Meta spend move together a lot. It doesn't necessarily mean those channels hurt sales.
- The adstock decay and Hill parameters were set by hand, not estimated from the data.
- Before moving real budget, I'd want to confirm this with an incrementality test or a geo experiment.
- The title of the course project mentions Meridian, but this notebook uses my own adstock/Hill code with scikit-learn, not the Meridian library.

## Running it

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
jupyter notebook notebooks/mmm_ridge_shap.ipynb
```

You'll need pandas, numpy, scikit-learn, matplotlib, seaborn, shap and jupyter.

## Folder layout

```
data/        raw dataset (not included)
notebooks/   the analysis notebook
reports/     project slides (PDF)
```

## Author

Davina David
