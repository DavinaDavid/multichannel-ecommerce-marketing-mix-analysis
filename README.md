# Marketing Mix Modeling with Ridge Regression & SHAP Explainability

> Quantifying channel contribution, ROI and marginal ROI (mROI) for a multi-channel eCommerce business, and using SHAP to explain the model and guide budget allocation.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Ridge%20%7C%20BayesianRidge-orange)
![SHAP](https://img.shields.io/badge/explainability-SHAP-purple)

## Overview

This project builds a Marketing Mix Model (MMM) for a multi-channel eCommerce company. The goals are to:

- Estimate how much each paid media channel contributes to purchases
- Compute **ROI** and **marginal ROI (mROI)** per channel
- Model **carryover (adstock)** and **diminishing returns (Hill saturation)**
- Explain the model with **SHAP** and compare explanations with MMM contributions
- Simulate budget scenarios to recommend a better allocation

## Dataset

| Item | Detail |
|---|---|
| Observations | 132,759 rows |
| Variables | 50 columns |
| Time range | ~2019 to 2024 (daily, `DATE_DAY`) |
| Target | `ALL_PURCHASES` |
| Media groups | `media_spend_cols`, `media_impr_cols`, `media_click_cols` |
| Channels | Google (Paid Search, Shopping, PMax, Display, Video), Meta (Facebook, Instagram, Other), TikTok |

> The raw data is not included in this repository. Place your file in `data/` (see [Usage](#usage)).

## Methodology

### 1. Preprocessing
- Converted `DATE_DAY` to datetime and sorted chronologically
- Dropped rows with missing `ALL_PURCHASES`
- Filled missing media spend/impression/click values with `0` (assumes no activity)
- High missingness in media columns, e.g. TikTok (97.4%), Google Video (94.3%), Google Display (85.9%)

### 2. Feature Engineering
- **Media variables:** paid channel spend, transformed with adstock and Hill saturation
- **Control variables:** territory, promotions, trend, seasonality (month, quarter, weekday)
- **Time variables:** day of week, month, quarter, weekend flag

### 3. Media Transformations
- **Adstock:** recursive decay (`decay = 0.6`) to model lagged ad effects
- **Hill function:** non-linear saturation using `alpha` and `gamma` parameters

### 4. Train / Validation Split
Time-based split to avoid leakage: earliest 80% of unique dates for training (97,425 rows), latest 20% for validation (35,334 rows).

### 5. Models
| Model | Role |
|---|---|
| `Ridge(alpha=1.0)` | Primary MMM for contributions, ROI/mROI and scenarios |
| `BayesianRidge()` | Robustness check on coefficient stability |

### 6. Explainability
SHAP is applied to the Ridge model to rank features, identify positive/negative drivers, and compare against MMM contributions.

## Results

### Model performance (validation)

| Metric | Value |
|---|---|
| R² | 0.0029 |
| RMSE | 1493.85 |
| MAPE | 384.35% |

**Note:** predictive fit is very low. The model is best viewed as a baseline framework; results below should be treated as directional, not causal. See [Limitations](#limitations).

### Channel ROI and efficiency

| Channel | ROI / mROI | Efficiency class |
|---|---|---|
| Meta Facebook | 79.74 | High (under-invested) |
| Google Shopping | 39.66 | High (under-invested) |
| TikTok | 8.03 | Medium |
| Google Paid Search | 7.27 | Medium |
| Google PMax | -24.85 | Low (over-invested) |
| Meta Instagram | -65.41 | Low (over-invested) |

### SHAP vs MMM
- **Meta Facebook:** positive MMM contribution, aligned with SHAP
- **Meta Instagram and Google PMax:** high SHAP importance but negative MMM contribution (conflict)
- **Google Shopping:** high ROI but low SHAP importance (partial mismatch)
- SHAP dependency plots suggest Meta and PMax show saturation while TikTok still responds to additional spend

### Scenario simulation

| Scenario | Incremental Purchases | ROI |
|---|---|---|
| Equal allocation (+20% all channels) | 1.53M | 12.16 |
| **mROI-optimized (+50% on best channel)** | **5.26M** | **79.74** |
| Saturation-aware (-30% on saturated channels) | 2.80M | 46.61 |
| Random allocation (+10% on half the channels) | 0.87M | 23.51 |

## Key Recommendations

1. Increase investment in high-mROI channels (Meta Facebook, Google Shopping)
2. Reduce spend in saturated or negative-return channels (Google PMax, Meta Instagram)
3. Maintain stable mid-performing channels (TikTok, Google Paid Search)
4. Reserve a small budget for experimentation, e.g. testing TikTok scale-up

## Limitations

- Very low R² on validation: the model explains almost none of the variance in purchases
- Ridge is fit with no positivity constraint, so negative channel effects may reflect multicollinearity rather than true harm
- A single extreme purchase spike (~2021) can heavily influence results and should be handled explicitly
- Media channels are strongly correlated with each other, which makes attribution unstable
- Adstock decay and Hill parameters are fixed manually rather than estimated
- ROI figures are model-based estimates; validate with incrementality tests or geo experiments before reallocating budget


## Usage

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
jupyter notebook notebooks/mmm_ridge_shap.ipynb
```

Suggested `requirements.txt`:

```
pandas
numpy
scikit-learn
matplotlib
seaborn
shap
jupyter
```

## Tech Stack

Python, pandas, NumPy, scikit-learn (Ridge, BayesianRidge), SHAP, Matplotlib, Seaborn

## Author

**Davina David**
Marketing Analytics Project


Released under the MIT License. See `LICENSE` for details.

