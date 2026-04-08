# Preprocessing Plan

- Load and clean dataset (drop ID/index-like columns).
- Validate missing values and dtypes.
- Create binary target (class 1 vs others) and multiclass target.
- Split features/labels and apply standard scaling inside CV folds.
- Perform statistical analysis:
  - min, max, mean, variance, std, skewness, kurtosis
  - covariance matrix and correlation
- Save analysis tables and visualizations for paper section.
