# Similar Projects Playbook (Epileptic Seizure Recognition)

This guide summarizes what strong seizure-recognition AI projects commonly do and how to apply those ideas in this repository.

## 1) Common Patterns Seen in Similar Projects

1. Strong preprocessing before modeling
- Typical steps: filtering, artifact handling, bad-channel control, and reference handling for EEG pipelines.
- Why it matters: cleaner signal quality improves downstream stability.

2. Multi-family model comparisons
- Most projects compare classical ML (SVM/RF/KNN/LR) and DL (CNN/LSTM/GRU/BiLSTM).
- Why it matters: different architectures respond differently to scaling, PCA, and feature selection.

3. Explicit feature pipeline experiments
- Teams test scaling variants, dimensionality reduction (PCA), and selection (chi-square / ANOVA / wrappers).
- Why it matters: many high-scoring results are driven by feature pipeline quality, not only model choice.

4. Robust evaluation protocols
- Common practice: stratified CV, leakage-safe pipeline design, and per-run reproducibility tracking.
- Why it matters: prevents inflated metrics and makes results defensible.

5. Failure tracking and reproducibility artifacts
- Better repos log skipped/failed configs, seed settings, environment, and run manifests.
- Why it matters: reviewers can audit why some combinations fail.

6. Practical deployment awareness
- Advanced repos include runtime constraints, explainability notes, and generalization limits.
- Why it matters: strong benchmark scores alone are not enough for real use.

## 2) What to Apply Here Next (Concrete)

1. Add a hyperparameter Cartesian stage
- Include model grids for SVM, KNN, MLP, DT, LR.
- Keep current full-stage combinations; add optional tuned mode.

2. Add multi-seed evaluation
- Repeat runs across seeds (for example 3 or 5), then aggregate mean/std per combo.

3. Add leakage-safe grouped CV option
- Add `GroupKFold` mode for subject/session-aware splits when group IDs are available.

4. Add imbalance handling stage
- Compare `none`, class weights, and SMOTE variants as one extra benchmark axis.

5. Add an interactive results browser
- Streamlit/Gradio page to filter by track/model/stages and compare run history quickly.

6. Add “publish profile”
- One command that runs strict validation, snapshots outputs, and exports a clean release package.

## 3) Sources

- MNE preprocessing tutorials: https://mne.tools/1.7/auto_tutorials/preprocessing/index.html
- scikit-learn cross-validation guide: https://scikit-learn.org/stable/modules/cross_validation.html
- scikit-learn nested CV example: https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
- Systematic review (ML/DL seizure recognition): https://www.sciencedirect.com/science/article/abs/pii/S0006899325003580
- Optimization study (scaling/PCA/chi-square + DL): https://link.springer.com/article/10.1007/s00521-023-09204-6
- Example open-source seizure ML workflow repo: https://github.com/dragonpilee/Epileptic-Seizure-Detection-System
- Example open-source seizure app repo: https://github.com/mrpintime/Seizure-Prediction-System
