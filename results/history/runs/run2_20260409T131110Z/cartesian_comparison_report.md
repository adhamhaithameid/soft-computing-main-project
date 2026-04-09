# Cartesian Comparison Report

## Binary Top 10

|   rank | preprocessing   | reduction   | selection   | model   |   accuracy |     f1 |   delta_vs_baseline_accuracy |
|-------:|:----------------|:------------|:------------|:--------|-----------:|-------:|-----------------------------:|
|      1 | quantile        | pca         | none        | svm     |     0.9763 | 0.9393 |                       0.0003 |
|      2 | quantile        | none        | none        | svm     |     0.976  | 0.9386 |                       0      |
|      3 | quantile        | svd         | none        | svm     |     0.9746 | 0.9349 |                      -0.0014 |
|      4 | quantile        | none        | embedded_l1 | svm     |     0.9736 | 0.9323 |                      -0.0024 |
|      5 | standard        | svd         | none        | svm     |     0.9719 | 0.9279 |                      -0.0041 |
|      6 | minmax          | svd         | none        | mlp_ann |     0.9718 | 0.9302 |                       0.0003 |
|      7 | minmax          | pca         | none        | svm     |     0.9717 | 0.9273 |                      -0.0043 |
|      8 | robust          | svd         | none        | svm     |     0.9717 | 0.9272 |                      -0.0043 |
|      9 | robust          | pca         | none        | svm     |     0.9716 | 0.9269 |                      -0.0044 |
|     10 | standard        | none        | none        | mlp_ann |     0.9715 | 0.9267 |                       0      |

## Multiclass Top 10

|   rank | preprocessing   | reduction   | selection          | model   |   accuracy |     f1 |   delta_vs_baseline_accuracy |
|-------:|:----------------|:------------|:-------------------|:--------|-----------:|-------:|-----------------------------:|
|      1 | minmax          | pca         | none               | mlp_ann |     0.6857 | 0.685  |                       0.011  |
|      2 | standard        | none        | none               | mlp_ann |     0.6747 | 0.6755 |                       0      |
|      3 | standard        | svd         | none               | mlp_ann |     0.6713 | 0.6718 |                      -0.0034 |
|      4 | standard        | pca         | none               | mlp_ann |     0.6709 | 0.6711 |                      -0.0038 |
|      5 | robust          | none        | none               | mlp_ann |     0.6697 | 0.6704 |                      -0.005  |
|      6 | robust          | none        | embedded_l1        | mlp_ann |     0.6631 | 0.6637 |                      -0.0116 |
|      7 | robust          | pca         | none               | mlp_ann |     0.6622 | 0.6624 |                      -0.0125 |
|      8 | standard        | none        | embedded_l1        | mlp_ann |     0.662  | 0.6624 |                      -0.0127 |
|      9 | minmax          | pca         | filter_correlation | mlp_ann |     0.6535 | 0.6526 |                      -0.0212 |
|     10 | minmax          | pca         | embedded_l1        | mlp_ann |     0.6506 | 0.6489 |                      -0.0241 |