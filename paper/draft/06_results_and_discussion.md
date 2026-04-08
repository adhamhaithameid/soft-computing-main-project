# Results and Discussion

## Binary Track Top Results

| feature_set   | model   |   accuracy |   precision |   recall |       f1 |   roc_auc |
|:--------------|:--------|-----------:|------------:|---------:|---------:|----------:|
| svd           | SVM     |   0.971913 |    0.9541   | 0.903042 | 0.927855 |  0.994476 |
| original      | MLP_ANN |   0.971478 |    0.952678 | 0.902168 | 0.926708 |  0.987729 |
| pca           | SVM     |   0.971217 |    0.954334 | 0.899128 | 0.925894 |  0.99444  |
| original      | SVM     |   0.970695 |    0.954641 | 0.896084 | 0.924411 |  0.994633 |
| pca           | MLP_ANN |   0.970261 |    0.942601 | 0.906519 | 0.924179 |  0.988188 |
| svd           | MLP_ANN |   0.968956 |    0.936237 | 0.906519 | 0.921132 |  0.988838 |
| embedded_l1   | MLP_ANN |   0.968956 |    0.956179 | 0.885658 | 0.919219 |  0.984242 |
| embedded_l1   | SVM     |   0.968348 |    0.95194  | 0.88652  | 0.918028 |  0.994567 |
| ga_selection  | MLP_ANN |   0.966174 |    0.927138 | 0.90174  | 0.914248 |  0.990071 |
| ga_selection  | SVM     |   0.965391 |    0.948161 | 0.874787 | 0.909982 |  0.992508 |
| filter_kbest  | SVM     |   0.953738 |    0.928258 | 0.83304  | 0.877969 |  0.981683 |
| wrapper_sfs   | SVM     |   0.951391 |    0.928086 | 0.820441 | 0.870699 |  0.981993 |

## Multiclass Track Top Results

| feature_set   | model   |   accuracy |   precision |   recall |       f1 |
|:--------------|:--------|-----------:|------------:|---------:|---------:|
| original      | MLP_ANN |   0.674696 |    0.676843 | 0.674695 | 0.675497 |
| svd           | MLP_ANN |   0.671305 |    0.672467 | 0.671304 | 0.671783 |
| pca           | MLP_ANN |   0.670869 |    0.671832 | 0.670867 | 0.671112 |
| embedded_l1   | MLP_ANN |   0.662    |    0.663587 | 0.662005 | 0.6624   |
| ga_selection  | MLP_ANN |   0.631739 |    0.636359 | 0.631749 | 0.633125 |
| filter_kbest  | MLP_ANN |   0.572695 |    0.574289 | 0.572684 | 0.572833 |
| pca           | SVM     |   0.550434 |    0.596806 | 0.550434 | 0.532447 |
| svd           | SVM     |   0.548348 |    0.593987 | 0.548347 | 0.530409 |
| original      | SVM     |   0.547217 |    0.595486 | 0.547216 | 0.529386 |
| embedded_l1   | SVM     |   0.543565 |    0.592687 | 0.543568 | 0.525249 |
| ga_selection  | SVM     |   0.527305 |    0.57369  | 0.527304 | 0.506558 |
| wrapper_sfs   | MLP_ANN |   0.515912 |    0.523028 | 0.51591  | 0.518585 |

## Notes
- Discuss preprocessing effects and statistical analysis.
- Compare PCA/LDA/SVD performance.
- Compare filter/wrapper/embedded/GA feature selection outcomes.
- Discuss overfitting/underfitting patterns from fold behavior and confusion matrices.
