# Feature Selection Plan

Methods:
- Filter: ANOVA F-test + SelectKBest.
- Wrapper: Sequential Forward Selection.
- Embedded: L1-regularized logistic regression.
- GA: custom genetic feature mask optimizer.

Outputs:
- Selected feature indices per method.
- Performance comparison across models.
- Time/performance tradeoff notes.
