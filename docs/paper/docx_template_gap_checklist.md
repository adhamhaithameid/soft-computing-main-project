# DOCX Template Gap Checklist

Template file reviewed:
- `paper/template/Fill your Project Information in this document.docx`

Primary draft checked:
- `RESEARCH_PAPER_FINAL_DRAFT.md`

## Completion Status

1. Project Title
- Status: done

2. Abstract (500 words)
- Status: done (500 words)

3. Introduction
- Define main problem: done
- Brief techniques used: done
- Main contribution: done
- Organization of the rest of project: **missing (recommended to add a short paper structure paragraph)**

4. Related Work
- At least 10 studies: done
- Table with Reference/Year/Methods/Results: done

5. Methodology
- Brief description of each method used: done

6. Proposed Model
- Describe phases (preprocessing, feature selection, feature reduction, classification/regression, metrics): done
- High-resolution model diagram: **partially missing in paper body (figures exist in results folder, but add one explicit workflow figure in the paper text)**

7. Results and Discussion
- Dataset description: done
- Results for each method/step + comparisons: partially done
- Preprocessing phase details:
  - Data visualization: **needs explicit subsection/text in paper**
  - Missing-values treatment: **needs explicit subsection/text in paper**
  - Binning (if exists): **state whether used or not used**
  - Statistical analysis (min/max/mean/variance/std/skewness/kurtosis): **needs explicit summary table reference**
  - Covariance/correlation/heatmap: **needs explicit summary text and figure references**
  - Chi-square test / Z-test or t-test / ANOVA: **not clearly reported as statistical tests in paper body**
- Feature reduction interpretation (LDA/PCA/SVD): **needs deeper comparison paragraph and cited table/figure references**
- Classification results with tables/figures: partially done
- 80/20 train-test split in addition to K-fold:
  - Current paper reports 3-fold CV.
  - Template explicitly asks 80/20 split: **missing (or justify replacement with CV-only protocol)**
- Confusion matrix for each classifier + comparison table: **missing**
- Overfitting/underfitting interpretation from confusion matrix/metrics: **missing**

8. Conclusion and Future Work
- Status: done

9. References
- In-text citations with `[Ref No.]`: done
- APA style references: done

## High-Priority Additions

1. Add an explicit “Paper Organization” paragraph in Introduction.
2. Add preprocessing-results subsection with descriptive stats and references to generated tables/figures.
3. Add feature-reduction comparison subsection for PCA vs LDA projection vs SVD.
4. Add confusion matrix section (per classifier or top classifiers) with interpretation.
5. Add overfitting/underfitting analysis paragraph.
6. Add explicit note on 80/20 split requirement:
   - either run and report an extra 80/20 experiment,
   - or justify why stratified 3-fold CV is used as the primary protocol.

