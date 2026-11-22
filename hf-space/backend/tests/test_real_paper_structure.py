"""Regression tests built on the structure of a real IEEE-style paper (GFB-KAN): extra Params column,
percent values, efficiency/literature tables that must not be read as results, and an unreported seed."""
from app.ingest import heuristic_spec

TEXT = """Gated Fuzzy-Bell Kolmogorov-Arnold Networks (GFB-KAN): A Compact
Shallow Neuro-Fuzzy Classifier
Dr. Ali Abbas
Dept. of Computer Science
Abstract—We evaluate GFB-KAN under stratified 5-fold cross-validation on six benchmarks.
I. INTRODUCTION
Neuro-fuzzy models are compact.
TABLE I
GFB-KAN PERFORMANCE OVER ALL EVALUATION CRITERIA
%s
While Table I shows raw performance the model is small.
TABLE II
PARAMETER COUNT AND SIZE COMPARISON: GFB-KAN VS. BASELINE KAN FAMILIES
%s
y = yfuzzy + ybase
REFERENCES
[1] A. Author, Title, 2020.
""" % ('Dataset\nParams\nYouden’s J\nMacro Rec.\nG-Mean\nBal. Acc.\nMacro Spec.\nADULT\n118\n53.84%\n76.92%\n76.92%\n76.92%\n76.92%\nHAR\n13,506\n95.77%\n96.49%\n97.87%\n97.88%\n99.27%\nMNIST\n31,470\n87.15%\n88.43%\n93.43%\n93.57%\n98.72%', 'Dataset\nDims (I × C)\nGFB-KAN\nSize Reduction\nADULT\n14 × 2\n118\n11.9× to 34.7× smaller\nHAR\n561 × 6\n13,506\n4.0× to 12.0× smaller\nMNIST\n784 × 10\n31,470\n2.4× to 7.2× smaller')


def _spec():
    return heuristic_spec([TEXT])


def test_datasets_are_names_only():
    s = _spec()
    assert s.datasets == ["ADULT", "HAR", "MNIST"]
    assert not any(x.lower().startswith(("dr.", "dept", "params", "youden")) for x in s.datasets)


def test_percent_results_with_extra_params_column():
    s = _spec()
    r = {(x.dataset, x.metric): x.value for x in s.reported_results}
    assert r[("ADULT", "youden\u2019s j")] == 0.5384
    assert r[("HAR", "g-mean")] == 0.9787
    assert len(s.reported_results) == 15          # 3 datasets x 5 metrics, nothing from Table II


def test_efficiency_table_not_used_and_formula_not_hyperparameter():
    s = _spec()
    assert s.hyperparameters == {}
    assert "No hyperparameters found" in s.missing_info
    assert not any("smaller" in str(x.value) for x in s.reported_results)


def test_unreported_items_stay_missing_and_cv_found():
    s = _spec()
    assert "Random seed not reported" in s.missing_info
    assert any("5-fold" in x for x in s.splits)
