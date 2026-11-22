import os, tempfile, shutil
from app.config import Settings
from app.storage import Storage
from app.sandbox import LocalSandbox
from app.graph import Engine
from app.schemas import Budget

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "..", "sample_papers", "fisher_lda_iris_fixture.pdf")


def sample_pdf() -> bytes:
    with open(SAMPLE, "rb") as f:
        return f.read()


def tmp_settings(**kw) -> Settings:
    d = tempfile.mkdtemp(prefix="p2p_test_")
    return Settings(data_dir=d, db_path=os.path.join(d, "t.db"), **kw)


class StubProvider:
    """TEST DOUBLE ONLY - returns canned dicts. Never used by the product; marks usage so tests can assert calls."""
    name, is_mock = "stub", True

    def __init__(self, responses):
        self.responses, self.calls = list(responses), 0
        from app.llm.provider import Usage
        self.usage = Usage()

    def generate_json(self, prompt, system=""):
        self.calls += 1
        return self.responses.pop(0)

    def info(self):
        return {"provider": "stub", "mock": True}


def make_engine(provider=None, **kw):
    s = tmp_settings(**kw)
    db = Storage(s.db_path)
    return Engine(s, db, LocalSandbox(), provider), s, db


def drive(engine, pid, accept=True):
    engine.run_ingest(pid)
    engine.approve_spec(pid, None)
    engine.run_plan(pid)
    engine.approve_plan(pid, accept)
    engine.run_execution(pid)


def make_pdf(pages: list[list[str]], row_height: int = 13) -> bytes:
    """Build a text PDF where EVERY string is its own text line (like table cells extracted from a real paper)."""
    import pymupdf
    doc = pymupdf.open()
    for lines in pages:
        pg = doc.new_page()
        y = 50
        for t in lines:
            pg.insert_text((50, y), t, fontsize=9)
            y += row_height
    return doc.tobytes()


# A realistic (non-template) paper: unnumbered/numbered headings that are NOT the fixture's exact names,
# and a results TABLE whose header cells are "Dataset", "Youden's J", "Macro Rec.", "G-Mean".
IMBALANCED_PAPER = [[
    "Cost-Sensitive Imbalanced Learning with Margin Boosting",
    "A. Researcher and B. Scientist",
    "Abstract",
    "We propose MarginBoost for imbalanced multi-class learning.",
    "1. Introduction",
    "Imbalanced learning is hard.",
    "3. Methodology",
    "MarginBoost reweights samples by their margin at each round.",
    "3.1 Margin loss",
    "The loss penalises small margins.",
    "4. Experimental Setup",
    "We use a learning rate of 0.05, 200 estimators and a random seed of 7.",
    "Evaluation uses a 70/30 stratified split.",
    "5. Results",
    "Table 2 reports performance on the benchmark datasets.",
    "Dataset", "Youden's J", "Macro Rec.", "G-Mean",
    "Wine", "0.91", "0.93", "0.94",
    "Glass", "0.62", "0.66", "0.70",
    "6. Conclusion",
    "MarginBoost works.",
]]

WINE_PAPER = [[
    "Nearest centroid classification of wine",
    "Test Authors",
    "Abstract",
    "We classify wines with a nearest centroid classifier.",
    "Method",
    "A nearest centroid classifier on standardised features.",
    "Datasets",
    "- Wine recognition dataset (178 samples)",
    "Hyperparameters",
    "- random_seed: 42",
    "- test_size: 0.3",
    "Evaluation Metrics",
    "Accuracy on the held-out split.",
    "Reported Results",
    "Accuracy: 0.96 on the wine dataset using a 70/30 split.",
]]
