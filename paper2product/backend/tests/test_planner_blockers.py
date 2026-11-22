import unittest
from unittest.mock import patch

from app.planner import apply_dataset_choice
from app.schemas import Plan, ResearchSpec


class TestPlannerBlockers(unittest.TestCase):
    def make_plan(self, blockers):
        return Plan(
            template="llm_generic",
            approach="test",
            tasks=[],
            files=["run_experiment.py"],
            packages=["numpy", "scikit-learn"],
            dataset_requirements=[],
            test_strategy="test",
            metrics=["accuracy"],
            expected_outputs=[],
            compute="CPU",
            risks=[],
            assumptions=["existing assumption"],
            open_questions=[],
            blockers=blockers,
        )

    def test_llm_gap_is_assumption_and_resolved_upload_has_no_hard_blockers(self):
        plan = self.make_plan([
            "The specification omits the precise text vectorization technique.",
            "Dataset: stale dataset blocker",
        ])
        spec = ResearchSpec(title="Text paper", datasets=["IMDB"], metrics=["accuracy"])
        preview = {
            "resolvable": True,
            "source": "upload:my.csv",
            "reason": "",
            "differs_from_paper": True,
            "user_supplied": True,
            "needs": "",
        }
        with patch("app.planner.ds_mod.preview_resolution", return_value=preview):
            result = apply_dataset_choice(plan, spec, "upload:my.csv")

        self.assertEqual(result.blockers, [])
        gap = "Gap flagged by the planner (not a hard stop; accepting it means the run proceeds without it): "
        self.assertTrue(any(a.startswith(gap) and "text vectorization" in a for a in result.assumptions))

    def test_unresolved_dataset_has_only_ds_prefix_hard_blockers(self):
        plan = self.make_plan(["The specification omits exact preprocessing details."])
        spec = ResearchSpec(title="Unknown data paper", datasets=["private-corpus"], metrics=["accuracy"])
        preview = {
            "resolvable": False,
            "source": "",
            "reason": "the paper's dataset(s) private-corpus cannot be fetched automatically",
            "differs_from_paper": False,
            "user_supplied": False,
            "needs": "Upload a CSV.",
        }
        with patch("app.planner.ds_mod.preview_resolution", return_value=preview):
            result = apply_dataset_choice(plan, spec, "")

        self.assertTrue(result.blockers)
        self.assertTrue(all(b.startswith("Dataset: ") for b in result.blockers))
        gap = "Gap flagged by the planner (not a hard stop; accepting it means the run proceeds without it): "
        self.assertTrue(any(a.startswith(gap) and "preprocessing" in a for a in result.assumptions))


if __name__ == "__main__":
    unittest.main()
