import os, unittest

HTML = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "index.html")


class TestFrontendStatic(unittest.TestCase):
    """Regression guard for the '[object HTMLTableRowElement]' bug (rows were nested arrays; h() flattened one level only)."""
    def setUp(self):
        with open(HTML, encoding="utf-8") as f:
            self.src = f.read()

    def test_dom_helper_flattens_nested_children_fully(self):
        self.assertIn("kids.flat(Infinity)", self.src)
        self.assertNotIn("kids.flat()", self.src)

    def test_no_innerhtml_with_untrusted_content(self):
        self.assertNotIn("innerHTML", self.src)    # PDF text must only ever be inserted as text nodes

    def test_report_comparison_rows_are_appended_as_dom_nodes(self):
        self.assertIn("const comparisonCard = h('div'", self.src)
        self.assertIn("tableBody.append(h('tr'", self.src)
        self.assertNotIn("s.comparisons.map(c => [h('tr'", self.src)


if __name__ == "__main__":
    unittest.main()
