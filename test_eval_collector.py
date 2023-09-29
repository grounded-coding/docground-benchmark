import glob
import numpy as np
from src.eval_collector import DSTCHumanEvalCollector, BEGINHumanEvalCollector
import unittest


# Unittest for DSTCHumanEvalCollector

class TestEvalCollector(unittest.TestCase):
    def setUp(self):
        self.dstc_eval_path = "tests/static/entry0.human_eval.json"
        self.begin_eval_path = "tests/static/begin_dev_cmu.tsv"
        self.dstc_eval_collector = DSTCHumanEvalCollector(self.dstc_eval_path)
        self.begin_eval_collector = BEGINHumanEvalCollector(self.begin_eval_path)

    def test_extract_ratings(self):
        sample_indices = [7, 18]
        human_rating_1 = self.dstc_eval_collector.extract_ratings(sample_indices, human_dims=["accuracy", "appropriateness"])
        self.assertEqual(len(human_rating_1), len(sample_indices))

        sample_indices = [7, 13, 14, 15, 16]
        human_rating_2 = self.begin_eval_collector.extract_ratings(sample_indices, human_dims=["accuracy", "appropriateness"])
        self.assertEqual(len(human_rating_2), len(sample_indices))


if __name__ == '__main__':
    unittest.main()