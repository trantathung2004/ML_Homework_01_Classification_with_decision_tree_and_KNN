import unittest
import argparse
# Import your separate unit test files
from tests import *

MODE = {
    "tree":["ig","gini"],
    "nn":["knn","ebnn"]
}

TEST_CASES = {
    "gini":DecisionTreeTestCaseGini,
    "ig":DecisionTreeTestCaseIG,
    "knn":KNNTestCase,
    "ebnn":EpsilonBallTestCase
}
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='The main script to run tests on your code implementation.'
        )
    parser.add_argument('--model', type=str,
                        help='Select a model to test, "tree" for decision tree,"nn" for nearest neighbors',
                        default="tree")
    parser.add_argument('--mode',  type=str,
                        help='Select a mode in your model to demo, this can be "ig" and "gini" for decision tree, and "knn" and "ebnn" for ',
                        default="gini")
    args = parser.parse_args()
    
    if args.mode not in MODE[args.model]:
        resp = f"Model {args.model} only has {len(MODE[args.model])} modes:"
        for mode in MODE[args.model]:
            resp += f"\n\t - {mode}"
        raise Exception(resp)
    
    # Create a test suite
    suite = unittest.TestSuite()

    # Add test classes from your separate unit test files
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TEST_CASES[args.mode]))

    # Run the tests
    runner = unittest.TextTestRunner()
    result = runner.run(suite)