'Module containing unit tests on yield function fitting functionality.'

import unittest

import numpy as np

from formable.yielding.yield_functions import Dummy2DYieldFunction


class BasicFittingTestCase(unittest.TestCase):
    'Basic fitting tests using the 2D dummy yield function.'

    def test_known_fit(self):
        'Test the fit parameter is as expected for given input data.'
        # TODO
