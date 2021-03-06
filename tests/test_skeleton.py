#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from sibp_project.skeleton import fib

__author__ = "Ognjen Babovic"
__copyright__ = "Ognjen Babovic"
__license__ = "none"


def test_fib():
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)
