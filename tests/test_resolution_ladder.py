"""The resolution certificate must test the rung used in production."""

import pytest

from validation.resolution_ladder import production_pair


def test_production_rung_is_the_decisive_pair():
    rungs, pair = production_pair([6608, 1652, 3304], 1652)
    assert rungs == [1652, 3304, 6608]
    assert pair == (1652, 3304)


def test_ladder_without_production_rung_fails_closed():
    with pytest.raises(ValueError, match="lowest rung must be production"):
        production_pair([3304, 6608], 1652)


def test_ladder_needs_two_distinct_rungs():
    with pytest.raises(ValueError, match="at least two distinct"):
        production_pair([1652, 1652], 1652)
