import pandas as pd

from matching_items_code.exploration.exploration import Explore


def test_input_correct_type():
    expl = Explore()
    assert isinstance(expl.frame, pd.DataFrame)
