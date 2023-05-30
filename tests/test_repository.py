import pytest
import pandas as pd
from keras import Sequential

from matching_items_code.product_matching.repository import GenerateModel


class TestGenerateModel:
    @pytest.fixture
    def model(self):
        df = pd.DataFrame({'cleaned_text': ['This is a simple text', 'And this is another text']})
        return GenerateModel(df)

    def test_init(self, model):
        assert model.model is None
        assert model.padded_sequences is None
        assert model.sequences is None
        assert model.tokeniser is None

    def test_prepare_text(self, model):
        model._prepare_text()
        assert model.tokeniser is not None
        assert model.sequences is not None
        assert model.padded_sequences is not None

    def test_generate(self, model):
        returned_model, tokeniser = model.generate()
        assert isinstance(returned_model, Sequential)
        assert model.model is not None
        assert model.tokeniser is not None
        assert model.model == returned_model
        assert model.tokeniser == tokeniser