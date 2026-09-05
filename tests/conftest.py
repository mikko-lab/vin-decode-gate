import pytest

from vindecode.gate import Decoder
from vindecode.model import Predictor


@pytest.fixture(scope="session")
def predictor():
    return Predictor("artifacts")


@pytest.fixture(scope="session")
def decoder(predictor):
    return Decoder(predictor)
