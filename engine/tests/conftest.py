import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

if not torch.cuda.is_available():
    pytest.skip("the engine tests need a GPU (docker compose run dev)", allow_module_level=True)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
