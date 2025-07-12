import pytest
from rorycommon.utils import Utils

def test_01():
    lr = Utils.read_chunks_numpy(
        chunk_shape=(10,2),
        filename="/rory/source/classificationc0r996a5k31model.npy"
    )

    for c in lr:
        print(c)
    # print(lr)
