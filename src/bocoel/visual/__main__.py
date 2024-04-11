import numpy as np

from . import launch
from .reducers import PCAReducer

if __name__ == "__main__":
    processor = PCAReducer()
    launch.main(debug=True, X=np.random.rand(100, 512) * 100, reducer=processor)
