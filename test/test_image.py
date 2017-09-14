
import unittest
import numpy as np
from dagen.image.image import gen_item, merge_sample


class Tests(unittest.TestCase):

    def test_file_iter(self):
        print("generating sample output")
        cnt_samples = 10
        X_train = np.array([gen_item(i % 2) for i in range(cnt_samples)])
        Y_train = np.array([i % 2 for i in range(cnt_samples)], dtype=np.int32)
        print(X_train.shape, Y_train.shape)
        im = merge_sample(X_train, Y_train)
        im.save("/tmp/test.png")
