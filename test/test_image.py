
import unittest
import numpy as np
from dagen.image.image import gen_item, merge_samples, get_ds_counting, get_ds_simple


class Tests(unittest.TestCase):

    def test_simple_old(self):
        print("generating sample output")
        cnt_samples = 10
        X_train = np.array([gen_item(i % 2) for i in range(cnt_samples)])
        Y_train = np.array([i % 2 for i in range(cnt_samples)], dtype=np.int32)
        print(X_train.shape, Y_train.shape)
        im = merge_samples(X_train, Y_train)
        im.save("/tmp/test.png")

    def test_counting(self):
        X_train, Y_train = get_ds_counting()
        print(X_train.shape, Y_train.shape)
        im = merge_samples(X_train, Y_train)
        im.save("/tmp/counting.png")

    def test_simple(self):
        X_train, Y_train = get_ds_simple(cnt_samples=10)
        print(X_train.shape, Y_train.shape)
        im = merge_samples(X_train, Y_train)
        im.save("/tmp/simple.png")
