
import unittest
import numpy as np
import dagen.image
from dagen.image.image import gen_item, merge_samples, get_ds_counting, get_ds_simple, get_ds_naive


class Tests(unittest.TestCase):

    def test_simple_old(self):
        print("generating sample output")
        cnt_samples = 10
        X_train = np.array([gen_item(i % 2) for i in range(cnt_samples)])
        Y_train = np.array([i % 2 for i in range(cnt_samples)], dtype=np.int32)
        print(X_train.shape, Y_train.shape)
        im = merge_samples(X_train, Y_train)
        im.save("/tmp/test_legacy.png")

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

    def test_naive(self):
        X_train, Y_train = get_ds_naive(cnt_samples=10)
        print(X_train.shape, Y_train.shape)
        im = merge_samples(X_train, Y_train)
        im.save("/tmp/naive.png")

    def test_merge_w_channels(self):
        X_train, Y_train = get_ds_simple(cnt_samples=10)
        X_train = np.expand_dims(X_train, axis=1).astype(np.float32) / 255
        Y_train = Y_train[:, np.newaxis]
        print(X_train.shape, Y_train.shape)
        im = merge_samples(X_train, Y_train)
        im.save("/tmp/simple_channel.png")

    def test_size(self):
        X_train, Y_train = get_ds_simple(dim_image=128, cnt_samples=10)
        print(X_train.shape, Y_train.shape)
        im = merge_samples(X_train, Y_train)
        im.save("/tmp/size128.png")
