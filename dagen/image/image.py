import numpy as np
import PIL
import PIL.Image
from contextfree import contextfree as cf


def get_box(dim_image=64):
    cf.init(canvas_size=(dim_image, dim_image), face_color="#FFFFFF")
    with cf.translate(cf.rnd(0.5), cf.rnd(0.5)):
        cf.box(cf.rnd(0.5) + 0.4)
    a = cf.get_npimage()
    return a[:, :, 0]


def get_circle(dim_image=64):
    cf.init(canvas_size=(dim_image, dim_image), face_color="#FFFFFF")
    with cf.translate(cf.rnd(0.5), cf.rnd(0.5)):
        cf.circle(0.25 + cf.rnd(0.2))
    a = cf.get_npimage()
    return a[:, :, 0]


def get_counting_sample(dim_image=64):
    cf.init(canvas_size=(dim_image, dim_image), face_color="#FFFFFF")
    cnt_circles = 0
    cnt_boxes = 0
    cnt_items = 3
    with cf.scale(1 / (cnt_items + 1)):
        with cf.translate(-cnt_items / 2, -cnt_items / 2):
            for y in range(cnt_items):
                with cf.translate(0, y * 1.1):
                    for x in range(cnt_items):
                        with cf.translate(x * 1.1, 0):
                            if cf.coinflip(2):
                                if cf.coinflip(2):
                                    cf.circle(0.4 + cf.rnd(0.2))
                                    cnt_circles += 1
                                else:
                                    cf.box(0.85 + cf.rnd(0.2))
                                    cnt_boxes += 1
    a = cf.get_npimage()
    return a[:, :, 0], cnt_circles, cnt_boxes


def gen_item(label, test=False):
    if label == 0:
        a = get_circle()
    else:
        a = get_box()
    return a


def merge_samples(X, Y, cnt_sampes=10):
    if len(X.shape) == 3:
        bar = np.ones([X.shape[1], 2])
        im_ar = np.hstack([np.hstack([X[i], bar]) for i in range(cnt_sampes)])
    else:
        bar = np.ones([1, X.shape[2], 2])
        im_ar = np.dstack([np.dstack([X[i], bar]) for i in range(cnt_sampes)])
        im_ar = np.rollaxis(im_ar, 0, 3)
        if im_ar.shape[-1] == 1:
            im_ar = im_ar[:, :, 0]

    im_ar *= 255
    im = PIL.Image.fromarray(im_ar)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    return im


def get_ds_naive(dim_image=64, cnt_samples=100):
    data = []
    labels = []

    for i in range(cnt_samples):
        cf.init(canvas_size=(dim_image, dim_image), face_color="#FFFFFF")
        if i % 2:
            cf.circle(0.3)
            labels.append(0)
        else:
            cf.box(0.5)
            labels.append(1)
        a = cf.get_npimage()
        data.append(a[:, :, 0])
    X = np.array(data)
    Y = np.array(labels, dtype=np.int32)
    return X, Y


def get_ds_simple(dim_image=64, cnt_samples=100):
    data = []
    labels = []
    for i in range(cnt_samples):
        if i % 2:
            data.append(get_box(dim_image=dim_image))
            labels.append(0)
        else:
            data.append(get_circle(dim_image=dim_image))
            labels.append(1)

    X = np.array(data)
    Y = np.array(labels, dtype=np.int32)
    return X, Y


def get_ds_counting(cnt_samples=100):
    data = []
    cnts = []
    for i in range(cnt_samples):
        a, c, b = get_counting_sample()
        data.append(a)
        cnts.append([c, b])
    X = np.array(data)
    Y = np.array(cnts, dtype=np.int32)
    return X, Y


def main():
    print("generating sample output")
    cnt_samples = 10
    X_train = np.array([gen_item(i % 2) for i in range(cnt_samples)])
    Y_train = np.array([i % 2 for i in range(cnt_samples)], dtype=np.int32)
    print(X_train.shape, Y_train.shape)
    im = merge_samples(X_train, Y_train)
    im.save("/tmp/test.png")

if __name__ == "__main__":
    main()
