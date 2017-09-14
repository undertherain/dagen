import numpy as np
import skimage
import skimage.draw
import random
import PIL
import PIL.Image
import contextfree


def draw_box(y, x, r):
    rr, cc = skimage.draw.polygon_perimeter([y+r, y-r, y-r, y+r], [x-r, x-r, x+r, x+r])
    return rr, cc


def gen_item(cl, test=False):
    dim_image = 64
    a = np.zeros([dim_image, dim_image], dtype=np.float32)
    radius = 12
    min_radius = 6
    if test:
        xc_random = int(random.random() * (dim_image // 2 - radius * 2 - 1) + radius + dim_image // 2 + 1)
    else:
        xc_random = int(random.random() * (dim_image // 2 - radius * 2 - 1) + radius)
    yc_random = int(random.random() * (dim_image - radius * 2) + radius)
    r = int(random.random() * (min_radius) + radius-min_radius)
    if cl == 0:
        rr, cc = skimage.draw.circle_perimeter(yc_random, xc_random, r)
        a[rr, cc] = 1
    else:
        rr, cc = draw_box(yc_random, xc_random, r)
        a[rr, cc] = 1
    return a


def merge_sample(X, Y):
    bar = np.ones([X.shape[1], 2])
    im_ar = np.hstack([np.hstack([X[i], bar]) for i in range(4)])
    im = PIL.Image.fromarray(im_ar * 255)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    return im


def main():
    print("generating sample output")
    cnt_samples = 10
    X_train = np.array([gen_item(i % 2) for i in range(cnt_samples)])
    Y_train = np.array([i % 2 for i in range(cnt_samples)], dtype=np.int32)
    print(X_train.shape, Y_train.shape)
    im = merge_sample(X_train, Y_train)
    im.save("/tmp/test.png")

if __name__ == "__main__":
    main()
