import skimage


def main():
    print("testing something")


def draw_box(y, x, r):
    rr, cc = skimage.draw.polygon_perimeter([y+r, y-r, y-r, y+r], [x-r, x-r, x+r, x+r])
    return rr, cc


if __name__ == "__main__":
    main()
