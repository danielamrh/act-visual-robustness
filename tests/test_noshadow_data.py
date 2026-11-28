import numpy as np

from avr.noshadow_data import _apply, _homography, red_center


def test_red_center_finds_the_cube():
    img = np.full((48, 64, 3), 80, np.uint8)
    img[10:14, 20:26] = (220, 30, 30)
    (x, y), n = red_center(img)
    assert (x, y, n) == (22.5, 11.5, 24)
    assert red_center(np.full((8, 8, 3), 80, np.uint8))[1] == 0


def test_homography_roundtrip():
    h_true = np.array([[900.0, 20.0, 100.0], [10.0, -950.0, 700.0], [0.05, 0.1, 1.0]])
    world = [(x, y) for x in (0.0, 0.1, 0.2) for y in (0.4, 0.5, 0.6)]
    pix = [_apply(h_true, w) for w in world]
    h = _homography(world, pix)
    h_inv = np.linalg.inv(h)
    for w, p in zip(world, pix):
        assert np.allclose(_apply(h_inv, p), w, atol=1e-6)
