from types import SimpleNamespace

import numpy as np

from render_retarget_contact_comparison import _camera


def test_original_camera_does_not_depend_on_candidate():
    original = SimpleNamespace(xpos=np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]))
    first = SimpleNamespace(xpos=np.zeros((3, 3)))
    second = SimpleNamespace(xpos=np.ones((3, 3)) * 100)
    args = SimpleNamespace(camera_source='original', azimuth=145, elevation=-55, distance=1.25)
    camera_a = _camera(original, first, 0, {'left_hand': 1, 'right_hand': 2}, args)
    camera_b = _camera(original, second, 0, {'left_hand': 1, 'right_hand': 2}, args)
    np.testing.assert_array_equal(camera_a.lookat, camera_b.lookat)
    np.testing.assert_allclose(camera_a.lookat, .75 * original.xpos[0] + .25 * original.xpos[1:].mean(axis=0))


def test_shared_camera_default_remains_the_pair_average():
    original = SimpleNamespace(xpos=np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]))
    candidate = SimpleNamespace(xpos=np.zeros((3, 3)))
    args = SimpleNamespace(azimuth=145, elevation=-55, distance=1.25)
    camera = _camera(original, candidate, 0, {'left_hand': 1, 'right_hand': 2}, args)
    expected = .75 * original.xpos[0] / 2 + .25 * original.xpos[1:].mean(axis=0) / 2
    np.testing.assert_allclose(camera.lookat, expected)
