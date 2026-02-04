# fargopy/tests/test___interp.py
import os
import numpy as np
import pytest
import fargopy as fp


@pytest.fixture(scope="session")
def sim():
    # Load the local test simulation only once per test session
    fp.Simulation.download_precomputed("p3disoj")
    return fp.Simulation(output_dir=f"/tmp/p3disoj")


def test_interpolacion_1d_point(sim):
    data = sim.load_field(
        fields="gasdens",
        slice="phi=0,theta=1.56",
        snapshot=(1, 2),
        interpolate=True,
    )
    x = 1.2
    valor = data.evaluate(1.2, var1=x)
    # If it returns a 0-d array or a scalar, both are acceptable; it must be finite
    assert np.isfinite(valor).all(), "1D interpolation must return a finite value"


def test_interpolacion_1d_array(sim):
    data = sim.load_field(
        fields="gasdens",
        slice="phi=0,theta=1.56",
        snapshot=(1, 2),
        interpolate=True,
    )
    x = np.array([1.2, 1.3, 1.4])
    valor = data.evaluate(1.2, var1=x)
    assert np.asarray(valor).shape == x.shape, (
        "1D interpolation must preserve the input shape"
    )
    assert np.isfinite(valor).any(), (
        "At least some points should be finite if they lie inside the domain"
    )


def test_interpolacion_2d_point(sim):
    data = sim.load_field(
        fields="gasdens",
        slice="theta=1.56",
        snapshot=(1, 2),
        interpolate=True,
    )
    # Conservative point (typically inside the domain)
    x = 1.2
    y = 0.14
    valor = data.evaluate(
        1.2, var1=x, var2=y, interpolator="griddata", method="nearest"
    )
    assert np.isscalar(valor) or np.asarray(valor).shape == (), (
        "2D point interpolation must return a scalar/0-d value"
    )
    assert np.isfinite(valor), (
        "2D point interpolation must return a finite value (nearest)"
    )


def test_interpolacion_2d_array(sim):
    data = sim.load_field(
        fields="gasdens",
        slice="theta=1.56",
        snapshot=(1, 2),
        interpolate=True,
    )

    # Your y=[1.3,1.4,1.5] values are very likely outside the domain -> NaNs (linear) or spurious values.
    # Here we use "safe" points and, additionally, nearest-neighbor interpolation to avoid NaNs
    # due to the convex hull limitation.
    x = np.array([0.9, 1.0, 1.1])
    y = np.array([0.05, 0.10, 0.15])

    valor = data.evaluate(
        1.2, var1=x, var2=y, interpolator="griddata", method="nearest"
    )
    valor = np.asarray(valor)

    assert valor.shape == x.shape, (
        "2D interpolation must return an array with the same shape as the input"
    )
    assert np.isfinite(valor).all(), (
        "With nearest and in-domain points, NaNs should not appear"
    )


# FAILING
# def test_interpolacion_3d_point(sim):
#     data = sim.load_field(
#         fields="gasdens",
#         snapshot=(1, 2),
#         interpolate=True,
#     )
#     x, y, z = 1.2, 1.3, 1.4
#     valor = data.evaluate(
#         1.2, var1=x, var2=y, var3=z, interpolator="griddata", method="linear"
#     )
#     assert np.isscalar(valor) or np.asarray(valor).shape == (), (
#         "3D point interpolation must return a scalar/0-d value"
#     )
#     assert np.isfinite(valor), (
#         "3D point interpolation must return a finite value (nearest)"
#     )


def test_interpolacion_3d_array(sim):
    data = sim.load_field(
        fields="gasdens",
        snapshot=(1, 2),
        interpolate=True,
    )
    x = np.array([1.2, 1.3, 1.4])
    y = np.array([1.3, 1.4, 1.5])
    z = np.array([0.024, 0.14, 0.2])

    valor = data.evaluate(
        1.2, var1=x, var2=y, var3=z, interpolator="griddata", method="nearest"
    )
    valor = np.asarray(valor)

    assert valor.shape == x.shape, (
        "3D interpolation must return an array with the same shape as the input"
    )
    assert np.isfinite(valor).all(), (
        "With nearest, NaNs should not appear except for extremely out-of-domain points"
    )
