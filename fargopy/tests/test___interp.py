import numpy as np
import pytest
import fargopy as fp
from fargopy.fields import FieldInterpolator
import sys
import os

FILE=__file__
ROOTDIR=os.path.abspath(os.path.dirname(FILE))



@pytest.fixture(scope="session")
def sim():
    # Descarga y prepara la simulación real solo una vez por sesión de test
    return fp.Simulation(output_dir=f"{ROOTDIR}/data/p3disoj")

def test_interpolacion_1d_point(sim):
    data =sim.load_field(
        fields="gasdens",
        slice='phi=0,theta=1.56',
        snapshot=(1,2),
        interpolate=True
    )
    # Interpolación 1D: solo en x
    x = 1.2
    valor = data.evaluate(1.2, var1=x)
    assert not np.isnan(valor).any(), "La interpolación 1D debe devolver un escalar"

def test_interpolacion_1d_array(sim):
    data =sim.load_field(
        fields="gasdens",
        slice='phi=0,theta=1.56',
        snapshot=(1,2),
        interpolate=True
    )
    # Interpolación 1D: solo en x
    x = np.array([1.2, 1.3, 1.4])
    valor = data.evaluate(1.2, var1=x)
    assert valor.shape == x.shape, "La interpolación 1D debe devolver un arreglo del mismo tamaño que la entrada"


def test_interpolacion_2d_point(sim):
    data = sim.load_field(
        fields="gasdens",
        slice="theta=1.56",
        snapshot=(1,2),
        interpolate=True
    )
    # Interpolación 2D: x, y en el plano theta fijo
    x = 1.2
    y = 0.14
    valor = data.evaluate(1.2, var1=x, var2=y)
    assert np.isscalar(valor) and not np.isnan(valor), "La interpolación 2D debe devolver un escalar"

def test_interpolacion_2d_array(sim):
    data = sim.load_field(
        fields="gasdens",
        slice="theta=1.56",
        snapshot=(1,2),
        interpolate=True
    )
    # Interpolación 2D: x, y en el plano theta fijo
    x = np.array([1.2, 1.3, 1.4])
    y = np.array([1.3, 1.4, 1.5])
    valor = data.evaluate(1.2, var1=x, var2=y)
    assert valor.shape == x.shape and not np.isnan(valor).any(), "La interpolación 2D debe devolver un arreglo del mismo tamaño que la entrada"

def test_interpolacion_3d_point(sim):
    data = sim.load_field(
        fields="gasdens",
        snapshot=(1,2),
        interpolate=True
    )
    # Interpolación 3D: x, y, z
    x = 1.2
    y = 1.3
    z = 1.4
    valor = data.evaluate(1.2, var1=x, var2=y, var3=z, interpolator='griddata', method='nearest')
    assert np.isscalar(valor) and not np.isnan(valor), "La interpolación 3D debe devolver un escalar"

def test_interpolacion_3d_array(sim):
    data = sim.load_field(
        fields="gasdens",
        snapshot=(1,2),
        interpolate=True
    )
    # Interpolación 3D: x, y, z
    x = np.array([1.2, 1.3, 1.4])
    y = np.array([1.3, 1.4, 1.5])
    z = np.array([0.024,0.14 , 0.2])
    valor = data.evaluate(1.2, var1=x, var2=y, var3=z, interpolator='griddata', method='nearest')
    assert valor.shape == x.shape and not np.isnan(valor).any(), "La interpolación 3D debe devolver un arreglo del mismo tamaño que la entrada"

