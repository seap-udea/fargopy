import os
import numpy as np
import pytest
import fargopy as fp

FILE = __file__
ROOTDIR = os.path.abspath(os.path.dirname(FILE))


@pytest.fixture(scope="session")
def sim():
    # Load the local test simulation only once per test session
    fp.Simulation.download_precomputed("p3disoj")
    return fp.Simulation(output_dir=f"/tmp/p3disoj")


def test_sphere_tessellation_properties():
    # basic sanity checks on sphere tessellation
    s = fp.flux.Surface(type="sphere", radius=1.0, subdivisions=1)
    # number of triangles for 1 subdivision: 20*(4**1) = 80
    assert s.num_triangles == 20 * (4**1)
    assert s.centers.shape == (s.num_triangles, 3)
    assert s.normals.shape == (s.num_triangles, 3)
    assert s.areas.shape == (s.num_triangles,)
    # areas positive
    assert np.all(s.areas > 0)
    # normals are unit length
    norms = np.linalg.norm(s.normals, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)
    # total area approximates sphere area 4*pi*r^2
    total_area = np.sum(s.areas)
    assert np.isfinite(total_area)
    assert np.isclose(total_area, 4 * np.pi * s.radius**2, rtol=0.12)


def test_plane_tessellation_and_areas():
    # plane tessellation with subdivisions should yield correct area sum
    s = fp.flux.Surface(type="plane", radius=1.0, subdivisions=4, width=2.0, length=3.0)
    assert s.centers.shape[0] == s.subdivisions**2
    assert np.all(s.areas > 0)
    total_area = np.sum(s.areas)
    assert np.isclose(total_area, 2.0 * 3.0, rtol=1e-8)


def test_cylinder_tessellation_basic():
    # cylinder tessellation exposes top/bottom/lateral arrays
    s = fp.flux.Surface(type="cylinder", radius=0.5, height=1.0, subdivisions=8)
    # ensure expected attributes exist and have consistent sizes
    assert hasattr(s, "top_centers")
    assert hasattr(s, "bottom_centers")
    assert hasattr(s, "lateral_centers")
    assert s.top_centers.shape[0] == s.bottom_centers.shape[0]
    assert s.lateral_centers.shape[0] > 0


# FAILED
def test_total_mass_grid_integration(sim):
    # Use the test simulation to run total_mass on a small sphere
    s = fp.flux.Surface(
        type="sphere", radius=0.1, subdivisions=0, center=(0.0, 0.0, 0.0)
    )
    # compute mass for a single snapshot (should be finite and non-negative)
    mass = s.total_mass(sim, field="gasdens", snapshot=1, follow_planet=False)
    assert np.isfinite(mass)
    assert mass >= 0.0


def test_total_mass_multiple_snapshots_returns_array(sim):
    s = fp.flux.Surface(
        type="sphere", radius=0.1, subdivisions=0, center=(0.0, 0.0, 0.0)
    )
    masses = s.total_mass(sim, field="gasdens", snapshot=[1, 2], follow_planet=False)
    assert isinstance(masses, np.ndarray)
    assert masses.shape[0] == 2
    assert np.all(np.isfinite(masses))
    assert np.all(masses >= 0.0)
