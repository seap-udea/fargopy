"""
Test script to verify mass_flux calculations with different coordinate systems.
The results should be identical regardless of coords='cartesian' or coords='spherical'.
"""
import numpy as np
import fargopy as fp
import time

# Load simulation
PATH = fp.Simulation.download_precomputed('p3disoj')
sim = fp.Simulation(output_dir=PATH)
sim.units('CGS')
sim.set_units(UL=1*sim.AU, UM=1*sim.MSUN)

# Get planet info
SNAP = 10
jupiter = sim.load_planets(snapshot=SNAP)[0]
r_hill = jupiter.hill_radius
xp, yp, zp = jupiter.pos.x, jupiter.pos.y, jupiter.pos.z

# Test 1: Surface with coords='spherical'
print("=" * 70)
print("TEST 1: Surface with coords='spherical'")
print("=" * 70)

sphere_sph = fp.Surface(
    type='sphere',
    radius=0.5*r_hill,
    subdivisions=4,  # Smaller for faster test
    center=(xp, yp, zp),
    coords='spherical'
)

print(f"Number of centers: {len(sphere_sph.centers)}")
print(f"Centers coords system: {sphere_sph.coords}")
print(f"First 3 centers (spherical):\n{sphere_sph.centers[:3]}")
print()

# Calculate mass flux with coords='spherical' (should use RegularGridInterpolator)
t_start = time.time()
flux_sph_sph = sphere_sph.mass_flux(
    sim,
    snapshot=[SNAP, SNAP],
    coords='spherical',
    interpolator='regular_grid',
    follow_planet=False
)
t_sph_sph = time.time() - t_start

print(f"Mass flux (spherical Surface, coords='spherical'): {flux_sph_sph[0]:.6e}")
print(f"Time: {t_sph_sph:.3f} seconds")
print()

# Test 2: Surface with coords='cartesian'
print("=" * 70)
print("TEST 2: Surface with coords='cartesian'")
print("=" * 70)

sphere_cart = fp.Surface(
    type='sphere',
    radius=0.5*r_hill,
    subdivisions=4,
    center=(xp, yp, zp),
    coords='cartesian'
)

print(f"Number of centers: {len(sphere_cart.centers)}")
print(f"Centers coords system: {sphere_cart.coords}")
print(f"First 3 centers (cartesian):\n{sphere_cart.centers[:3]}")
print()

# Calculate mass flux with coords='spherical' (should use RegularGridInterpolator)
t_start = time.time()
flux_cart_sph = sphere_cart.mass_flux(
    sim,
    snapshot=[SNAP, SNAP],
    coords='spherical',
    interpolator='regular_grid',
    follow_planet=False
)
t_cart_sph = time.time() - t_start

print(f"Mass flux (cartesian Surface, coords='spherical'): {flux_cart_sph[0]:.6e}")
print(f"Time: {t_cart_sph:.3f} seconds")
print()

# Calculate mass flux with coords='cartesian' (will use griddata - slower)
t_start = time.time()
flux_cart_cart = sphere_cart.mass_flux(
    sim,
    snapshot=[SNAP, SNAP],
    coords='cartesian',
    interpolator='auto',  # Will use griddata for cartesian in spherical sim
    follow_planet=False
)
t_cart_cart = time.time() - t_start

print(f"Mass flux (cartesian Surface, coords='cartesian'): {flux_cart_cart[0]:.6e}")
print(f"Time: {t_cart_cart:.3f} seconds")
print()

# Test 3: Verify spherical Surface with coords='cartesian' method
print("=" * 70)
print("TEST 3: Spherical Surface with coords='cartesian' in method")
print("=" * 70)

t_start = time.time()
flux_sph_cart = sphere_sph.mass_flux(
    sim,
    snapshot=[SNAP, SNAP],
    coords='cartesian',
    interpolator='auto',
    follow_planet=False
)
t_sph_cart = time.time() - t_start

print(f"Mass flux (spherical Surface, coords='cartesian'): {flux_sph_cart[0]:.6e}")
print(f"Time: {t_sph_cart:.3f} seconds")
print()

# Summary
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"1. sphere(sph) + coords='spherical': {flux_sph_sph[0]:.6e}  [{t_sph_sph:.3f}s]")
print(f"2. sphere(cart) + coords='spherical': {flux_cart_sph[0]:.6e}  [{t_cart_sph:.3f}s]")
print(f"3. sphere(cart) + coords='cartesian': {flux_cart_cart[0]:.6e}  [{t_cart_cart:.3f}s]")
print(f"4. sphere(sph) + coords='cartesian':  {flux_sph_cart[0]:.6e}  [{t_sph_cart:.3f}s]")
print()

# Check consistency
print("CONSISTENCY CHECKS:")
print(f"All results match within 1%: ", end='')
all_fluxes = [flux_sph_sph[0], flux_cart_sph[0], flux_cart_cart[0], flux_sph_cart[0]]
mean_flux = np.mean(all_fluxes)
relative_diffs = [abs(f - mean_flux) / mean_flux * 100 for f in all_fluxes]
max_diff = max(relative_diffs)
print(f"{max_diff:.2f}% max difference")

if max_diff < 1.0:
    print("✅ PASS: All methods give consistent results")
else:
    print(f"❌ FAIL: Results differ by more than 1%")
    print("\nDetailed differences from mean:")
    for i, (f, d) in enumerate(zip(all_fluxes, relative_diffs), 1):
        print(f"  Test {i}: {d:+.3f}%")

print()
print("PERFORMANCE:")
fastest = min(t_sph_sph, t_cart_sph, t_cart_cart, t_sph_cart)
print(f"  RegularGrid speedup: {t_cart_cart/fastest:.1f}x faster than griddata")
