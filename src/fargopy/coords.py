###############################################################
# FARGOpy interdependencies
###############################################################
import fargopy

###############################################################
# Package documentation
###############################################################
"""
FARGOpy Coordinate Transformation Module
=========================================

This module provides functions for converting between different coordinate systems:
- Cartesian (x, y, z)
- Cylindrical (r, phi, z)
- Spherical (r, phi, theta)

Both position coordinates and velocity vectors can be transformed.
All functions support both 2D and 3D conversions.
"""

###############################################################
# Required packages
###############################################################
import numpy as np

###############################################################
# Module exports
###############################################################
__all__ = [
    "transform_coords",
    "transform_velocity",
]

###############################################################
# Internal conversion functions for coordinates
###############################################################

def _cartesian_to_cylindrical(x, y, z=None):
    """
    Convert Cartesian coordinates to cylindrical coordinates.
    
    Parameters
    ----------
    x : float or array_like
        x-coordinate(s)
    y : float or array_like
        y-coordinate(s)
    z : float or array_like, optional
        z-coordinate(s). If None, performs 2D conversion (returns r, phi only).
    
    Returns
    -------
    phi : float or array_like
        Azimuthal angle in radians (-π to π)
    r : float or array_like
        Radial distance from z-axis
    z : float or array_like (only if z is provided)
        Height coordinate (unchanged)
    
    Notes
    -----
    Uses FARGO3D convention: (phi, r, z) order.
    
    Examples
    --------
    2D conversion:
    
    >>> phi, r = fp.cartesian_to_cylindrical(1.0, 1.0)
    >>> print(f"phi={phi:.3f}, r={r:.3f}")
    phi=0.785, r=1.414
    
    3D conversion:
    
    >>> phi, r, z = fp.cartesian_to_cylindrical(1.0, 1.0, 2.0)
    >>> print(f"phi={phi:.3f}, r={r:.3f}, z={z:.3f}")
    phi=0.785, r=1.414, z=2.000
    
    Array conversion:
    
    >>> x = np.array([1, 0, -1])
    >>> y = np.array([0, 1, 0])
    >>> phi, r = fp.cartesian_to_cylindrical(x, y)
    """
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    
    if z is None:
        return phi, r
    else:
        return phi, r, z


def _cylindrical_to_cartesian(phi, r, z=None):
    """
    Convert cylindrical coordinates to Cartesian coordinates.
    
    Parameters
    ----------
    phi : float or array_like
        Azimuthal angle in radians
    r : float or array_like
        Radial distance from z-axis
    z : float or array_like, optional
        Height coordinate. If None, performs 2D conversion (returns x, y only).
    
    Returns
    -------
    x : float or array_like
        x-coordinate
    y : float or array_like
        y-coordinate
    z : float or array_like (only if z is provided)
        z-coordinate (unchanged)
    
    Examples
    --------
    2D conversion:
    
    >>> x, y = fp.cylindrical_to_cartesian(1.414, np.pi/4)
    >>> print(f"x={x:.3f}, y={y:.3f}")
    x=1.000, y=1.000
    
    3D conversion:
    
    >>> x, y, z = fp.cylindrical_to_cartesian(1.414, np.pi/4, 2.0)
    >>> print(f"x={x:.3f}, y={y:.3f}, z={z:.3f}")
    x=1.000, y=1.000, z=2.000
    """
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    
    if z is None:
        return x, y
    else:
        return x, y, z


def _cartesian_to_spherical(x, y, z):
    """
    Convert Cartesian coordinates to spherical coordinates.
    
    Parameters
    ----------
    x : float or array_like
        x-coordinate(s)
    y : float or array_like
        y-coordinate(s)
    z : float or array_like
        z-coordinate(s)
    
    Returns
    -------
    phi : float or array_like
        Azimuthal angle in radians (-π to π)
    r : float or array_like
        Radial distance from origin
    theta : float or array_like
        Polar angle from z-axis in radians (0 to π)
    
    Notes
    -----
    Uses FARGO3D convention: (phi, r, theta) order.
    
    Examples
    --------
    >>> phi, r, theta = fp.cartesian_to_spherical(1.0, 1.0, 1.0)
    >>> print(f"phi={phi:.3f}, r={r:.3f}, theta={theta:.3f}")
    phi=0.785, r=1.732, theta=0.955
    
    Array conversion:
    
    >>> x = np.array([1, 0, 0])
    >>> y = np.array([0, 1, 0])
    >>> z = np.array([0, 0, 1])
    >>> phi, r, theta = fp.cartesian_to_spherical(x, y, z)
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(y, x)
    theta = np.arccos(np.clip(z / (r + 1e-15), -1.0, 1.0))  # Add small epsilon to avoid division by zero
    
    return phi, r, theta


def _spherical_to_cartesian(phi, r, theta):
    """
    Convert spherical coordinates to Cartesian coordinates.
    
    Parameters
    ----------
    phi : float or array_like
        Azimuthal angle in radians
    r : float or array_like
        Radial distance from origin
    theta : float or array_like
        Polar angle from z-axis in radians (0 to π)
    
    Returns
    -------
    x : float or array_like
        x-coordinate
    y : float or array_like
        y-coordinate
    z : float or array_like
        z-coordinate
    
    Examples
    --------
    >>> x, y, z = fp.spherical_to_cartesian(1.732, np.pi/4, np.pi/4)
    >>> print(f"x={x:.3f}, y={y:.3f}, z={z:.3f}")
    x=0.866, y=0.866, z=1.225
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    return x, y, z


def _cylindrical_to_spherical(phi, r_cyl, z):
    """
    Convert cylindrical coordinates to spherical coordinates.
    
    Parameters
    ----------
    phi : float or array_like
        Azimuthal angle in radians
    r_cyl : float or array_like
        Radial distance from z-axis
    z : float or array_like
        Height coordinate
    
    Returns
    -------
    phi : float or array_like
        Azimuthal angle in radians (unchanged)
    r : float or array_like
        Radial distance from origin
    theta : float or array_like
        Polar angle from z-axis in radians (0 to π)
    
    Notes
    -----
    Uses FARGO3D convention: input (phi, r_cyl, z), output (phi, r, theta).
    
    Examples
    --------
    >>> phi_out, r, theta = fp.cylindrical_to_spherical(np.pi/4, 1.414, 1.0)
    >>> print(f"phi={phi_out:.3f}, r={r:.3f}, theta={theta:.3f}")
    phi=0.785, r=1.732, theta=0.955
    """
    r = np.sqrt(r_cyl**2 + z**2)
    theta = np.arctan2(r_cyl, z)
    
    return phi, r, theta


def _spherical_to_cylindrical(phi, r, theta):
    """
    Convert spherical coordinates to cylindrical coordinates.
    
    Parameters
    ----------
    phi : float or array_like
        Azimuthal angle in radians
    r : float or array_like
        Radial distance from origin
    theta : float or array_like
        Polar angle from z-axis in radians (0 to π)
    
    Returns
    -------
    phi : float or array_like
        Azimuthal angle in radians (unchanged)
    r_cyl : float or array_like
        Radial distance from z-axis
    z : float or array_like
        Height coordinate
    
    Notes
    -----
    Uses FARGO3D convention: input (phi, r, theta), output (phi, r_cyl, z).
    
    Examples
    --------
    >>> phi_out, r_cyl, z = fp.spherical_to_cylindrical(np.pi/4, 1.732, np.pi/4)
    >>> print(f"phi={phi_out:.3f}, r_cyl={r_cyl:.3f}, z={z:.3f}")
    phi=0.785, r_cyl=1.225, z=1.225
    """
    r_cyl = r * np.sin(theta)
    z = r * np.cos(theta)
    
    return phi, r_cyl, z


###############################################################
# Velocity vector conversions
###############################################################

def _velocity_cartesian_to_cylindrical(vx, vy, vz, x, y, z=None):
    """
    Convert velocity vector from Cartesian to cylindrical coordinates.
    
    The transformation depends on the position where the velocity is defined.
    
    Parameters
    ----------
    vx : float or array_like
        Velocity component in x-direction
    vy : float or array_like
        Velocity component in y-direction
    vz : float or array_like
        Velocity component in z-direction. Can be None for 2D conversion.
    x : float or array_like
        x-coordinate of position
    y : float or array_like
        y-coordinate of position
    z : float or array_like, optional
        z-coordinate of position. Not used in transformation but kept for consistency.
    
    Returns
    -------
    vphi : float or array_like
        Azimuthal velocity component
    vr : float or array_like
        Radial velocity component
    vz : float or array_like (only if vz was provided)
        Vertical velocity component (unchanged)
    
    Notes
    -----
    Uses FARGO3D convention: output (vphi, vr, vz).
    
    Examples
    --------
    2D conversion:
    
    >>> vphi, vr = fp.velocity_cartesian_to_cylindrical(1.0, 0.0, None, 1.0, 0.0)
    >>> print(f"vphi={vphi:.3f}, vr={vr:.3f}")
    vphi=0.000, vr=1.000
    
    3D conversion:
    
    >>> vphi, vr, vz = fp.velocity_cartesian_to_cylindrical(1.0, 0.0, 0.5, 1.0, 0.0, 1.0)
    >>> print(f"vphi={vphi:.3f}, vr={vr:.3f}, vz={vz:.3f}")
    vphi=0.000, vr=1.000, vz=0.500
    """
    r = np.sqrt(x**2 + y**2)
    cos_phi = x / (r + 1e-15)
    sin_phi = y / (r + 1e-15)
    
    vr = vx * cos_phi + vy * sin_phi
    vphi = -vx * sin_phi + vy * cos_phi
    
    if vz is None:
        return vphi, vr
    else:
        return vphi, vr, vz


def _velocity_cylindrical_to_cartesian(vphi, vr, vz, phi, r_cyl, z=None):
    """
    Convert velocity vector from cylindrical to Cartesian coordinates.
    
    The transformation depends on the position where the velocity is defined.
    
    Parameters
    ----------
    vphi : float or array_like
        Azimuthal velocity component
    vr : float or array_like
        Radial velocity component
    vz : float or array_like
        Vertical velocity component. Can be None for 2D conversion.
    phi : float or array_like
        Azimuthal angle in radians
    r_cyl : float or array_like
        Radial distance from z-axis
    z : float or array_like, optional
        Height coordinate. Not used in transformation but kept for consistency.
    
    Returns
    -------
    vx : float or array_like
        Velocity component in x-direction
    vy : float or array_like
        Velocity component in y-direction
    vz : float or array_like (only if vz was provided)
        Velocity component in z-direction (unchanged)
    
    Notes
    -----
    Uses FARGO3D convention: input (vphi, vr, vz) and (phi, r_cyl, z).
    
    Examples
    --------
    2D conversion:
    
    >>> vx, vy = fp.velocity_cylindrical_to_cartesian(0.0, 1.0, None, 0.0, 1.0)
    >>> print(f"vx={vx:.3f}, vy={vy:.3f}")
    vx=1.000, vy=0.000
    
    3D conversion:
    
    >>> vx, vy, vz = fp.velocity_cylindrical_to_cartesian(0.0, 1.0, 0.5, 0.0, 1.0, 1.0)
    >>> print(f"vx={vx:.3f}, vy={vy:.3f}, vz={vz:.3f}")
    vx=1.000, vy=0.000, vz=0.500
    """
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    
    vx = vr * cos_phi - vphi * sin_phi
    vy = vr * sin_phi + vphi * cos_phi
    
    if vz is None:
        return vx, vy
    else:
        return vx, vy, vz


def _velocity_cartesian_to_spherical(vx, vy, vz, x, y, z):
    """
    Convert velocity vector from Cartesian to spherical coordinates.
    
    The transformation depends on the position where the velocity is defined.
    
    Parameters
    ----------
    vx : float or array_like
        Velocity component in x-direction
    vy : float or array_like
        Velocity component in y-direction
    vz : float or array_like
        Velocity component in z-direction
    x : float or array_like
        x-coordinate of position
    y : float or array_like
        y-coordinate of position
    z : float or array_like
        z-coordinate of position
    
    Returns
    -------
    vphi : float or array_like
        Azimuthal velocity component
    vr : float or array_like
        Radial velocity component
    vtheta : float or array_like
        Polar velocity component
    
    Notes
    -----
    Uses FARGO3D convention: output (vphi, vr, vtheta).
    
    Examples
    --------
    >>> vphi, vr, vtheta = fp.velocity_cartesian_to_spherical(1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    >>> print(f"vphi={vphi:.3f}, vr={vr:.3f}, vtheta={vtheta:.3f}")
    vphi=0.000, vr=1.000, vtheta=0.000
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    rho = np.sqrt(x**2 + y**2)
    
    # Add small epsilon to avoid division by zero
    r_safe = r + 1e-15
    rho_safe = rho + 1e-15
    
    sin_theta = rho / r_safe
    cos_theta = z / r_safe
    sin_phi = y / rho_safe
    cos_phi = x / rho_safe
    
    vr = vx * sin_theta * cos_phi + vy * sin_theta * sin_phi + vz * cos_theta
    vtheta = vx * cos_theta * cos_phi + vy * cos_theta * sin_phi - vz * sin_theta
    vphi = -vx * sin_phi + vy * cos_phi
    
    return vphi, vr, vtheta


def _velocity_spherical_to_cartesian(vphi, vr, vtheta, phi, r, theta):
    """
    Convert velocity vector from spherical to Cartesian coordinates.
    
    The transformation depends on the position where the velocity is defined.
    
    Parameters
    ----------
    vphi : float or array_like
        Azimuthal velocity component
    vr : float or array_like
        Radial velocity component
    vtheta : float or array_like
        Polar velocity component
    phi : float or array_like
        Azimuthal angle in radians
    r : float or array_like
        Radial distance from origin
    theta : float or array_like
        Polar angle from z-axis in radians (0 to π)
    
    Returns
    -------
    vx : float or array_like
        Velocity component in x-direction
    vy : float or array_like
        Velocity component in y-direction
    vz : float or array_like
        Velocity component in z-direction
    
    Notes
    -----
    Uses FARGO3D convention: input (vphi, vr, vtheta) and (phi, r, theta).
    
    Examples
    --------
    >>> vx, vy, vz = fp.velocity_spherical_to_cartesian(0.0, 1.0, 0.0, 0.0, 1.0, np.pi/2)
    >>> print(f"vx={vx:.3f}, vy={vy:.3f}, vz={vz:.3f}")
    vx=1.000, vy=0.000, vz=0.000
    """
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    
    vx = vr * sin_theta * cos_phi + vtheta * cos_theta * cos_phi - vphi * sin_phi
    vy = vr * sin_theta * sin_phi + vtheta * cos_theta * sin_phi + vphi * cos_phi
    vz = vr * cos_theta - vtheta * sin_theta
    
    return vx, vy, vz


def _velocity_cylindrical_to_spherical(vphi, vr_cyl, vz_cyl, phi, r_cyl, z):
    """
    Convert velocity vector from cylindrical to spherical coordinates.
    
    The transformation depends on the position where the velocity is defined.
    
    Parameters
    ----------
    vphi : float or array_like
        Azimuthal velocity component
    vr_cyl : float or array_like
        Radial velocity component (cylindrical)
    vz_cyl : float or array_like
        Vertical velocity component
    phi : float or array_like
        Azimuthal angle in radians
    r_cyl : float or array_like
        Radial distance from z-axis
    z : float or array_like
        Height coordinate
    
    Returns
    -------
    vphi : float or array_like
        Azimuthal velocity component (unchanged)
    vr : float or array_like
        Radial velocity component (spherical)
    vtheta : float or array_like
        Polar velocity component
    
    Notes
    -----
    Uses FARGO3D convention: input (vphi, vr_cyl, vz_cyl) with (phi, r_cyl, z),
    output (vphi, vr, vtheta).
    
    Examples
    --------
    >>> vphi_out, vr, vtheta = fp.velocity_cylindrical_to_spherical(0.0, 1.0, 0.0, 0.0, 1.0, 1.0)
    >>> print(f"vphi={vphi_out:.3f}, vr={vr:.3f}, vtheta={vtheta:.3f}")
    vphi=0.000, vr=0.707, vtheta=0.707
    """
    r = np.sqrt(r_cyl**2 + z**2)
    
    # Add small epsilon to avoid division by zero
    r_safe = r + 1e-15
    
    sin_theta = r_cyl / r_safe
    cos_theta = z / r_safe
    
    vr = vr_cyl * sin_theta + vz_cyl * cos_theta
    vtheta = vr_cyl * cos_theta - vz_cyl * sin_theta
    
    return vphi, vr, vtheta


def _velocity_spherical_to_cylindrical(vphi, vr, vtheta, phi, r, theta):
    """
    Convert velocity vector from spherical to cylindrical coordinates.
    
    The transformation depends on the position where the velocity is defined.
    
    Parameters
    ----------
    vphi : float or array_like
        Azimuthal velocity component
    vr : float or array_like
        Radial velocity component (spherical)
    vtheta : float or array_like
        Polar velocity component
    phi : float or array_like
        Azimuthal angle in radians
    r : float or array_like
        Radial distance from origin
    theta : float or array_like
        Polar angle from z-axis in radians (0 to π)
    
    Returns
    -------
    vphi : float or array_like
        Azimuthal velocity component (unchanged)
    vr_cyl : float or array_like
        Radial velocity component (cylindrical)
    vz : float or array_like
        Vertical velocity component
    
    Notes
    -----
    Uses FARGO3D convention: input (vphi, vr, vtheta) with (phi, r, theta),
    output (vphi, vr_cyl, vz).
    
    Examples
    --------
    >>> vphi_out, vr_cyl, vz = fp.velocity_spherical_to_cylindrical(0.0, 1.0, 0.0, 0.0, 1.0, np.pi/4)
    >>> print(f"vphi={vphi_out:.3f}, vr_cyl={vr_cyl:.3f}, vz={vz:.3f}")
    vphi=0.000, vr_cyl=0.707, vz=0.707
    """
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    vr_cyl = vr * sin_theta + vtheta * cos_theta
    vz = vr * cos_theta - vtheta * sin_theta
    
    return vphi, vr_cyl, vz


###############################################################
# Main public interface functions
###############################################################

def transform_coords(coord_in, coord_out, *coords):
    """
    Transform position coordinates between different coordinate systems.
    
    Supported coordinate systems (FARGO3D convention):
    - 'cartesian': (x, y) or (x, y, z)
    - 'cylindrical': (phi, r) or (phi, r, z)
    - 'spherical': (phi, r, theta)
    
    Parameters
    ----------
    coord_in : str
        Input coordinate system: 'cartesian', 'cylindrical', or 'spherical'
    coord_out : str
        Output coordinate system: 'cartesian', 'cylindrical', or 'spherical'
    *coords : float or array_like
        Coordinate values in the input system.
        - For 2D conversions: provide 2 values (not available for spherical)
        - For 3D conversions: provide 3 values
    
    Returns
    -------
    tuple
        Coordinates in the output system (2 or 3 values depending on input)
    
    Examples
    --------
    Cartesian to cylindrical (2D):
    
    >>> phi, r = fp.transform_coords('cartesian', 'cylindrical', 1.0, 1.0)
    >>> print(f"phi={phi:.3f}, r={r:.3f}")
    phi=0.785, r=1.414
    
    Cartesian to cylindrical (3D):
    
    >>> phi, r, z = fp.transform_coords('cartesian', 'cylindrical', 1.0, 1.0, 2.0)
    >>> print(f"phi={phi:.3f}, r={r:.3f}, z={z:.3f}")
    phi=0.785, r=1.414, z=2.000
    
    Cylindrical to spherical:
    
    >>> phi, r, theta = fp.transform_coords('cylindrical', 'spherical', np.pi/4, 1.414, 1.0)
    >>> print(f"phi={phi:.3f}, r={r:.3f}, theta={theta:.3f}")
    phi=0.785, r=1.732, theta=0.955
    
    Spherical to cartesian:
    
    >>> x, y, z = fp.transform_coords('spherical', 'cartesian', np.pi/4, 1.732, np.pi/4)
    >>> print(f"x={x:.3f}, y={y:.3f}, z={z:.3f}")
    x=0.866, y=0.866, z=1.225
    
    Array conversion:
    
    >>> x = np.array([1, 0, -1])
    >>> y = np.array([0, 1, 0])
    >>> phi, r = fp.transform_coords('cartesian', 'cylindrical', x, y)
    
    Notes
    -----
    - Uses FARGO3D convention: phi (azimuthal angle) is the first coordinate 
      for cylindrical and spherical systems, followed by r (radial distance)
    - Angles are in radians
    - Phi is azimuthal angle (-π to π)
    - Theta is polar angle from z-axis (0 to π)
    - 2D conversions are only available between cartesian and cylindrical
    - Spherical coordinates always require 3 values
    """
    # Normalize coordinate system names
    coord_in = coord_in.lower()
    coord_out = coord_out.lower()
    
    # Validate coordinate system names
    valid_systems = ['cartesian', 'cylindrical', 'spherical']
    if coord_in not in valid_systems:
        raise ValueError(f"Invalid input coordinate system '{coord_in}'. Must be one of {valid_systems}")
    if coord_out not in valid_systems:
        raise ValueError(f"Invalid output coordinate system '{coord_out}'. Must be one of {valid_systems}")
    
    # If same coordinate system, just return the input
    if coord_in == coord_out:
        return coords
    
    # Mapping of conversion functions
    # Key format: (coord_in, coord_out, num_coords)
    conversion_map = {
        ('cartesian', 'cylindrical', 2): _cartesian_to_cylindrical,
        ('cartesian', 'cylindrical', 3): _cartesian_to_cylindrical,
        ('cylindrical', 'cartesian', 2): _cylindrical_to_cartesian,
        ('cylindrical', 'cartesian', 3): _cylindrical_to_cartesian,
        ('cartesian', 'spherical', 3): _cartesian_to_spherical,
        ('spherical', 'cartesian', 3): _spherical_to_cartesian,
        ('cylindrical', 'spherical', 3): _cylindrical_to_spherical,
        ('spherical', 'cylindrical', 3): _spherical_to_cylindrical,
    }
    
    # Validate number of coordinates
    num_coords = len(coords)
    if num_coords < 2 or num_coords > 3:
        raise ValueError(f"Expected 2 or 3 coordinate values, got {num_coords}")
    
    # Check for invalid combinations
    if coord_in == 'spherical' and num_coords != 3:
        raise ValueError("Spherical coordinates require exactly 3 values (r, phi, theta)")
    if coord_out == 'spherical' and num_coords != 3:
        raise ValueError("Cannot convert to spherical coordinates from 2D input")
    
    # Get the appropriate conversion function
    key = (coord_in, coord_out, num_coords)
    if key not in conversion_map:
        raise ValueError(f"Conversion from {coord_in} to {coord_out} with {num_coords} coordinates is not supported")
    
    conversion_func = conversion_map[key]
    return conversion_func(*coords)


def transform_velocity(coord_in, coord_out, velocity, position):
    """
    Transform velocity vectors between different coordinate systems.
    
    Velocity transformations are position-dependent, so both velocity components
    and position coordinates must be provided.
    
    Supported coordinate systems (FARGO3D convention):
    - 'cartesian': velocity (vx, vy, vz), position (x, y, z)
    - 'cylindrical': velocity (vphi, vr, vz), position (phi, r, z)
    - 'spherical': velocity (vphi, vr, vtheta), position (phi, r, theta)
    
    Parameters
    ----------
    coord_in : str
        Input coordinate system: 'cartesian', 'cylindrical', or 'spherical'
    coord_out : str
        Output coordinate system: 'cartesian', 'cylindrical', or 'spherical'
    velocity : tuple or list
        Velocity components in the input system (2 or 3 values).
        For 2D: provide (v1, v2, None) or just (v1, v2)
    position : tuple or list
        Position coordinates in the input system (2 or 3 values).
        Required for the transformation matrix calculation.
    
    Returns
    -------
    tuple
        Velocity components in the output system (2 or 3 values)
    
    Examples
    --------
    Cartesian to cylindrical (2D):
    
    >>> vphi, vr = fp.transform_velocity('cartesian', 'cylindrical', 
    ...                                   (1.0, 0.0), (1.0, 0.0))
    >>> print(f"vphi={vphi:.3f}, vr={vr:.3f}")
    vphi=0.000, vr=1.000
    
    Cartesian to cylindrical (3D):
    
    >>> vphi, vr, vz = fp.transform_velocity('cartesian', 'cylindrical',
    ...                                       (1.0, 0.0, 0.5), (1.0, 0.0, 1.0))
    >>> print(f"vphi={vphi:.3f}, vr={vr:.3f}, vz={vz:.3f}")
    vphi=0.000, vr=1.000, vz=0.500
    
    Cylindrical to spherical:
    
    >>> vphi, vr, vtheta = fp.transform_velocity('cylindrical', 'spherical',
    ...                                           (0.0, 1.0, 0.0), (0.0, 1.0, 1.0))
    >>> print(f"vphi={vphi:.3f}, vr={vr:.3f}, vtheta={vtheta:.3f}")
    vphi=0.000, vr=0.707, vtheta=0.707
    
    Spherical to cartesian:
    
    >>> vx, vy, vz = fp.transform_velocity('spherical', 'cartesian',
    ...                                     (0.0, 1.0, 0.0), (0.0, 1.0, np.pi/2))
    >>> print(f"vx={vx:.3f}, vy={vy:.3f}, vz={vz:.3f}")
    vx=1.000, vy=0.000, vz=0.000
    
    Array conversion:
    
    >>> vx = np.array([1, 0, -1])
    >>> vy = np.array([0, 1, 0])
    >>> x = np.array([1, 0, -1])
    >>> y = np.array([0, 1, 0])
    >>> vphi, vr = fp.transform_velocity('cartesian', 'cylindrical', 
    ...                                   (vx, vy), (x, y))
    
    Notes
    -----
    - Uses FARGO3D convention: vphi (azimuthal velocity) is the first component 
      for cylindrical and spherical systems
    - Velocity transformations depend on the position where the velocity is defined
    - For 2D conversions, provide 2 velocity components and 2 position coordinates
    - For 3D conversions, provide 3 velocity components and 3 position coordinates
    - Angles in position must be in radians
    - 2D conversions are only available between cartesian and cylindrical
    """
    # Normalize coordinate system names
    coord_in = coord_in.lower()
    coord_out = coord_out.lower()
    
    # Validate coordinate system names
    valid_systems = ['cartesian', 'cylindrical', 'spherical']
    if coord_in not in valid_systems:
        raise ValueError(f"Invalid input coordinate system '{coord_in}'. Must be one of {valid_systems}")
    if coord_out not in valid_systems:
        raise ValueError(f"Invalid output coordinate system '{coord_out}'. Must be one of {valid_systems}")
    
    # If same coordinate system, just return the input
    if coord_in == coord_out:
        return velocity if isinstance(velocity, tuple) else tuple(velocity)
    
    # Convert velocity and position to lists for easier handling
    vel = list(velocity) if not isinstance(velocity, (list, tuple)) else list(velocity)
    pos = list(position) if not isinstance(position, (list, tuple)) else list(position)
    
    # Handle 2D case by adding None for z components if needed
    if len(vel) == 2:
        vel.append(None)
    if len(pos) == 2:
        pos.append(None)
    
    # Validate dimensions
    if len(vel) != 3 or len(pos) != 3:
        raise ValueError(f"Expected 2 or 3 components for velocity and position")
    
    # Check for spherical constraints
    if coord_in == 'spherical' and vel[2] is None:
        raise ValueError("Spherical velocities require 3 components (vr, vphi, vtheta)")
    if coord_out == 'spherical' and vel[2] is None:
        raise ValueError("Cannot convert to spherical velocity from 2D input")
    
    # Mapping of velocity conversion functions
    conversion_map = {
        ('cartesian', 'cylindrical'): _velocity_cartesian_to_cylindrical,
        ('cylindrical', 'cartesian'): _velocity_cylindrical_to_cartesian,
        ('cartesian', 'spherical'): _velocity_cartesian_to_spherical,
        ('spherical', 'cartesian'): _velocity_spherical_to_cartesian,
        ('cylindrical', 'spherical'): _velocity_cylindrical_to_spherical,
        ('spherical', 'cylindrical'): _velocity_spherical_to_cylindrical,
    }
    
    # Get the appropriate conversion function
    key = (coord_in, coord_out)
    if key not in conversion_map:
        raise ValueError(f"Velocity conversion from {coord_in} to {coord_out} is not supported")
    
    conversion_func = conversion_map[key]
    
    # Call the conversion function with velocity and position
    result = conversion_func(*vel, *pos)
    
    return result
