"""ProjectAirSim EarthUtils atmosphere, using altitude above sea level.

Mirrors core_sim/include/core_sim/earth_utils.hpp, not a fitted flight model.
Rotor coefficients use base density 1.225; the runtime multiplier divides
local density by EarthUtils::kSeaLevelAirDensity (1.224999).
"""
import math

ROTOR_BASE_DENSITY = 1.225
SEA_LEVEL_DENSITY = 1.224999
EARTH_RADIUS = 6378137.0
STANDARD_GRAVITY = 9.80665


def atmosphere(altitude_m):
    """Return (air density kg/m^3, gravity m/s^2), matching EarthUtils."""
    if not math.isfinite(altitude_m) or altitude_m <= -EARTH_RADIUS:
        raise ValueError("Altitude must be finite and above the Earth's center")
    altitude_km = altitude_m / 1000.0
    radius_km = EARTH_RADIUS / 1000.0
    h = radius_km * altitude_km / (radius_km + altitude_km)
    if h <= 11:
        temperature = 288.15 - 6.5 * h
        pressure = 101325.0 * (288.15 / temperature)**-5.255877
    elif h <= 20:
        temperature = 216.65
        pressure = 22632.06 * math.exp(-.1577 * (h - 11))
    elif h <= 32:
        temperature = 196.65 + h
        pressure = 5474.889 * (216.65 / temperature)**34.16319
    elif h <= 47:
        temperature = 228.65 + 2.8 * (h - 32)
        pressure = 868.0187 * (228.65 / temperature)**12.2011
    elif h <= 51:
        temperature = 270.65
        pressure = 110.9063 * math.exp(-.1262 * (h - 47))
    elif h <= 71:
        temperature = 270.65 - 2.8 * (h - 51)
        pressure = 66.93887 * (270.65 / temperature)**-12.2011
    elif h <= 84.85:
        temperature = 214.65 - 2 * (h - 71)
        pressure = 3.956420 * (214.65 / temperature)**-17.0816
    else:
        temperature, pressure = 3.0, .001
    density = pressure / 287.053 / temperature
    if -10000 < altitude_m < 10000:
        gravity = STANDARD_GRAVITY
    elif -100000 < altitude_m < 100000:
        gravity = STANDARD_GRAVITY * (1 - 2 * altitude_m / EARTH_RADIUS)
    else:
        gravity = STANDARD_GRAVITY / (1 + altitude_m / EARTH_RADIUS)**2
    return density, gravity
