"""
Script to test that the caculated force and forcegradient match with the numerical derivatives of the energy and force
respectively.
"""

import numpy as np
import sys
sys.path.append('../..')
from caspwn.plane_sphere.compute_plsp import system as plsp_system
from caspwn.sphere_sphere.compute_spsp import system as spsp_system
from caspwn.materials import PEC, vacuum, silica, polystyrene, water_zwol

# PLANE-SPHERE
def test_plsp():
    mat_sphere = [PEC, polystyrene, PEC, polystyrene]
    mat_plane = [PEC, silica, PEC, silica]
    mat_medium = [vacuum, water_zwol, vacuum, water_zwol]
    R = [50.e-6, 50.e-6, 150.e-6, 150.e-6]
    L = [5.e-6, 5.e-6, 1.e-6, 1.e-6]
    h = [5.e-09, 5.e-09, 1.e-09, 1.e-09]

    for i in range(4):
        sys_lo = plsp_system(300., L[i]-h[i], R[i], mat_sphere[i], mat_plane[i], mat_medium[i])
        sys_lo.calculate('forcegradient')
        lo = sys_lo.result
        sys_mi = plsp_system(300., L[i], R[i], mat_sphere[i], mat_plane[i], mat_medium[i])
        sys_mi.calculate('forcegradient')
        mi = sys_mi.result
        sys_hi = plsp_system(300., L[i]+h[i], R[i], mat_sphere[i], mat_plane[i], mat_medium[i])
        sys_hi.calculate('forcegradient')
        hi = sys_hi.result
        num_force = -(hi[0]-lo[0])/2/h[i]
        np.testing.assert_allclose(num_force, mi[1], rtol=7.e-6)
        num_forcegradient = (hi[1] - lo[1]) / 2 / h[i]
        np.testing.assert_allclose(num_forcegradient, mi[2], rtol=8.e-4)


# SPHERE-SPHERE
def test_spsp():
    mat_sphere1 = [PEC, polystyrene, PEC, polystyrene]
    mat_sphere2 = [PEC, polystyrene, PEC, polystyrene]
    mat_medium = [vacuum, water_zwol, vacuum, water_zwol]
    R1 = [50.e-6, 50.e-6, 150.e-6, 150.e-6]
    R2 = [50.e-6, 50.e-6, 50.e-6, 50.e-6]
    L = [5.e-6, 5.e-6, 1.e-6, 1.e-6]
    h = [5.e-09, 5.e-09, 1.e-09, 1.e-09]

    for i in range(4):
        sys_lo = spsp_system(300., L[i]-h[i], R1[i], R2[i], mat_sphere1[i], mat_sphere2[i], mat_medium[i])
        sys_lo.calculate('forcegradient')
        lo = sys_lo.result
        sys_mi = spsp_system(300., L[i], R1[i], R2[i], mat_sphere1[i], mat_sphere2[i], mat_medium[i])
        sys_mi.calculate('forcegradient')
        mi = sys_mi.result
        sys_hi = spsp_system(300., L[i]+h[i], R1[i], R2[i], mat_sphere1[i], mat_sphere2[i], mat_medium[i])
        sys_hi.calculate('forcegradient')
        hi = sys_hi.result
        num_force = -(hi[0]-lo[0])/2/h[i]
        np.testing.assert_allclose(num_force, mi[1], rtol=7.e-6)
        num_forcegradient = (hi[1] - lo[1]) / 2 / h[i]
        np.testing.assert_allclose(num_forcegradient, mi[2], rtol=8.e-4)