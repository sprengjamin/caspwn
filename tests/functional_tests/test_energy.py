"""
Script to test results for the energy against an older version of this software and also against CAPS.

The data files from the older version contains the git version which they stem from.
"""
import numpy as np
import os
from scipy.constants import hbar, c
#sys.path.append('../..')
from caspwn.plane_sphere.compute_plsp import system as plsp_system
from caspwn.sphere_sphere.compute_spsp import system as spsp_system
from caspwn.materials import PEC, vacuum, silica, polystyrene, water_zwol, gold_drude


# PLANE-SPHERE

# T=300, PR
def test_T300_PR_01():
    s = plsp_system(300., 1.e-06, 50.e-06, PEC, PEC, vacuum)
    res = s.calculate('energy')
    ref = np.loadtxt(os.path.join(os.path.dirname(__file__), 'reference_data/caspwn/data01.out'))[0]
    np.testing.assert_allclose(res, ref, rtol=1e-06, err_msg='Problem occured when comparing to data01.out')

def test_T300_PR_02():
    s = plsp_system(300., 0.25e-06, 50.e-06, PEC, PEC, vacuum)
    res = s.calculate('energy')
    ref = np.loadtxt(os.path.join(os.path.dirname(__file__), 'reference_data/caspwn/data02.out'))[0]
    np.testing.assert_allclose(res, ref, rtol=1e-06, err_msg='Problem occured when comparing to data02.out')

def test_T300_PR_03():
    s = plsp_system(300., 3.e-06, 150.e-06, PEC, PEC, vacuum)
    res = s.calculate('energy')
    ref = np.loadtxt(os.path.join(os.path.dirname(__file__), 'reference_data/caspwn/data03.out'))[0]
    np.testing.assert_allclose(res, ref, rtol=1e-06, err_msg='Problem occured when comparing to data03.out')

def test_T300_PR_04():
    s = plsp_system(300., 0.75e-06, 150.e-06, PEC, PEC, vacuum)
    res = s.calculate('energy')
    ref = np.loadtxt(os.path.join(os.path.dirname(__file__), 'reference_data/caspwn/data04.out'))[0]
    np.testing.assert_allclose(res, ref, rtol=1e-06, err_msg='Problem occured when comparing to data04.out')

# T=300, Gold sphere
def test_T300_gold_01():
    s = plsp_system(300., 1.e-06, 50.e-06, gold_drude, PEC, vacuum)
    res = s.calculate('energy')
    ref = np.loadtxt(os.path.join(os.path.dirname(__file__), 'reference_data/caspwn/data05.out'))[0]
    np.testing.assert_allclose(res, ref, rtol=1e-06, err_msg='Problem occured when comparing to data05.out')

def test_T300_gold_02():
    s = plsp_system(300., 0.25e-06, 50.e-06, gold_drude, PEC, vacuum)
    res = s.calculate('energy')
    ref = np.loadtxt(os.path.join(os.path.dirname(__file__), 'reference_data/caspwn/data06.out'))[0]
    np.testing.assert_allclose(res, ref, rtol=1e-06, err_msg='Problem occured when comparing to data06.out')

def test_T300_gold_03():
    s = plsp_system(300., 3.e-06, 150.e-06, gold_drude, PEC, vacuum)
    res = s.calculate('energy')
    ref = np.loadtxt(os.path.join(os.path.dirname(__file__), 'reference_data/caspwn/data07.out'))[0]
    np.testing.assert_allclose(res, ref, rtol=1e-06, err_msg='Problem occured when comparing to data07.out')

def test_T300_gold_04():
    s = plsp_system(300., 0.75e-06, 150.e-06, gold_drude, PEC, vacuum)
    res = s.calculate('energy')
    ref = np.loadtxt(os.path.join(os.path.dirname(__file__), 'reference_data/caspwn/data08.out'))[0]
    np.testing.assert_allclose(res, ref, rtol=1e-06, err_msg='Problem occured when comparing to data08.out')


# T=0, PR
def test_T0_PR_01():
    s = plsp_system(0., 1.e-06, 50.e-06, PEC, PEC, vacuum)
    s.automatic_integration = False
    res = s.calculate('energy')
    ref = np.loadtxt(os.path.join(os.path.dirname(__file__), 'reference_data/caspwn/data09.out'))
    np.testing.assert_allclose(res, ref, rtol=1e-06, err_msg='Problem occured when comparing to data09.out')

def test_T0_PR_02():
    s = plsp_system(0., 0.25e-06, 50.e-06, PEC, PEC, vacuum)
    s.automatic_integration = False
    res = s.calculate('energy')
    ref = np.loadtxt(os.path.join(os.path.dirname(__file__), 'reference_data/caspwn/data10.out'))
    np.testing.assert_allclose(res, ref, rtol=1e-06, err_msg='Problem occured when comparing to data10.out')

def test_T0_PR_03():
    s = plsp_system(0., 3.e-06, 150.e-06, PEC, PEC, vacuum)
    s.automatic_integration = False
    res = s.calculate('energy')
    ref = np.loadtxt(os.path.join(os.path.dirname(__file__), 'reference_data/caspwn/data11.out'))
    np.testing.assert_allclose(res, ref, rtol=1e-06, err_msg='Problem occured when comparing to data11.out')

def test_T0_PR_04():
    s = plsp_system(0., 0.75e-06, 150.e-06, PEC, PEC, vacuum)
    s.automatic_integration = False
    res = s.calculate('energy')
    ref = np.loadtxt(os.path.join(os.path.dirname(__file__), 'reference_data/caspwn/data12.out'))
    np.testing.assert_allclose(res, ref, rtol=1e-06, err_msg='Problem occured when comparing to data12.out')


# sphere-sphere
def test_spsp_01():
    s = spsp_system(300., 1.e-06, 50.e-06, 50.e-6, PEC, PEC, vacuum)
    res = s.calculate('energy')
    ref = np.loadtxt(os.path.join(os.path.dirname(__file__), 'reference_data/caspwn/data13.out'))[0]
    np.testing.assert_allclose(res, ref, rtol=1e-06, err_msg='Problem occured when comparing to data13.out')

def test_spsp_02():
    s = spsp_system(300., 1.e-06, 100.e-06, 50.e-6, PEC, PEC, vacuum)
    res = s.calculate('energy')
    ref = np.loadtxt(os.path.join(os.path.dirname(__file__), 'reference_data/caspwn/data14.out'))[0]
    np.testing.assert_allclose(res, ref, rtol=1e-06, err_msg='Problem occured when comparing to data14.out')

def test_spsp_03():
    s = spsp_system(300., 1.e-06, 50.e-06, 100.e-6, PEC, PEC, vacuum)
    res = s.calculate('energy')
    ref = np.loadtxt(os.path.join(os.path.dirname(__file__), 'reference_data/caspwn/data15.out'))[0]
    np.testing.assert_allclose(res, ref, rtol=1e-06, err_msg='Problem occured when comparing to data15.out')

def test_spsp_04():
    s = spsp_system(293., 1.e-06, 12.5e-06, 2.5e-6, silica, silica, water_zwol)
    res = s.calculate('energy')
    ref = np.loadtxt(os.path.join(os.path.dirname(__file__), 'reference_data/caspwn/data16.out'))[0]
    np.testing.assert_allclose(res, ref, rtol=1e-06, err_msg='Problem occured when comparing to data16.out')

def test_spsp_05():
    s = spsp_system(293., 1.e-06, 2.5e-06, 12.5e-6, silica, silica, water_zwol)
    res = s.calculate('energy')
    ref = np.loadtxt(os.path.join(os.path.dirname(__file__), 'reference_data/caspwn/data17.out'))[0]
    np.testing.assert_allclose(res, ref, rtol=1e-06, err_msg='Problem occured when comparing to data17.out')

def test_spsp_06():
    s = spsp_system(293., 1.e-07, 12.5e-06, 2.5e-6, silica, silica, water_zwol)
    res = s.calculate('energy')
    ref = np.loadtxt(os.path.join(os.path.dirname(__file__), 'reference_data/caspwn/data18.out'))[0]
    np.testing.assert_allclose(res, ref, rtol=1e-06, err_msg='Problem occured when comparing to data18.out')

def test_spsp_07():
    s = spsp_system(300., 1.e-06, 40.e-06, 20.e-6, gold_drude, polystyrene, vacuum)
    res = s.calculate('energy')
    ref = np.loadtxt(os.path.join(os.path.dirname(__file__), 'reference_data/caspwn/data19.out'))[0]
    np.testing.assert_allclose(res, ref, rtol=1e-06, err_msg='Problem occured when comparing to data19.out')

def test_spsp_08():
    s = spsp_system(300., 1.e-06, 40.e-06, 50.e-6, gold_drude, silica, water_zwol)
    res = s.calculate('energy')
    ref = np.loadtxt(os.path.join(os.path.dirname(__file__), 'reference_data/caspwn/data20.out'))[0]
    np.testing.assert_allclose(res, ref, rtol=1e-06, err_msg='Problem occured when comparing to data20.out')

# COMPARE TO CAPS
def test_caps_01():
    s = plsp_system(300., 5.e-06, 50.e-06, gold_drude, gold_drude, vacuum)
    res = s.calculate('energy')
    ref = np.loadtxt(os.path.join(os.path.dirname(__file__), 'reference_data/caps/data01.out'), delimiter=',')
    L = ref[1]
    R = ref[2]
    ref = ref[5]*hbar*c/(L+R)
    np.testing.assert_allclose(res, ref, rtol=5e-06, err_msg='Problem occured when comparing to caps/data01.out')

def test_caps_02():
    s = plsp_system(300., 2.e-06, 100.e-06, gold_drude, gold_drude, vacuum)
    res = s.calculate('energy')
    ref = np.loadtxt(os.path.join(os.path.dirname(__file__), 'reference_data/caps/data02.out'), delimiter=',')
    L = ref[1]
    R = ref[2]
    ref = ref[5]*hbar*c/(L+R)
    np.testing.assert_allclose(res, ref, rtol=5e-06, err_msg='Problem occured when comparing to caps/data02.out')

def test_caps_03():
    s = plsp_system(300., 6.e-06, 150.e-06, gold_drude, gold_drude, vacuum)
    res = s.calculate('energy')
    ref = np.loadtxt(os.path.join(os.path.dirname(__file__), 'reference_data/caps/data03.out'), delimiter=',')
    L = ref[1]
    R = ref[2]
    ref = ref[5]*hbar*c/(L+R)
    np.testing.assert_allclose(res, ref, rtol=5e-06, err_msg='Problem occured when comparing to caps/data03.out')

def test_caps_04():
    s = plsp_system(300., 1.e-06, 100.e-06, gold_drude, gold_drude, vacuum)
    res = s.calculate('energy')
    ref = np.loadtxt(os.path.join(os.path.dirname(__file__), 'reference_data/caps/data04.out'), delimiter=',')
    L = ref[1]
    R = ref[2]
    ref = ref[5]*hbar*c/(L+R)
    np.testing.assert_allclose(res, ref, rtol=5e-06, err_msg='Problem occured when comparing to caps/data04.out')

#print('All tests successful!')
