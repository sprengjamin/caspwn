import numpy as np
import sys
from scipy.constants import hbar, c
sys.path.append('../..')
from nystrom.plane_sphere.compute_plsp import system as plsp_system
from nystrom.sphere_sphere.compute_spsp import system as spsp_system
from nystrom.materials import PEC, vacuum, silica, polystyrene, water_zwol, gold_drude

# PLANE-SPHERE

# T=300, PR
s = plsp_system(300., 1.e-06, 50.e-06, PEC, PEC, vacuum)
res = s.calculate('energy')
ref = np.loadtxt("reference_data/nystrom/data01.out")[0]
np.testing.assert_allclose(res, ref, rtol=1e-06, err_msg="Problem occured when comparing to data01.out")

s = plsp_system(300., 0.25e-06, 50.e-06, PEC, PEC, vacuum)
res = s.calculate('energy')
ref = np.loadtxt("reference_data/nystrom/data02.out")[0]
np.testing.assert_allclose(res, ref, rtol=1e-06, err_msg="Problem occured when comparing to data02.out")

s = plsp_system(300., 3.e-06, 150.e-06, PEC, PEC, vacuum)
res = s.calculate('energy')
ref = np.loadtxt("reference_data/nystrom/data03.out")[0]
np.testing.assert_allclose(res, ref, rtol=1e-06, err_msg="Problem occured when comparing to data03.out")

s = plsp_system(300., 0.75e-06, 150.e-06, PEC, PEC, vacuum)
res = s.calculate('energy')
ref = np.loadtxt("reference_data/nystrom/data04.out")[0]
np.testing.assert_allclose(res, ref, rtol=1e-06, err_msg="Problem occured when comparing to data04.out")

# T=300, Gold sphere

s = plsp_system(300., 1.e-06, 50.e-06, gold_drude, PEC, vacuum)
res = s.calculate('energy')
ref = np.loadtxt("reference_data/nystrom/data05.out")[0]
np.testing.assert_allclose(res, ref, rtol=1e-06, err_msg="Problem occured when comparing to data05.out")

s = plsp_system(300., 0.25e-06, 50.e-06, gold_drude, PEC, vacuum)
res = s.calculate('energy')
ref = np.loadtxt("reference_data/nystrom/data06.out")[0]
np.testing.assert_allclose(res, ref, rtol=1e-06, err_msg="Problem occured when comparing to data06.out")

s = plsp_system(300., 3.e-06, 150.e-06, gold_drude, PEC, vacuum)
res = s.calculate('energy')
ref = np.loadtxt("reference_data/nystrom/data07.out")[0]
np.testing.assert_allclose(res, ref, rtol=1e-06, err_msg="Problem occured when comparing to data07.out")

s = plsp_system(300., 0.75e-06, 150.e-06, gold_drude, PEC, vacuum)
res = s.calculate('energy')
ref = np.loadtxt("reference_data/nystrom/data08.out")[0]
np.testing.assert_allclose(res, ref, rtol=1e-06, err_msg="Problem occured when comparing to data08.out")


# T=0, PR

s = plsp_system(0., 1.e-06, 50.e-06, PEC, PEC, vacuum)
s.automatic_integration = False
res = s.calculate('energy')
ref = np.loadtxt("reference_data/nystrom/data09.out")
np.testing.assert_allclose(res, ref, rtol=1e-06, err_msg="Problem occured when comparing to data09.out")

s = plsp_system(0., 0.25e-06, 50.e-06, PEC, PEC, vacuum)
s.automatic_integration = False
res = s.calculate('energy')
ref = np.loadtxt("reference_data/nystrom/data10.out")
np.testing.assert_allclose(res, ref, rtol=1e-06, err_msg="Problem occured when comparing to data10.out")

s = plsp_system(0., 3.e-06, 150.e-06, PEC, PEC, vacuum)
s.automatic_integration = False
res = s.calculate('energy')
ref = np.loadtxt("reference_data/nystrom/data11.out")
np.testing.assert_allclose(res, ref, rtol=1e-06, err_msg="Problem occured when comparing to data11.out")

s = plsp_system(0., 0.75e-06, 150.e-06, PEC, PEC, vacuum)
s.automatic_integration = False
res = s.calculate('energy')
ref = np.loadtxt("reference_data/nystrom/data12.out")
np.testing.assert_allclose(res, ref, rtol=1e-06, err_msg="Problem occured when comparing to data12.out")


# sphere-sphere

s = spsp_system(300., 1.e-06, 50.e-06, 50.e-6, PEC, PEC, vacuum)
res = s.calculate('energy')
ref = np.loadtxt("reference_data/nystrom/data13.out")[0]
np.testing.assert_allclose(res, ref, rtol=1e-06, err_msg="Problem occured when comparing to data13.out")


s = spsp_system(300., 1.e-06, 100.e-06, 50.e-6, PEC, PEC, vacuum)
res = s.calculate('energy')
ref = np.loadtxt("reference_data/nystrom/data14.out")[0]
np.testing.assert_allclose(res, ref, rtol=1e-06, err_msg="Problem occured when comparing to data14.out")


s = spsp_system(300., 1.e-06, 50.e-06, 100.e-6, PEC, PEC, vacuum)
res = s.calculate('energy')
ref = np.loadtxt("reference_data/nystrom/data15.out")[0]
np.testing.assert_allclose(res, ref, rtol=1e-06, err_msg="Problem occured when comparing to data15.out")


s = spsp_system(293., 1.e-06, 12.5e-06, 2.5e-6, silica, silica, water_zwol)
res = s.calculate('energy')
ref = np.loadtxt("reference_data/nystrom/data16.out")[0]
np.testing.assert_allclose(res, ref, rtol=1e-06, err_msg="Problem occured when comparing to data16.out")

s = spsp_system(293., 1.e-06, 2.5e-06, 12.5e-6, silica, silica, water_zwol)
res = s.calculate('energy')
ref = np.loadtxt("reference_data/nystrom/data17.out")[0]
np.testing.assert_allclose(res, ref, rtol=1e-06, err_msg="Problem occured when comparing to data17.out")

s = spsp_system(293., 1.e-07, 12.5e-06, 2.5e-6, silica, silica, water_zwol)
res = s.calculate('energy')
ref = np.loadtxt("reference_data/nystrom/data18.out")[0]
np.testing.assert_allclose(res, ref, rtol=1e-06, err_msg="Problem occured when comparing to data18.out")

s = spsp_system(300., 1.e-06, 40.e-06, 20.e-6, gold_drude, polystyrene, vacuum)
res = s.calculate('energy')
ref = np.loadtxt("reference_data/nystrom/data19.out")[0]
np.testing.assert_allclose(res, ref, rtol=1e-06, err_msg="Problem occured when comparing to data19.out")

s = spsp_system(300., 1.e-06, 40.e-06, 50.e-6, gold_drude, silica, water_zwol)
res = s.calculate('energy')
ref = np.loadtxt("reference_data/nystrom/data20.out")[0]
np.testing.assert_allclose(res, ref, rtol=1e-06, err_msg="Problem occured when comparing to data20.out")

# COMPARE TO CAPS

s = plsp_system(300., 5.e-06, 50.e-06, gold_drude, gold_drude, vacuum)
res = s.calculate('energy')
ref = np.loadtxt("reference_data/caps/data01.out", delimiter=",")
L = ref[1]
R = ref[2]
ref = ref[5]*hbar*c/(L+R)
np.testing.assert_allclose(res, ref, rtol=5e-06, err_msg="Problem occured when comparing to caps/data01.out")

s = plsp_system(300., 2.e-06, 100.e-06, gold_drude, gold_drude, vacuum)
res = s.calculate('energy')
ref = np.loadtxt("reference_data/caps/data02.out", delimiter=",")
L = ref[1]
R = ref[2]
ref = ref[5]*hbar*c/(L+R)
np.testing.assert_allclose(res, ref, rtol=5e-06, err_msg="Problem occured when comparing to caps/data02.out")

s = plsp_system(300., 6.e-06, 150.e-06, gold_drude, gold_drude, vacuum)
res = s.calculate('energy')
ref = np.loadtxt("reference_data/caps/data03.out", delimiter=",")
L = ref[1]
R = ref[2]
ref = ref[5]*hbar*c/(L+R)
np.testing.assert_allclose(res, ref, rtol=5e-06, err_msg="Problem occured when comparing to caps/data03.out")

s = plsp_system(300., 1.e-06, 100.e-06, gold_drude, gold_drude, vacuum)
res = s.calculate('energy')
ref = np.loadtxt("reference_data/caps/data04.out", delimiter=",")
L = ref[1]
R = ref[2]
ref = ref[5]*hbar*c/(L+R)
np.testing.assert_allclose(res, ref, rtol=5e-06, err_msg="Problem occured when comparing to caps/data04.out")

print("All tests successful!")
