import sys, os
import argparse
from multiprocessing import cpu_count
from math import sqrt
from scipy.constants import k, c
from plsp_interaction import contribution_finite, contribution_zero
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ufuncs"))
from integration import fc_quadrature
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../material"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../plane"))


parser = argparse.ArgumentParser(description="Computation of the Casimir energy in the plane-sphere geometry.")

# POSITIONAL ARGUMENTS
# --------------------
parser.add_argument("R", help="radius of the sphere [m]", type=float)
parser.add_argument("L", help="surface-to-surface distance [m]", type=float)
parser.add_argument("T", help="temperature [K]", type=float)


# OPTIONAL ARGUMENTS
# ------------------

# observables
obs = parser.add_mutually_exclusive_group()
obs.add_argument("--energy", help="compute only the Casimir energy", action="store_true")
obs.add_argument("--force", help="compute Casimir energy and force", action="store_true")
obs.add_argument("--forcegradient", help="compute Casimir energy, force and force gradient (default)", action="store_true")

# materials
parser.add_argument("--sphere", help="material of sphere", default="PR", type=str, metavar="")
parser.add_argument("--medium", help="material of medium", default="vacuum", type=str, metavar="")
parser.add_argument("--plane", help="material of plane", default="PR", type=str, metavar="")

# additional layer
parser.add_argument("--add_plane_layer", help="material of addtional plane layer", type=str, metavar="")
parser.add_argument("--layer_thickness", help="thickness of front plane layer", type=float, metavar="")

# convergence parameters
parser.add_argument("--etaN", help="radial discretization parameter", default=5.6, type=float, metavar="")
parser.add_argument("--N", help="radial discretization order", type=int, metavar="")
parser.add_argument("--etaM", help="angular discretization parameter", default=5.1, type=float, metavar="")
parser.add_argument("--M", help="angular discretization order", type=int, metavar="")
parser.add_argument("--etalmax", help="cut-off parameter for l-sum", default=12., type=float, metavar="")
parser.add_argument("--lmax", help="cut-off for l-sum", type=int, metavar="")

# frequency summation/integration
group = parser.add_mutually_exclusive_group()
group.add_argument("--psd", help="use Pade-spectrum-decomposition for frequency summation (default for T>0)", action="store_true")
group.add_argument("--msd", help="use Matsubara-spectrum-decomposition for frequency summation", action="store_true")
group.add_argument("--ht", help="compute only high-temperature limit", action="store_true")
group.add_argument("--quad", help="use QUADPACK for frequency integration (default for T=0)", action="store_true")
parser.add_argument("--epsrel", help="relative error for --psd, --msd or --quad", default=1.e-08, type=float, metavar="")
group.add_argument("--fcq", help="use Fourier-Chebyshev quadrature scheme for frequency integration", action="store_true")
parser.add_argument("--X", help="quadrature order of Fourier-Chebyshev scheme (when --fcq is used)", type=int, metavar="")
parser.add_argument("--O", help="order of PSD or MSD (overwrites --epsrel)", type=int, metavar="")

# multiprocessing
parser.add_argument("--cores", help="number of CPU cores assigned", default=cpu_count(), type=int, metavar="")
args = parser.parse_args()

if args.R <= 0.:
    parser.error("R needs to be positive!")
if args.L <= 0.:
    parser.error("L needs to be positive!")
if args.T < 0.:
    parser.error("T cannot be negative!")
elif args.T == 0.:
    if (args.psd or args.msd):
        parser.error("--psd or --msd can only be used when T>0")
    if args.fcq:
        if args.X == None:
            parser.error("--X needs to be specified")
else:
    if args.quad or args.fcq:
        parser.error("--quad or --fcq can only be used when T=0")


NYSTROM_PATH = os.path.expanduser("~/wd/nystrom")
sys.path.insert(0, os.path.join(NYSTROM_PATH, "plane-sphere"))
import datetime
starttime = datetime.datetime.now()
print("# start time:", starttime.replace(microsecond=0))
import os
uname = os.uname()
print("# computed on:", uname.sysname, uname.nodename, uname.release, uname.machine)
import subprocess
HEAD_id = subprocess.check_output("git rev-parse --short HEAD", shell=True).strip().decode("utf-8")
print("# git HEAD:", HEAD_id)
branch = subprocess.check_output("git rev-parse --abbrev-ref HEAD", shell=True).strip().decode("utf-8")
print("# git branch:", branch)

print("#  ")
print("# geometry: plane-sphere")
if args.energy:
    observable = "energy"
    print("# observables: energy")
elif args.force:
    observable = "force"
    print("# observables: energy, force")
else:
    observable = "forcegradient"
    print("# observables: energy, force, force gradient")

print("# R [m]:", args.R)
print("# L [m]:", args.L)
print("# T [K]:", args.T)
print("# material (sphere):", args.sphere)
print("# material (medium):", args.medium)
print("# material (plane):", args.plane)
print("#  ")
rho = max(args.R/args.L, 50.)
if args.N == None:
    print("# etaN:", args.etaN)
    N = int(args.etaN*sqrt(rho))
else:
    print("# etaN: nan")
    N = args.N
print("# N:", N)
if args.M == None:
    print("# etaM:", args.etaM)
    M = int(args.etaM*sqrt(rho))
else:
    print("# etaM: nan")
    M = args.M
print("# M:", M)
if args.lmax == None:
    print("# etalmax:", args.etalmax)
    lmax = int(args.etalmax*rho)
else:
    print("# etalmax: nan")
    lmax = args.lmax
print("# lmax:", lmax)

print("#")
print("# cores:", args.cores)
print("#")

# load materials
import importlib
mat_plane = importlib.import_module(args.plane)
mat_medium = importlib.import_module(args.medium)
mat_sphere = importlib.import_module(args.sphere)
nfunc_plane = lambda xi: sqrt(mat_plane.epsilon(xi))
nfunc_medium = lambda xi: sqrt(mat_medium.epsilon(xi))
nfunc_sphere = lambda xi: sqrt(mat_sphere.epsilon(xi))

# define plane reflection coefficients
if args.add_plane_layer == None:
    from fresnel import rTM_zero, rTM_finite, rTE_zero, rTE_finite
    if mat_plane.materialclass == "dielectric":
        eps = mat_plane.epsilon(0.)/mat_medium.epsilon(0.)
    else:
        eps = 0. # dummy
    reflection_TM_zero = lambda k: rTM_zero(k, eps, mat_plane.materialclass)
    reflection_TM_finite = lambda k0, k: rTM_finite(k0*nfunc_medium(c*k0/args.L), k, mat_plane.epsilon(c * k0 / args.L)/mat_medium.epsilon(c * k0 / args.L))
    if mat_plane.materialclass == "plasma":
        Kp = mat_plane.wp * args.L / c
    else:
        Kp = 0. # dummy
    reflection_TE_zero = lambda k: rTE_zero(k, Kp, mat_plane.materialclass)
    reflection_TE_finite = lambda k0, k: rTE_finite(k0*nfunc_medium(c*k0/args.L), k, mat_plane.epsilon(c * k0 / args.L)/mat_medium.epsilon(c * k0 / args.L))

else:
    mat_layer = importlib.import_module(args.add_plane_layer)
    nfunc_layer = lambda xi: sqrt(mat_layer.epsilon(xi))

    from two_layers import rTM_1slab_zero, rTM_1slab_finite, rTE_1slab_zero, rTE_1slab_finite
    if mat_plane.materialclass == "dielectric":
        eps1 = nfunc_plane(0.)**2/nfunc_medium(0.)**2
        Kp1 = 0. # dummy
    else:
        raise NotImplementedError("In layered system only dielectric front layers (specified by --plane) are allowed.")
    if mat_layer.materialclass == "dielectric":
        eps2 = nfunc_layer(0.)**2/nfunc_plane(0.)**2
        Kp2 = 0. # dummy
    elif mat_layer.materialclass == "plasma":
        eps2 = 0. # dummy
        Kp2 = mat_layer.wp * args.L / c
    else:
        eps2 = 0. # dummy
        Kp2 = 0. # dummy

    reflection_TM_zero = lambda k: rTM_1slab_zero(k, eps1, mat_plane.materialclass, eps2, mat_layer.materialclass, args.layer_thickness/args.L)
    reflection_TM_finite = lambda k0, k: rTM_1slab_finite(k0, k, mat_medium.epsilon(c * k0 / args.L), mat_plane.epsilon(c * k0 / args.L), mat_layer.epsilon(c * k0 / args.L), args.layer_thickness/args.L)
    reflection_TE_zero = lambda k: rTE_1slab_zero(k, Kp1, mat_plane.materialclass, Kp2, mat_layer.materialclass, args.layer_thickness/args.L)
    reflection_TE_finite = lambda k0, k: rTE_1slab_finite(k0, k, mat_medium.epsilon(c * k0 / args.L), mat_plane.epsilon(c * k0 / args.L), mat_layer.epsilon(c * k0 / args.L), args.layer_thickness/args.L)


nds, wts = fc_quadrature(N)

if args.T == 0.:
    if observable == "energy":
        j = 0
    if observable == "force":
        j = 1
    if observable == "forcegradient":
        j = 2



    func = lambda k0: \
            contribution_finite(args.R, args.L, k0, nfunc_medium(c * k0 / args.L) * k0, nfunc_plane(c * k0 / args.L)/nfunc_medium(c * k0 / args.L), nfunc_sphere(c * k0 / args.L)/nfunc_medium(c * k0 / args.L), reflection_TM_finite, reflection_TE_finite, N, M, nds, wts, lmax, args.cores, observable)[j]

    if args.fcq:
        print("# integration method: fcq")
        print("# X:", args.X)
        print("#")
        print("# K, logdet, t(matrix construction), t(dft), t(matrix operations)")
        from integration import integral_fcq
        res = integral_fcq(args.L, func, args.X)
    else:
        print("# integration method: quad")
        print("# epsrel:", args.epsrel)
        print("#")
        print("# K, logdet, t(matrix construction), t(dft), t(matrix operations)")
        from integration import integral_quad
        res = integral_quad(args.L, func, args.epsrel)
    print("#")
    finishtime = datetime.datetime.now()
    print("# finish time:", finishtime.replace(microsecond=0))
    totaltime = finishtime-starttime
    print("# total elapsed time:", totaltime)
    print("#")
    if args.energy:
        print("# energy [J]")
    elif args.force:
        print("# force [N]")
    else:
        print("# force gradient [N/m]")
    print(res)
else: # T > 0
    func = lambda k0: \
        contribution_finite(args.R, args.L, k0, nfunc_medium(c * k0 / args.L) * k0, nfunc_plane(c * k0 / args.L)/nfunc_medium(c * k0 / args.L), nfunc_sphere(c * k0 / args.L)/nfunc_medium(c * k0 / args.L), reflection_TM_finite, reflection_TE_finite, N, M, nds, wts, lmax, args.cores, observable)

    if args.msd:
        print("# summation method: msd")
        print("# epsrel:", args.epsrel)
        from summation import msd_sum
    else: # psd
        print("# summation method: psd")
        print("# epsrel:", args.epsrel)
        from summation import psd_sum
    if args.ht:
        print("# mode: high-temperature limit")
    print("#")
    print("# K, logdet, t(matrix construction), t(dft), t(matrix operations)")
    materialclass_sphere = mat_sphere.materialclass
    if materialclass_sphere == "dielectric":
        alpha_sphere = nfunc_sphere(0.)**2/nfunc_medium(0.)**2
    elif materialclass_sphere == "plasma":
        alpha_sphere = eval("material." + args.sphere + ".K_plasma")*args.R
    else: # will not be used
        alpha_sphere = 0.

    res0_TM, res0_TE = contribution_zero(args.R, args.L, alpha_sphere, mat_plane.materialclass, mat_sphere.materialclass, reflection_TM_zero, reflection_TE_zero, N, M, nds,
                              wts, lmax, args.cores, observable)
    res0_TM *= 0.5*k*args.T
    res0_TE *= 0.5*k*args.T
    if args.msd:
        res1 = msd_sum(args.T, args.L, func, args.epsrel, nmax=args.O)
    elif not(args.ht):
        res1 = psd_sum(args.T, args.L, func, args.epsrel, order=args.O)
    print("#")
    finishtime = datetime.datetime.now()
    print("# finish time:", finishtime.replace(microsecond=0))
    totaltime = finishtime - starttime
    print("# total elapsed time:", totaltime)
    print("#")
    if args.ht:
        print("# energy [J] (n=0 TM, n=0 TE)")
        print(res0_TM[0], res0_TE[0])
        if not (args.energy):
            print("# force [N] (n=0 TM, n=0 TE)")
            print(res0_TM[1], res0_TE[1])
            if not (args.force):
                print("# force gradient [N/m] (n=0 TM, n=0 TE)")
                print(res0_TM[2], res0_TE[2])
    else:
        print("# energy [J] (n=0 TM, n=0 TE, n>0)")
        print(res0_TM[0], res0_TE[0], res1[0])
        if not (args.energy):
            print("# force [N] (n=0 TM, n=0 TE, n>0)")
            print(res0_TM[1], res0_TE[1], res1[1])
            if not (args.force):
                print("# force gradient [N/m] (n=0 TM, n=0 TE, n>0)")
                print(res0_TM[2], res0_TE[2], res1[2])
