import sys, os
import argparse
from multiprocessing import cpu_count
from math import sqrt
from scipy.constants import k
from spsp_interaction import contribution_finite, contribution_zero
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ufuncs"))
from integration import fc_quadrature
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../material"))
import material

parser = argparse.ArgumentParser(description="Computation of the Casimir energy in the sphere-sphere geometry.")

# POSITIONAL ARGUMENTS
# --------------------
parser.add_argument("R1", help="radius of sphere 1 [m]", type=float)
parser.add_argument("R2", help="radius of sphere 1 [m]", type=float)
parser.add_argument("L", help="surface-to-surface distance [m]", type=float)
parser.add_argument("T", help="temperature [K]", type=float)

# OPTIONAL ARGUMENTS
# ------------------

# observables
obs = parser.add_mutually_exclusive_group()
obs.add_argument("--energy", help="compute only the Casimir energy", action="store_true")
obs.add_argument("--force", help="compute Casimir energy and force", action="store_true")
obs.add_argument("--pressure", help="compute Casimir energy, force and pressure (default)", action="store_true")

# materials
parser.add_argument("--sphere1", help="material of sphere 1", default="PR", type=str, metavar="")
parser.add_argument("--medium", help="material of medium", default="Vacuum", type=str, metavar="")
parser.add_argument("--sphere2", help="material of sphere 2", default="PR", type=str, metavar="")

# convergence parameters
parser.add_argument("--etaNin", help="inner radial discretization parameter", default=8.4, type=float, metavar="")
parser.add_argument("--Nin", help="inner radial discretization order", type=int, metavar="")
parser.add_argument("--etaNout", help="outer radial discretization parameter", default=5.3, type=float, metavar="")
parser.add_argument("--Nout", help="outer radial discretization order", type=int, metavar="")
parser.add_argument("--etaM", help="angular discretization parameter", default=6.2, type=float, metavar="")
parser.add_argument("--M", help="angular discretization order", type=int, metavar="")
parser.add_argument("--etalmax1", help="cut-off parameter for l-sum at sphere 1", default=12., type=float, metavar="")
parser.add_argument("--lmax1", help="cut-off for l-sum at sphere 1", type=int, metavar="")
parser.add_argument("--etalmax2", help="cut-off parameter for l-sum at sphere 2", default=12., type=float, metavar="")
parser.add_argument("--lmax2", help="cut-off for l-sum at sphere 2", type=int, metavar="")

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

if args.R1 <= 0.:
    parser.error("R1 needs to be positive!")
if args.R1 <= 0.:
    parser.error("R2 needs to be positive!")
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
sys.path.insert(0, os.path.join(NYSTROM_PATH, "sphere-sphere"))

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
print("# geometry: sphere-sphere")
if args.energy:
    observable = "energy"
    print("# observables: energy")
elif args.force:
    observable = "force"
    print("# observables: energy, force")
else:
    observable = "pressure"
    print("# observables: energy, force, pressure")
print("# R1 [m]:", args.R1)
print("# R2 [m]:", args.R2)
print("# L [m]:", args.L)
print("# T [K]:", args.T)
print("# material (sphere 1):", args.sphere1)
print("# material (medium):", args.medium)
print("# material (sphere 2):", args.sphere2)
print("#  ")
rhosum = max((args.R1 + args.R2) / args.L, 50)
if args.Nin == None:
    print("# etaNin:", args.etaNin)
    Nin = int(args.etaNin*sqrt(rhosum))
else:
    print("# etaNin: nan")
    Nin = args.Nin
print("# Nin:", Nin)
if args.Nout == None:
    print("# etaNout:", args.etaNout)
    Reff = 1/(1/args.R1+1/args.R2)
    rhoeff = max(Reff/args.L, 50)
    Nout = int(args.etaNout*sqrt(rhoeff))
else:
    print("# etaNout: nan")
    Nout = args.Nout
print("# Nout:", Nout)
if args.M == None:
    print("# etaM:", args.etaM)
    M = int(args.etaM*sqrt(rhosum))
else:
    print("# etaM: nan")
    M = args.M
print("# M:", M)
rho1 = max(args.R1/args.L, 50)
if args.lmax1 == None:
    print("# etalmax1:", args.etalmax1)
    lmax1 = int(args.etalmax1*rho1)
else:
    print("# etalmax1: nan")
    lmax1 = args.lmax1
print("# lmax1:", lmax1)
rho2 = max(args.R2/args.L, 50)
if args.lmax2 == None:
    print("# etalmax2:", args.etalmax2)
    lmax2 = int(args.etalmax2*rho2)
else:
    print("# etalmax2: nan")
    lmax2 = args.lmax2
print("# lmax2:", lmax2)

print("#")
print("# cores:", args.cores)
print("#")
# load materials
mat_sphere1 = eval("material."+args.sphere1)
mat_medium = eval("material."+args.medium)
mat_sphere2 = eval("material."+args.sphere2)
nfunc_sphere1 = mat_sphere1.n
nfunc_medium = mat_medium.n
nfunc_sphere2 = mat_sphere2.n

nds_outer, wts_outer = fc_quadrature(Nout)
nds_inner, wts_inner = fc_quadrature(Nin)

if args.T == 0.:
    if observable == "energy":
        j = 0
    if observable == "force":
        j = 1
    if observable == "pressure":
        j = 2
    func = lambda K: \
    contribution_finite(args.R1, args.R2, args.L, nfunc_medium(K / args.L) * K, nfunc_sphere1(K / args.L),
                        nfunc_sphere2(K / args.L), Nout, Nin, M, nds_outer, wts_outer, nds_inner,
                        wts_inner, lmax1, lmax2, args.cores, observable)[j]

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
        print("# pressure [N/m]")
    print(res)
else: # T > 0
    func = lambda K: \
        contribution_finite(args.R1, args.R2, args.L, nfunc_medium(K / args.L) * K, nfunc_sphere1(K / args.L)/nfunc_medium(K / args.L),
                            nfunc_sphere2(K / args.L)/nfunc_medium(K / args.L), Nout, Nin, M, nds_outer, wts_outer, nds_inner,
                            wts_inner, lmax1, lmax2, args.cores, observable)
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
    materialclass_sphere1 = mat_sphere1.materialclass
    materialclass_sphere2 = mat_sphere2.materialclass
    if materialclass_sphere1 == "dielectric":
        alpha_sphere1 = nfunc_sphere1(0.)**2/nfunc_medium(0.)**2
    elif materialclass_sphere1 == "plasma":
        alpha_sphere1 = eval("material." + args.sphere1 + ".K_plasma")*args.R1
    else: # will not be used
        alpha_sphere1 = 0.
    if materialclass_sphere2 == "dielectric":
        alpha_sphere2 = nfunc_sphere2(0.)**2/nfunc_medium(0.)**2
    elif materialclass_sphere2 == "plasma":
        alpha_sphere2 = eval("material." + args.sphere2 + ".K_plasma")*args.R2
    else: # will not be used
        alpha_sphere2 = 0.
    res0 = contribution_zero(args.R1, args.R2, args.L, alpha_sphere1, alpha_sphere2, materialclass_sphere1, materialclass_sphere2, Nout, Nin, M, nds_outer, wts_outer, nds_inner, wts_inner, lmax1, lmax2, args.cores, observable)
    res0 *= 0.5*k*args.T
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
        print("# energy [J] (n=0)")
        print(res0[0])
        if not (args.energy):
            print("# force [N] (n=0)")
            print(res0[1])
            if not (args.force):
                print("# pressure [N/m] (n=0)")
                print(res0[2])
    else:
        print("# energy [J] (n=0, n>0)")
        print(res0[0], res1[0])
        if not (args.energy):
            print("# force [N] (n=0, n>0)")
            print(res0[1], res1[1])
            if not (args.force):
                print("# pressure [N/m] (n=0, n>0)")
                print(res0[2], res1[2])
