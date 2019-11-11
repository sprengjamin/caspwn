import sys, os
import argparse
from multiprocessing import cpu_count
from math import sqrt

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
obs.add_argument("--pressure", help="compute Casimir energy, force and pressure (default)", action="store_true")

# materials
parser.add_argument("--sphere", help="material of sphere", default="PR", type=str, metavar="")
parser.add_argument("--medium", help="material of medium", default="Vacuum", type=str, metavar="")
parser.add_argument("--plane", help="material of plane", default="PR", type=str, metavar="")

# convergence parameters
parser.add_argument("--etaN", help="radial discretization parameter", default=5.6, type=float, metavar="")
parser.add_argument("--etaM", help="angular discretization parameter", default=5.1, type=float, metavar="")
parser.add_argument("--etalmax", help="cut-off parameter", default=12., type=float, metavar="")

# frequency summation/integration
group = parser.add_mutually_exclusive_group()
group.add_argument("--psd", help="use Pade-spectrum-decomposition for frequency summation (default for T>0)", action="store_true")#, metavar="")
group.add_argument("--msd", help="use Matsubara-spectrum-decomposition for frequency summation", action="store_true")#, metavar="")
group.add_argument("--quad", help="use QUADPACK for frequency integration (default for T=0)", type=str, metavar="")
parser.add_argument("--epsrel", help="relative error for --psd, --msd or --quad", default=1.e-08, type=float, metavar="")
group.add_argument("--fcqs", help="use Fourier-Chebyshev quadrature scheme for frequency integration", action="store_true")#, metavar="")
parser.add_argument("--X", help="quadrature order of Fourier-Chebyshev scheme (when --fcqs is used)", type=int, metavar="")

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
    if args.fcqs:
        if args.X == None:
            parser.error("--X needs to be specified")
else:
    if args.quad or args.fcqs:
        parser.error("--quad or --fcqs can only be used when T=0")


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
    observable = "pressure"
    print("# observables: energy, force, pressure")

print("# R [m]:", args.R)
print("# L [m]:", args.L)
print("# T [K]:", args.T)
print("# material (sphere):", args.sphere)
print("# material (medium):", args.medium)
print("# material (plane):", args.plane)
print("#  ")
rho = max(args.R/args.L, 50.)
print("# etaN:", args.etaN)
N = int(args.etaN*sqrt(rho))
print("# N:", N)
print("# etaM:", args.etaM)
M = int(args.etaM*sqrt(rho))
print("# M:", M)
print("# etalmax:", args.etalmax)
lmax = int(args.etalmax*rho)
print("# lmax:", lmax)

if args.T == 0.:
    if args.fcqs:
        print("# integration method: fcqs")
        print("# X:", args.X)
        print("#")
        print("# xi, logdet, timing: matrix construction, timing: logdet computation")
        from energy import casimir_zero
        res = casimir_zero(args.R, args.L, (args.plane, args.medium, args.sphere), N, M, lmax, args.cores, observable, args.X)
    else:
        print("# integration method: quad")
        print("# epsrel:", args.epsrel)
        print("#")
        print("# xi, logdet, timing: matrix construction, timing: logdet computation")
        from energy import casimir_quad
        res = casimir_quad(args.R, args.L, (args.plane, args.medium, args.sphere), N, M, lmax, args.cores, observable)
    print("#")
    finishtime = datetime.datetime.now()
    print("# finish time:", finishtime.replace(microsecond=0))
    totaltime = finishtime - starttime
    print("# total elapsed time:", totaltime)
    print("#")
    print("# energy [J]")
    print(res[0])
    if not (args.energy):
        print("# force [N]")
        print(res[1])
        if not (args.force):
            print("# pressure [N/m]")
            print(res[2])
else:
    if args.msd:
        mode = "msd"
    else:
        mode = "psd"
    print("# summation method:", mode)
    print("# epsrel:", args.epsrel)
    print("#")
    print("# xi, logdet, timing: matrix construction, timing: logdet computation")
    from energy import casimir_finite
    res = casimir_finite(args.R, args.L, args.T, (args.plane, args.medium, args.sphere), N, M, lmax, mode, args.epsrel, args.cores, observable)
    print("#")
    finishtime = datetime.datetime.now()
    print("# finish time:", finishtime.replace(microsecond=0))
    totaltime = finishtime - starttime
    print("# total elapsed time:", totaltime)
    print("#")
    print("# energy [J] (n>=0, n>0)")
    print(res[0][0], res[1][0])
    if not(args.energy):
        print("# force [N] (n>=0, n>0)")
        print(res[0][1], res[1][1])
        if not (args.force):
            print("# pressure [N/m] (n>=0, n>0)")
            print(res[0][2], res[1][2])
