import sys, os
import argparse
from multiprocessing import cpu_count
from math import sqrt

parser = argparse.ArgumentParser(description="Computation of the Casimir energy in the plane-sphere geometry.")
parser.add_argument("R", help="radius of the sphere [m]", type=float)
parser.add_argument("L", help="surface-to-surface distance [m]", type=float)
parser.add_argument("T", help="temperatue [K]", type=float, metavar="")
parser.add_argument("--sphere", help="material of sphere", default="PR", type=str, metavar="")
parser.add_argument("--medium", help="material of medium", default="Vacuum", type=str, metavar="")
parser.add_argument("--plane", help="material of plane", default="PR", type=str, metavar="")
parser.add_argument("--etaN", help="radial discretization parameter", default=5.6, type=float, metavar="")
parser.add_argument("--etaM", help="angular discretization parameter", default=5.1, type=float, metavar="")
group = parser.add_mutually_exclusive_group()
group.add_argument("--psd", help="use Pade-spectrum-decoposition for frequency summation (default for T>0)", action="store_true")#, metavar="")
group.add_argument("--msd", help="use Matsubara-spectrum-decoposition for frequency summation", action="store_true")#, metavar="")
group.add_argument("--quad", help="use QUADPACK for frequency integration (default for T=0)", type=str, metavar="")
parser.add_argument("--epsrel", help="relative error for --psd, --msd or --quad", default=1.e-08, action="store_true")#, metavar="")
group.add_argument("--fcqs", help="use Fourier-Chebyshev quadrature scheme for frequency integration", action="store_true")#, metavar="")
parser.add_argument("--X", help="quadrature order of Fourier-Chebyshev scheme (when --fcqs is used)", type=int, metavar="")
parser.add_argument("--cores", help="number of CPU cores assigned", default=cpu_count(), type=int, metavar="")
args = parser.parse_args()

if args.R <= 0.:
    parse.error("R needs to be positive!")
if args.L <= 0.:
    parse.error("L needs to be positive!")
if args.T < 0.:
    parse.error("T cannot be negative!")
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
print("# start time:", datetime.datetime.now().replace(microsecond=0))
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
print("# R [m]:", args.R)
print("# L [m]:", args.L)
print("# T [K]:", args.T)
print("# material (sphere):", args.sphere)
print("# material (medium):", args.medium)
print("# material (plane):", args.plane)
print("#  ")
print("# etaN:", args.etaN)
N = int(args.etaN*sqrt(args.R/args.L))
print("# N:", N)
print("# etaM:", args.etaM)
M = int(args.etaM*sqrt(args.R/args.L))
print("# M:", M)

if args.T == 0.:
    if args.fcqs:
        print("# integration method: fcqs")
        print("# X:", args.X)
        print("#")
        print("# xi, logdet, timing: matrix construction, timing: logdet computation")
        from energy import energy_zero
        en = energy_zero(args.R, args.L, (args.sphere, args.medium, args.plane), N, M, args.cores, args.X)
    else:
        print("# integration method: quad")
        print("# epsrel:", args.epsrel)
        print("#")
        print("# xi, logdet, timing: matrix construction, timing: logdet computation")
        from energy import energy_quad
        en = energy_quad(args.R, args.L, (args.sphere, args.medium, args.plane), N, M, args.cores)
    print("#")
    print("# finish time:", datetime.datetime.now().replace(microsecond=0))
    print("#")
    print("# energy [J]")
    print(en)
else:
    if args.msd:
        mode = "msd"
    else:
        mode = "psd"
    print("# summation method:", mode)
    print("# epsrel:", args.epsrel)
    print("#")
    print("# xi, logdet, timing: matrix construction, timing: logdet computation")
    from energy import energy_finite
    en = energy_finite(args.R, args.L, args.T, (args.sphere, args.medium, args.plane), N, M, mode, args.epsrel, args.cores)
    print("#")
    print("# Finish time:", datetime.datetime.now().replace(microsecond=0))
    #print("# total elapsed time:")
    #print("# constructing matrices:")
    äprint("# computing logdet:")
    print("#")
    print("# energy [J] (n>=0, n>0)")
    print(en[0], en[1])
