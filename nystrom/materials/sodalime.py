materialclass = "dielectric"
wj = 1.911e16 # 12.6 eV/hbar
cj = 1.282

def epsilon(xi):
    return 1.+cj*wj**2/(wj**2+xi**2)

