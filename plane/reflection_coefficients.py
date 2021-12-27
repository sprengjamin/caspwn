from math import sqrt, exp
from scipy.constants import c

def def_reflection_coeff(medium, materials, thicknesses):
    Nlayers = len(materials)
    if Nlayers == 1:
        fresnel_coeff = def_fresnel_coefficients(medium, materials[0])
        reflection_coeff = fresnel_coeff

    if Nlayers == 2:
        mat1 = materials[0]
        mat2 = materials[1]
        d1 = thicknesses[0]
        fresnelm1 = def_fresnel_coefficients(medium, mat1)
        fresnel12 = def_fresnel_coefficients(mat1, mat2)

        def reflection_coeff(k0, k):
            rTMm1, rTEm1 = fresnelm1(k0, k)
            rTM12, rTE12 = fresnel12(k0, k)
            kappa1 = kappa(mat1, k0, k)
            rTM = (rTMm1 + rTM12*exp(-2*kappa1*d1))/(1 + rTMm1*rTM12*exp(-2*kappa1*d1))
            rTE = (rTEm1 + rTE12*exp(-2*kappa1*d1))/(1 + rTEm1*rTE12*exp(-2*kappa1*d1))
            return rTM, rTE

    if Nlayers == 3:
        mat1 = materials[0]
        mat2 = materials[1]
        mat3 = materials[2]
        d1 = thicknesses[0]
        d2 = thicknesses[1]
        fresnelm1 = def_fresnel_coefficients(medium, mat1)
        fresnel12 = def_fresnel_coefficients(mat1, mat2)
        fresnel23 = def_fresnel_coefficients(mat2, mat3)

        def reflection_coeff(k0, k):
            rTMm1, rTEm1 = fresnelm1(k0, k)
            rTM12, rTE12 = fresnel12(k0, k)
            rTM23, rTE23 = fresnel23(k0, k)
            kappa1 = kappa(mat1, k0, k)
            kappa2 = kappa(mat2, k0, k)

            rTM12_eff = (rTM12 + rTM23*exp(-2*kappa2*d2))/(1 + rTM12*rTM23*exp(-2*kappa2*d2))
            rTE12_eff = (rTE12 + rTE23*exp(-2*kappa2*d2))/(1 + rTE12*rTE23*exp(-2*kappa2*d2))

            rTM = (rTMm1 + rTM12_eff*exp(-2*kappa1*d1))/(1 + rTMm1*rTM12_eff*exp(-2*kappa1*d1))
            rTE = (rTEm1 + rTE12_eff*exp(-2*kappa1*d1))/(1 + rTEm1*rTE12_eff*exp(-2*kappa1*d1))
            return rTM, rTE

    if Nlayers == 4:
        mat1 = materials[0]
        mat2 = materials[1]
        mat3 = materials[2]
        mat4 = materials[3]
        d1 = thicknesses[0]
        d2 = thicknesses[1]
        d3 = thicknesses[2]
        fresnelm1 = def_fresnel_coefficients(medium, mat1)
        fresnel12 = def_fresnel_coefficients(mat1, mat2)
        fresnel23 = def_fresnel_coefficients(mat2, mat3)
        fresnel34 = def_fresnel_coefficients(mat3, mat4)

        def reflection_coeff(k0, k):
            rTMm1, rTEm1 = fresnelm1(k0, k)
            rTM12, rTE12 = fresnel12(k0, k)
            rTM23, rTE23 = fresnel23(k0, k)
            rTM34, rTE34 = fresnel34(k0, k)
            kappa1 = kappa(mat1, k0, k)
            kappa2 = kappa(mat2, k0, k)
            kappa3 = kappa(mat3, k0, k)

            rTM23_eff = (rTM23 + rTM34 * exp(-2 * kappa3 * d3)) / (1 + rTM23 * rTM34 * exp(-2 * kappa3 * d3))
            rTE23_eff = (rTE23 + rTE34 * exp(-2 * kappa3 * d3)) / (1 + rTE23 * rTE34 * exp(-2 * kappa3 * d3))

            rTM12_eff = (rTM12 + rTM23_eff*exp(-2*kappa2*d2))/(1 + rTM12*rTM23_eff*exp(-2*kappa2*d2))
            rTE12_eff = (rTE12 + rTE23_eff*exp(-2*kappa2*d2))/(1 + rTE12*rTE23_eff*exp(-2*kappa2*d2))

            rTM = (rTMm1 + rTM12_eff*exp(-2*kappa1*d1))/(1 + rTMm1*rTM12_eff*exp(-2*kappa1*d1))
            rTE = (rTEm1 + rTE12_eff*exp(-2*kappa1*d1))/(1 + rTEm1*rTE12_eff*exp(-2*kappa1*d1))
            return rTM, rTE

    return reflection_coeff

def kappa(mat, k0, k):
    if k0 == 0.:
        if mat.materialclass == "dielectric":
            return k
        elif mat.materialclass == "drude":
            return k
        elif mat.materialclass == "plasma":
            Kp = mat.wp/c
            return sqrt(Kp**2 + k**2)
    else: # k0 > 0
        return sqrt(mat.epsilon(k0*c)*k0**2 + k**2)


def def_fresnel_coefficients(mat1, mat2):
    def fresnel_coefficients(k0, k):
        if mat2.materialclass == "PEC":
            return 1., -1.
        if k0 == 0.:
            if mat1.materialclass == "dielectric":
                if mat2.materialclass == "dielectric":
                    eps1 = mat1.epsilon(0.)
                    eps2 = mat2.epsilon(0.)
                    rTM = (eps2 - eps1)/(eps2 + eps1)
                    rTE = 0.
                elif mat2.materialclass == "drude":
                    rTM = 1.
                    rTE = 0.
                elif mat2.materialclass == "plasma":
                    Kp = mat2.wp/c
                    rTM = 1.
                    rTE = (k - sqrt(Kp**2 + k**2))/(k + sqrt(Kp**2 + k**2))
            elif mat1.materialclass == "drude":
                if mat2.materialclass == "dielectric":
                    rTM = -1.
                    rTE = 0.
                elif mat2.materialclass == "drude":
                    wp1 = mat1.wp
                    gamma1 = mat1.gamma
                    wp2 = mat2.wp
                    gamma2 = mat2.gamma
                    rTM = (gamma1*wp2**2 - gamma2*wp1**2)/(gamma1*wp2**2 + gamma2*wp1**2)
                    rTE = 0.
            elif mat1.materialclass == "plasma":
                if mat2.materialclass == "dielectric":
                    Kp = mat1.wp/c
                    rTM = -1.
                    rTE = -(k - sqrt(Kp**2 + k**2))/(k + sqrt(Kp**2 + k**2))
                elif mat2.materialclass == "plasma":
                    Kp1 = mat1.wp/c
                    Kp2 = mat2.wp/c
                    q1 = sqrt(Kp1**2 + k**2)
                    q2 = sqrt(Kp2**2 + k**2)
                    rTM = (Kp2**2*q1 - Kp1**2*q2)/(Kp2**2*q1 + Kp1**2*q2)
                    rTE = (q1 - q2)/(q1 + q2)
            
        else: # k0 > 0
            eps1 = mat1.epsilon(k0*c)
            eps2 = mat2.epsilon(k0*c)
            q1 = sqrt(eps1*k0**2 + k**2)
            q2 = sqrt(eps2*k0**2 + k**2)
            rTM = (eps2*q1 - eps1*q2)/(eps1*q2 + eps2*q1)
            rTE = (q1 - q2)/(q2 + q1)

        return rTM, rTE
    return fresnel_coefficients

