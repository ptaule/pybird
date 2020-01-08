import os
#from time import time
#from timeit import timeit, Timer
import numpy as np
from numpy import pi, sin, log, exp
#from numpy.fft import rfft
from pyfftw.builders import rfft
from scipy.interpolate import interp1d
from scipy.special import gamma
from scipy.integrate import quad


# This is a matrix from powers of mu to Legendre polynomials
#TODO Maybe implement as a numpy array?
mu = {
    0: { 0: 1., 2: 0., 4: 0. },
    2: { 0: 1./3., 2: 2./3., 4: 0. },
    4: { 0: 1./5., 2: 4./7., 4: 8./35. },
    6: { 0: 1./7., 2: 10./21., 4: 24./77. },
    8: { 0: 1./9., 2: 40./99., 4: 48./148. }
}

M13b = {
    0: lambda n1: 1.125,
    1: lambda n1: -(1/(1 + n1)),
    2: lambda n1: 2.25,
    3: lambda n1: (3*(-1 + 3*n1))/(4.*(1 + n1)),
    4: lambda n1: -(1/(1 + n1)),
    5: lambda n1: -9/(4 + 4*n1),
    6: lambda n1: (9 + 18*n1)/(4 + 4*n1),
    7: lambda n1: (3*(-5 + 3*n1))/(8.*(1 + n1)),
    8: lambda n1: -9/(4 + 4*n1),
    9: lambda n1: (9*n1)/(4 + 4*n1),
}

def M13a(n1):
    return np.tan(n1*np.pi)/(14.*(-3 + n1)*(-2 + n1)*(-1 + n1)*n1*np.pi)

M22b = {
    0: lambda n1, n2: (6 + n1**4*(4 - 24*n2) - 7*n2 + 8*n1**5*n2 - 13*n2**2 + 4*n2**3 + 4*n2**4 + n1**2*(-13 + 38*n2 + 12*n2**2 - 8*n2**3) + 2*n1**3*(2 - 5*n2 - 4*n2**2 + 8*n2**3) + n1*(-7 - 6*n2 + 38*n2**2 - 10*n2**3 - 24*n2**4 + 8*n2**5))/(4.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2)),
    1: lambda n1, n2: (-18 + n1**2*(1 - 11*n2) - 12*n2 + n2**2 + 10*n2**3 + 2*n1**3*(5 + 7*n2) + n1*(-12 - 38*n2 - 11*n2**2 + 14*n2**3))/(7.*n1*(1 + n1)*n2*(1 + n2)),
    2: lambda n1, n2: (-3*n1 + 2*n1**2 + n2*(-3 + 2*n2))/(n1*n2),
    3: lambda n1, n2: (-4*(-24 + n2 + 10*n2**2) + 2*n1*(-2 + 51*n2 + 21*n2**2) + n1**2*(-40 + 42*n2 + 98*n2**2))/(49.*n1*(1 + n1)*n2*(1 + n2)),
    4: lambda n1, n2: (4*(3 - 2*n2 + n1*(-2 + 7*n2)))/(7.*n1*n2),
    5: lambda n1, n2: 2,
    6: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-2 + 3*n2 + 4*n1**4*n2 + 3*n2**2 - 2*n2**3 + n1**3*(-2 - 2*n2 + 4*n2**2) + n1**2*(3 - 10*n2 - 4*n2**2 + 4*n2**3) + n1*(3 + 2*n2 - 10*n2**2 - 2*n2**3 + 4*n2**4)))/(2.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2)),
    7: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(2 + 4*n2 + 5*n2**2 + n1**2*(5 + 7*n2) + n1*(4 + 10*n2 + 7*n2**2)))/(7.*n1*(1 + n1)*n2*(1 + n2)),
    8: lambda n1, n2: ((n1 + n2)*(-3 + 2*n1 + 2*n2))/(n1*n2),
    9: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(10 - 23*n2 + 28*n1**4*n2 + 5*n2**2 + 2*n2**3 + n1**3*(2 - 46*n2 + 28*n2**2) + n1**2*(5 - 38*n2 - 28*n2**2 + 28*n2**3) + n1*(-23 + 94*n2 - 38*n2**2 - 46*n2**3 + 28*n2**4)))/(14.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2)),
    10: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-58 + 4*n2 + 35*n2**2 + 7*n1**2*(5 + 7*n2) + n1*(4 + 14*n2 + 49*n2**2)))/(49.*n1*(1 + n1)*n2*(1 + n2)),
    11: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-8 + 7*n1 + 7*n2))/(7.*n1*n2),
    12: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(2 + 2*n1**3 - n2 - n2**2 + 2*n2**3 - n1**2*(1 + 2*n2) - n1*(1 + 2*n2 + 2*n2**2)))/(8.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2)),
    13: lambda n1, n2: ((1 + n1 + n2)*(2 + n1 + n2)*(-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2))/(8.*n1*(1 + n1)*n2*(1 + n2)),
    14: lambda n1, n2: -((-3 + 2*n1 + 2*n2)*(-6 - n1 + 2*n1**2 - n2 + 2*n2**2))/(8.*n1*(1 + n1)*n2*(1 + n2)),
    15: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(38 + 41*n2 + 112*n1**3*n2 - 66*n2**2 + 2*n1**2*(-33 - 18*n2 + 56*n2**2) + n1*(41 - 232*n2 - 36*n2**2 + 112*n2**3)))/(56.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2)),
    16: lambda n1, n2: -((-3 + 2*n1 + 2*n2)*(9 + 3*n1 + 3*n2 + 7*n1*n2))/(14.*n1*(1 + n1)*n2*(1 + n2)),
    17: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(5 + 5*n1 + 5*n2 + 7*n1*n2))/(14.*n1*(1 + n1)*n2*(1 + n2)),
    18: lambda n1, n2: (3 - 2*n1 - 2*n2)/(2.*n1*n2),
    19: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2))/(2.*n1*n2),
    20: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(50 - 9*n2 + 98*n1**3*n2 - 35*n2**2 + 7*n1**2*(-5 - 18*n2 + 28*n2**2) + n1*(-9 - 66*n2 - 126*n2**2 + 98*n2**3)))/(196.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2)),
    21: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(2 + n1 + 4*n1**3 + n2 - 8*n1*n2 - 8*n1**2*n2 - 8*n1*n2**2 + 4*n2**3))/(8.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2)),
    22: lambda n1, n2: ((2 + n1 + n2)*(-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(1 + 2*n1 + 2*n2))/(8.*n1*(1 + n1)*n2*(1 + n2)),
    23: lambda n1, n2: -((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(-2 + 7*n1 + 7*n2))/(56.*n1*(1 + n1)*n2*(1 + n2)),
    24: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(26 + 9*n2 + 56*n1**3*n2 - 38*n2**2 + 2*n1**2*(-19 - 18*n2 + 56*n2**2) + n1*(9 - 84*n2 - 36*n2**2 + 56*n2**3)))/(56.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2)),
    25: lambda n1, n2: (3*(-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2))/(32.*n1*(1 + n1)*n2*(1 + n2)),
    26: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(1 + 2*n1 + 2*n2)*(1 + 2*n1**2 - 8*n1*n2 + 2*n2**2))/(16.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2)),
    27: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(1 + 2*n1 + 2*n2)*(3 + 2*n1 + 2*n2))/(32.*n1*(1 + n1)*n2*(1 + n2)),
}

def M22a(n1, n2):
    return (gamma(1.5 - n1)*gamma(1.5 - n2)*gamma(-1.5 + n1 + n2))/(8.*np.pi**1.5*gamma(n1)*gamma(3 - n1 - n2)*gamma(n2))


def MPC(l, pn):
    """Matrix for spherical bessel transform from power spectrum to correlation function"""
    #TODO Check again
    return np.pi**-1.5 * 2.**(-2. * pn) * gamma(1.5+ l/2. - pn) / gamma(l/2. + pn)

def cH(Om, a):
    """Dimensionless Hubble function in conformal time, for matter + \Lambda"""
    return np.sqrt(Om / a + a**2 * (1. - Om))


def DgN(Om, a):
    """Linear growth factor"""
    integral = quad(lambda x: cH(Om, x)**(-3.), 0., a)[0]
    return 2.5 * Om * (cH(Om, a) / a) * integral


def fN(Om, z):
    """Logarithmic growth rate"""
    a = 1. / (1. + z)
    return 0.5 * (Om * (5. * a - 3. * DgN(Om, a))) / ((a**3 * (1. - Om) + Om) * DgN(Om, a))


Q100 = {
    0: lambda f: 0,
    1: lambda f: 0,
    2: lambda f: (-15 - 20*f - 22*f**2 - 12*f**3 - 3*f**4)/120.,
    3: lambda f: 0,
    4: lambda f: (35 + 70*f + 119*f**2 + 124*f**3 + 81*f**4 + 30*f**5 + 5*f**6)/840.,
    5: lambda f: 0,
    6: lambda f: (-945 - 2520*f - 5796*f**2 - 8856*f**3 - 9854*f**4 - 7720*f**5 - 3900*f**6 - 1120*f**7 - 140*f**8)/120960.,
    7: lambda f: 0,
    8: lambda f: (63 + 210*f + 609*f**2 + 1224*f**3 + 1766*f**4 + 1660*f**5 + 930*f**6 + 280*f**7 + 35*f**8)/60480.,
    9: lambda f: 0,
    10: lambda f: (-21 - 84*f - 294*f**2 - 732*f**3 - 1231*f**4 - 1256*f**5 - 732*f**6 - 224*f**7 - 28*f**8)/193536.,
    11: lambda f: 0,
    12: lambda f: (135 + 630*f + 2583*f**2 + 7668*f**3 + 14267*f**4 + 15250*f**5 + 9075*f**6 + 2800*f**7 + 350*f**8)/1.45152e7,
    13: lambda f: 0,
    14: lambda f: (-45 - 240*f - 1128*f**2 - 3888*f**3 - 7772*f**4 - 8560*f**5 - 5160*f**6 - 1600*f**7 - 200*f**8)/6.63552e7,
    15: lambda f: 0,
    16: lambda f: 0,
    17: lambda f: 0,
    18: lambda f: (-15 + 2*f**2 - 3*f**4)/180.,
    19: lambda f: (105 - 35*f**2 - 18*f**3 + 12*f**4)/630.,
    20: lambda f: (105 + 70*f + 21*f**2 - 36*f**3 + 3*f**4 + 30*f**5 + 15*f**6)/2520.,
    21: lambda f: (-105 - 70*f + 72*f**3 + 35*f**4 - 10*f**5 - 10*f**6)/1260.,
    22: lambda f: (-315 - 420*f - 420*f**2 - 36*f**3 + 182*f**4 + 20*f**5 - 180*f**6 - 140*f**7 - 35*f**8)/30240.,
    23: lambda f: (3465 + 4620*f + 3927*f**2 - 1386*f**3 - 4873*f**4 - 3040*f**5 + 225*f**6 + 910*f**7 + 280*f**8)/166320.,
    24: lambda f: (3465 + 6930*f + 11319*f**2 + 8712*f**3 + 330*f**4 - 4580*f**5 - 930*f**6 + 4200*f**7 + 4445*f**8 + 1890*f**9 + 315*f**10)/1.99584e6,
    25: lambda f: (-45045 - 90090*f - 138138*f**2 - 82368*f**3 + 62634*f**4 + 154700*f**5 + 105600*f**6 + 8400*f**7 - 29645*f**8 - 17010*f**9 - 3150*f**10)/1.297296e7,
    26: lambda f: (-693 - 1848*f - 4158*f**2 - 5544*f**3 - 2607*f**4 + 2608*f**5 + 2724*f**6 - 1512*f**7 - 3094*f**8 - 1512*f**9 - 252*f**10)/3.193344e6,
    27: lambda f: (45045 + 120120*f + 261261*f**2 + 321750*f**3 + 64350*f**4 - 364520*f**5 - 422790*f**6 - 104580*f**7 + 96740*f**8 + 68040*f**9 + 12600*f**10)/1.0378368e8,
    28: lambda f: (3465 + 11550*f + 33033*f**2 + 61380*f**3 + 42053*f**4 - 26550*f**5 - 41475*f**6 + 11200*f**7 + 36750*f**8 + 18900*f**9 + 3150*f**10)/1.596672e8,
    29: lambda f: (-45045 - 150150*f - 420420*f**2 - 751608*f**3 - 394823*f**4 + 692250*f**5 + 1050150*f**6 + 320600*f**7 - 223650*f**8 - 170100*f**9 - 31500*f**10)/1.0378368e9,
    30: lambda f: (-495 - 1980*f - 6864*f**2 - 16236*f**3 - 13332*f**4 + 6580*f**5 + 12840*f**6 - 2100*f**7 - 10225*f**8 - 5400*f**9 - 900*f**10)/2.737152e8,
    31: lambda f: (6435 + 25740*f + 87945*f**2 + 203346*f**3 + 143715*f**4 - 165880*f**5 - 298455*f**6 - 100050*f**7 + 61300*f**8 + 48600*f**9 + 9000*f**10)/1.7791488e9,
}
Q102 = {
    0: lambda f: 0,
    1: lambda f: 0,
    2: lambda f: 0,
    3: lambda f: -(f*(14 + 19*f + 12*f**2 + 3*f**3))/210.,
    4: lambda f: 0,
    5: lambda f: (f*(42 + 93*f + 112*f**2 + 78*f**3 + 30*f**4 + 5*f**5))/1260.,
    6: lambda f: 0,
    7: lambda f: -(f*(1386 + 4257*f + 7524*f**2 + 9071*f**3 + 7450*f**4 + 3855*f**5 + 1120*f**6 + 140*f**7))/166320.,
    8: lambda f: 0,
    9: lambda f: (f*(462 + 1815*f + 4224*f**2 + 6596*f**3 + 6460*f**4 + 3690*f**5 + 1120*f**6 + 140*f**7))/332640.,
    10: lambda f: 0,
    11: lambda f: -(f*(462 + 2211*f + 6380*f**2 + 11575*f**3 + 12260*f**4 + 7270*f**5 + 2240*f**6 + 280*f**7))/2.66112e6,
    12: lambda f: 0,
    13: lambda f: (f*(1386 + 7821*f + 26928*f**2 + 53882*f**3 + 59650*f**4 + 36075*f**5 + 11200*f**6 + 1400*f**7))/7.98336e7,
    14: lambda f: 0,
    15: lambda f: -(f*(66 + 429*f + 1716*f**2 + 3679*f**3 + 4190*f**4 + 2565*f**5 + 800*f**6 + 100*f**7))/4.56192e7,
    16: lambda f: 0,
    17: lambda f: 0,
    18: lambda f: (f*(-350 - 159*f + 36*f**2 + 84*f**3))/12600.,
    19: lambda f: (f*(-98 + 273*f + 72*f**2 - 132*f**3))/8820.,
    20: lambda f: (f*(210 + 289*f + 184*f**2 - 29*f**3 - 80*f**4 - 30*f**5))/12600.,
    21: lambda f: (f*(294 + 70*f - 212*f**2 - 103*f**3 + 80*f**4 + 55*f**5))/8820.,
    22: lambda f: (f*(-16170 - 36861*f - 46596*f**2 - 27034*f**3 + 1370*f**4 + 12675*f**5 + 7420*f**6 + 1540*f**7))/3.3264e6,
    23: lambda f: -(f*(35574 + 45969*f + 11352*f**2 - 26882*f**3 - 18950*f**4 + 6585*f**5 + 10640*f**6 + 3080*f**7))/2.32848e6,
    24: lambda f: (f*(60060 + 190476*f + 353496*f**2 + 389129*f**3 + 228510*f**4 + 4035*f**5 - 103460*f**6 - 80745*f**7 - 28350*f**8 - 4095*f**9))/6.48648e7,
    25: lambda f: (f*(336336 + 747747*f + 779064*f**2 + 189046*f**3 - 385020*f**4 - 339960*f**5 + 27440*f**6 + 174930*f**7 + 94500*f**8 + 17325*f**9))/9.081072e7,
    26: lambda f: (f*(-54054 - 219219*f - 533676*f**2 - 810784*f**3 - 697692*f**4 - 190050*f**5 + 215320*f**6 + 234360*f**7 + 90720*f**8 + 13104*f**9))/4.1513472e8,
    27: lambda f: -(f*(882882 + 2753751*f + 4650360*f**2 + 3199040*f**3 - 1327500*f**4 - 3044550*f**5 - 610400*f**6 + 1159200*f**7 + 756000*f**8 + 138600*f**9))/1.45297152e9,
    28: lambda f: (f*(150150 + 740883*f + 2227368*f**2 + 4153617*f**3 + 4233500*f**4 + 1535400*f**5 - 1104600*f**6 - 1414700*f**7 - 567000*f**8 - 81900*f**9))/1.0378368e10,
    29: lambda f: (f*(546546 + 2186184*f + 5012436*f**2 + 4609059*f**3 - 871900*f**4 - 3901725*f**5 - 1108800*f**6 + 1348900*f**7 + 945000*f**8 + 173250*f**9))/7.2648576e9,
    30: lambda f: (f*(-47190 - 274131*f - 979836*f**2 - 2111434*f**3 - 2373030*f**4 - 973005*f**5 + 562300*f**6 + 794100*f**7 + 324000*f**8 + 46800*f**9))/3.5582976e10,
    31: lambda f: -(f*(186186 + 907335*f + 2611752*f**2 + 2791438*f**3 - 263430*f**4 - 2250255*f**5 - 732400*f**6 + 742200*f**7 + 540000*f**8 + 99000*f**9))/2.49080832e10,
}
Q120 = {
    0: lambda f: 0,
    1: lambda f: 0,
    2: lambda f: -(f*(14 + 19*f + 12*f**2 + 3*f**3))/42.,
    3: lambda f: 0,
    4: lambda f: (f*(42 + 93*f + 112*f**2 + 78*f**3 + 30*f**4 + 5*f**5))/252.,
    5: lambda f: 0,
    6: lambda f: -(f*(1386 + 4257*f + 7524*f**2 + 9071*f**3 + 7450*f**4 + 3855*f**5 + 1120*f**6 + 140*f**7))/33264.,
    7: lambda f: 0,
    8: lambda f: (f*(462 + 1815*f + 4224*f**2 + 6596*f**3 + 6460*f**4 + 3690*f**5 + 1120*f**6 + 140*f**7))/66528.,
    9: lambda f: 0,
    10: lambda f: -(f*(462 + 2211*f + 6380*f**2 + 11575*f**3 + 12260*f**4 + 7270*f**5 + 2240*f**6 + 280*f**7))/532224.,
    11: lambda f: 0,
    12: lambda f: (f*(1386 + 7821*f + 26928*f**2 + 53882*f**3 + 59650*f**4 + 36075*f**5 + 11200*f**6 + 1400*f**7))/1.596672e7,
    13: lambda f: 0,
    14: lambda f: -(f*(66 + 429*f + 1716*f**2 + 3679*f**3 + 4190*f**4 + 2565*f**5 + 800*f**6 + 100*f**7))/9.12384e6,
    15: lambda f: 0,
    16: lambda f: 0,
    17: lambda f: 0,
    18: lambda f: (5*f**2 - 3*f**4)/63.,
    19: lambda f: (f**2*(-23 - 6*f + 9*f**2))/126.,
    20: lambda f: (f*(14 - 3*f - 16*f**2 - 2*f**3 + 10*f**4 + 5*f**5))/252.,
    21: lambda f: (f*(-308 + 99*f + 484*f**2 + 210*f**3 - 120*f**4 - 85*f**5))/2772.,
    22: lambda f: -(f*(231 + 231*f - 66*f**2 - 263*f**3 - 65*f**4 + 165*f**5 + 140*f**6 + 35*f**7))/8316.,
    23: lambda f: (f*(24024 + 22737*f - 14586*f**2 - 42913*f**3 - 24340*f**4 + 5415*f**5 + 10150*f**6 + 2905*f**7))/432432.,
    24: lambda f: (f*(18018 + 35607*f + 27456*f**2 - 10140*f**3 - 31420*f**4 - 10470*f**5 + 18480*f**6 + 21700*f**7 + 9450*f**8 + 1575*f**9))/2.594592e6,
    25: lambda f: -(f*(36036 + 69927*f + 44616*f**2 - 48828*f**3 - 109760*f**4 - 71142*f**5 + 1176*f**6 + 26852*f**7 + 14364*f**8 + 2583*f**9))/2.594592e6,
    26: lambda f: -(f*(12012 + 34749*f + 51480*f**2 + 17745*f**3 - 44120*f**4 - 42390*f**5 + 15120*f**6 + 37940*f**7 + 18900*f**8 + 3150*f**9))/1.0378368e7,
    27: lambda f: (f*(48048 + 137709*f + 193050*f**2 + 25545*f**3 - 274280*f**4 - 303798*f**5 - 56196*f**6 + 89978*f**7 + 57456*f**8 + 10332*f**9))/2.0756736e7,
    28: lambda f: (f*(30030 + 113685*f + 240240*f**2 + 148954*f**3 - 182250*f**4 - 244125*f**5 + 39200*f**6 + 180600*f**7 + 94500*f**8 + 15750*f**9))/2.0756736e8,
    29: lambda f: -(f*(60060 + 226083*f + 465036*f**2 + 231686*f**3 - 540600*f**4 - 769995*f**5 - 191240*f**6 + 210420*f**7 + 143640*f**8 + 25830*f**9))/2.0756736e8,
    30: lambda f: -(f*(1287 + 6006*f + 16302*f**2 + 12714*f**3 - 11495*f**4 - 18510*f**5 + 1500*f**6 + 12575*f**7 + 6750*f**8 + 1125*f**9))/8.895744e7,
    31: lambda f: (f*(72072 + 335049*f + 894894*f**2 + 621075*f**3 - 931420*f**4 - 1547175*f**5 - 433230*f**6 + 406315*f**7 + 287280*f**8 + 51660*f**9))/2.49080832e9,
}
Q122 = {
    0: lambda f: 0,
    1: lambda f: 0,
    2: lambda f: 0,
    3: lambda f: (-21 - 44*f - 58*f**2 - 36*f**3 - 9*f**4)/168.,
    4: lambda f: 0,
    5: lambda f: (231 + 726*f + 1551*f**2 + 1868*f**3 + 1317*f**4 + 510*f**5 + 85*f**6)/5544.,
    6: lambda f: 0,
    7: lambda f: (-27027 - 113256*f - 334620*f**2 - 596232*f**3 - 732778*f**4 - 610520*f**5 - 318660*f**6 - 92960*f**7 - 11620*f**8)/3.459456e6,
    8: lambda f: 0,
    9: lambda f: (9009 + 47190*f + 178035*f**2 + 419640*f**3 + 668810*f**4 + 663700*f**5 + 381750*f**6 + 116200*f**7 + 14525*f**8)/8.64864e6,
    10: lambda f: 0,
    11: lambda f: (-3003 - 18876*f - 86658*f**2 - 254020*f**3 - 470425*f**4 - 504440*f**5 - 300980*f**6 - 92960*f**7 - 11620*f**8)/2.7675648e7,
    12: lambda f: 0,
    13: lambda f: (3861 + 28314*f + 153153*f**2 + 536796*f**3 + 1096249*f**4 + 1227950*f**5 + 746925*f**6 + 232400*f**7 + 29050*f**8)/4.1513472e8,
    14: lambda f: 0,
    15: lambda f: (-9009 - 75504*f - 470184*f**2 - 1917552*f**3 - 4194988*f**4 - 4832240*f**5 - 2974440*f**6 - 929600*f**7 - 116200*f**8)/1.328431104e10,
    16: lambda f: 0,
    17: lambda f: 0,
    18: lambda f: (-105 - 226*f - 159*f**2 + 84*f**3 + 96*f**4)/5040.,
    19: lambda f: (-735 - 154*f + 381*f**2 + 108*f**3 - 198*f**4)/3528.,
    20: lambda f: (1155 + 3696*f + 6710*f**2 + 3892*f**3 - 1377*f**4 - 2580*f**5 - 880*f**6)/110880.,
    21: lambda f: (8085 + 10164*f + 2816*f**2 - 7556*f**3 - 3739*f**4 + 2720*f**5 + 1870*f**6)/77616.,
    22: lambda f: (-45045 - 191334*f - 532389*f**2 - 696072*f**3 - 395423*f**4 + 84250*f**5 + 260265*f**6 + 145460*f**7 + 29120*f**8)/1.729728e7,
    23: lambda f: (-315315 - 726726*f - 923637*f**2 - 183144*f**3 + 601429*f**4 + 426490*f**5 - 125655*f**6 - 220780*f**7 - 63910*f**8)/1.2108096e7,
    24: lambda f: (45045 + 238524*f + 880308*f**2 + 1729416*f**3 + 1971574*f**4 + 1114560*f**5 - 123660*f**6 - 697480*f**7 - 515235*f**8 - 177660*f**9 - 25200*f**10)/1.0378368e8,
    25: lambda f: (315315 + 1057056*f + 2277990*f**2 + 2315352*f**3 + 435778*f**4 - 1403208*f**5 - 1214280*f**6 + 43400*f**7 + 563955*f**8 + 309960*f**9 + 56826*f**10)/7.2648576e7,
    26: lambda f: (-45045 - 285714*f - 1305447*f**2 - 3384420*f**3 - 5340130*f**4 - 4561260*f**5 - 929670*f**6 + 1917160*f**7 + 1886220*f**8 + 710640*f**9 + 100800*f**10)/8.3026944e8,
    27: lambda f: (-315315 - 1387386*f - 4172883*f**2 - 7030140*f**3 - 4683460*f**4 + 2629572*f**5 + 5319330*f**6 + 1151080*f**7 - 1872360*f**8 - 1239840*f**9 - 227304*f**10)/5.81188608e8,
    28: lambda f: (45045 + 332904*f + 1807806*f**2 + 5793684*f**3 + 11166471*f**4 + 11321300*f**5 + 3453900*f**6 - 4065600*f**7 - 4569950*f**8 - 1776600*f**9 - 252000*f**10)/8.3026944e9,
    29: lambda f: (315315 + 1717716*f + 6608316*f**2 + 15255708*f**3 + 13859277*f**4 - 3911080*f**5 - 13561950*f**6 - 3981600*f**7 + 4361350*f**8 + 3099600*f**9 + 568260*f**10)/5.81188608e9,
    30: lambda f: (-45045 - 380094*f - 2387385*f**2 - 9089808*f**3 - 20115977*f**4 - 22460130*f**5 - 8003205*f**6 + 7403900*f**7 + 8994300*f**8 + 3553200*f**9 + 504000*f**10)/9.96323328e10,
    31: lambda f: (-315315 - 2048046*f - 9584289*f**2 - 27920256*f**3 - 29680889*f**4 + 5053230*f**5 + 27314595*f**6 + 9067100*f**7 - 8403150*f**8 - 6199200*f**9 - 1136520*f**10)/6.974263296e10,
}

Q000 = {
    0: lambda f: (-3 - 2*f - f**2)/6.,
    1: lambda f: 0,
    2: lambda f: (15 + 20*f + 22*f**2 + 12*f**3 + 3*f**4)/120.,
    3: lambda f: 0,
    4: lambda f: (-35 - 70*f - 119*f**2 - 124*f**3 - 81*f**4 - 30*f**5 - 5*f**6)/1680.,
    5: lambda f: 0,
    6: lambda f: (105 + 280*f + 644*f**2 + 984*f**3 + 846*f**4 + 360*f**5 + 60*f**6)/40320.,
    7: lambda f: 0,
    8: lambda f: (-21 - 70*f - 203*f**2 - 408*f**3 - 402*f**4 - 180*f**5 - 30*f**6)/80640.,
    9: lambda f: 0,
    10: lambda f: (7 + 28*f + 98*f**2 + 244*f**3 + 261*f**4 + 120*f**5 + 20*f**6)/322560.,
    11: lambda f: 0,
    12: lambda f: (-15 - 70*f - 287*f**2 - 852*f**3 - 963*f**4 - 450*f**5 - 75*f**6)/9.6768e6,
    13: lambda f: 0,
    14: lambda f: (15 + 80*f + 376*f**2 + 1296*f**3 + 1524*f**4 + 720*f**5 + 120*f**6)/1.548288e8,
    15: lambda f: 0,
    16: lambda f: (-3 + 2*f - f**2)/18.,
    17: lambda f: (15 - 10*f + 2*f**2)/45.,
    18: lambda f: (15 - 2*f**2 + 3*f**4)/180.,
    19: lambda f: (-105 + 35*f**2 + 18*f**3 - 12*f**4)/630.,
    20: lambda f: (-105 - 70*f - 21*f**2 + 36*f**3 - 3*f**4 - 30*f**5 - 15*f**6)/5040.,
    21: lambda f: (105 + 70*f - 72*f**3 - 35*f**4 + 10*f**5 + 10*f**6)/2520.,
    22: lambda f: (315 + 420*f + 420*f**2 + 36*f**3 - 182*f**4 - 20*f**5 + 180*f**6 + 140*f**7 + 35*f**8)/90720.,
    23: lambda f: (-3465 - 4620*f - 3927*f**2 + 1386*f**3 + 4873*f**4 + 3040*f**5 - 225*f**6 - 910*f**7 - 280*f**8)/498960.,
    24: lambda f: (-315 - 630*f - 1029*f**2 - 792*f**3 + 530*f**4 + 620*f**5 - 450*f**6 - 560*f**7 - 140*f**8)/725760.,
    25: lambda f: (3465 + 6930*f + 10626*f**2 + 6336*f**3 - 10978*f**4 - 14140*f**5 - 1080*f**6 + 3640*f**7 + 1120*f**8)/3.99168e6,
    26: lambda f: (63 + 168*f + 378*f**2 + 504*f**3 - 211*f**4 - 400*f**5 + 180*f**6 + 280*f**7 + 70*f**8)/1.45152e6,
    27: lambda f: (-3465 - 9240*f - 20097*f**2 - 24750*f**3 + 19690*f**4 + 37000*f**5 + 4350*f**6 - 9100*f**7 - 2800*f**8)/3.99168e7,
    28: lambda f: (-315 - 1050*f - 3003*f**2 - 5580*f**3 + 1777*f**4 + 4450*f**5 - 1575*f**6 - 2800*f**7 - 700*f**8)/8.70912e7,
    29: lambda f: (3465 + 11550*f + 32340*f**2 + 57816*f**3 - 31229*f**4 - 75650*f**5 - 10350*f**6 + 18200*f**7 + 5600*f**8)/4.790016e8,
    30: lambda f: (45 + 180*f + 624*f**2 + 1476*f**3 - 388*f**4 - 1180*f**5 + 360*f**6 + 700*f**7 + 175*f**8)/1.741824e8,
    31: lambda f: (-495 - 1980*f - 6765*f**2 - 15642*f**3 + 6545*f**4 + 19160*f**5 + 2835*f**6 - 4550*f**7 - 1400*f**8)/9.580032e8,
}
Q002 = {
    0: lambda f: 0,
    1: lambda f: -(f*(2 + f))/15.,
    2: lambda f: 0,
    3: lambda f: (f*(14 + 19*f + 12*f**2 + 3*f**3))/210.,
    4: lambda f: 0,
    5: lambda f: -(f*(42 + 93*f + 112*f**2 + 78*f**3 + 30*f**4 + 5*f**5))/2520.,
    6: lambda f: 0,
    7: lambda f: (f*(14 + 43*f + 76*f**2 + 69*f**3 + 30*f**4 + 5*f**5))/5040.,
    8: lambda f: 0,
    9: lambda f: -(f*(14 + 55*f + 128*f**2 + 132*f**3 + 60*f**4 + 10*f**5))/40320.,
    10: lambda f: 0,
    11: lambda f: (f*(42 + 201*f + 580*f**2 + 645*f**3 + 300*f**4 + 50*f**5))/1.2096e6,
    12: lambda f: 0,
    13: lambda f: -(f*(14 + 79*f + 272*f**2 + 318*f**3 + 150*f**4 + 25*f**5))/4.8384e6,
    14: lambda f: 0,
    15: lambda f: (f*(2 + 13*f + 52*f**2 + 63*f**3 + 30*f**4 + 5*f**5))/9.6768e6,
    16: lambda f: ((-2 + f)*f)/45.,
    17: lambda f: ((28 - 11*f)*f)/315.,
    18: lambda f: (f*(350 + 159*f - 36*f**2 - 84*f**3))/12600.,
    19: lambda f: (f*(98 - 273*f - 72*f**2 + 132*f**3))/8820.,
    20: lambda f: (f*(-210 - 289*f - 184*f**2 + 29*f**3 + 80*f**4 + 30*f**5))/25200.,
    21: lambda f: -(f*(294 + 70*f - 212*f**2 - 103*f**3 + 80*f**4 + 55*f**5))/17640.,
    22: lambda f: (f*(16170 + 36861*f + 46596*f**2 + 27034*f**3 - 1370*f**4 - 12675*f**5 - 7420*f**6 - 1540*f**7))/9.9792e6,
    23: lambda f: (f*(35574 + 45969*f + 11352*f**2 - 26882*f**3 - 18950*f**4 + 6585*f**5 + 10640*f**6 + 3080*f**7))/6.98544e6,
    24: lambda f: (f*(-4620 - 14652*f - 27192*f**2 - 27133*f**3 - 6550*f**4 + 9705*f**5 + 7420*f**6 + 1540*f**7))/1.99584e7,
    25: lambda f: -(f*(25872 + 57519*f + 59928*f**2 - 24658*f**3 - 53740*f**4 + 2280*f**5 + 21280*f**6 + 6160*f**7))/2.794176e7,
    26: lambda f: (f*(4158 + 16863*f + 41052*f**2 + 53408*f**3 + 18380*f**4 - 17430*f**5 - 14840*f**6 - 3080*f**7))/1.596672e8,
    27: lambda f: (f*(67914 + 211827*f + 357720*f**2 - 67520*f**3 - 295100*f**4 - 6750*f**5 + 106400*f**6 + 30800*f**7))/5.588352e8,
    28: lambda f: (f*(-11550 - 56991*f - 171336*f**2 - 263509*f**3 - 105100*f**4 + 82200*f**5 + 74200*f**6 + 15400*f**7))/4.790016e9,
    29: lambda f: -(f*(42042 + 168168*f + 385572*f**2 - 37457*f**3 - 308300*f**4 - 15825*f**5 + 106400*f**6 + 30800*f**7))/3.3530112e9,
    30: lambda f: (f*(3630 + 21087*f + 75372*f**2 + 130418*f**3 + 56510*f**4 - 39615*f**5 - 37100*f**6 - 7700*f**7))/1.9160064e10,
    31: lambda f: (f*(14322 + 69795*f + 200904*f**2 - 9274*f**3 - 158110*f**4 - 10635*f**5 + 53200*f**6 + 15400*f**7))/1.34120448e10,
}
Q020 = {
    0: lambda f: -(f*(2 + f))/3.,
    1: lambda f: 0,
    2: lambda f: (f*(14 + 19*f + 12*f**2 + 3*f**3))/42.,
    3: lambda f: 0,
    4: lambda f: -(f*(42 + 93*f + 112*f**2 + 78*f**3 + 30*f**4 + 5*f**5))/504.,
    5: lambda f: 0,
    6: lambda f: (f*(14 + 43*f + 76*f**2 + 69*f**3 + 30*f**4 + 5*f**5))/1008.,
    7: lambda f: 0,
    8: lambda f: -(f*(14 + 55*f + 128*f**2 + 132*f**3 + 60*f**4 + 10*f**5))/8064.,
    9: lambda f: 0,
    10: lambda f: (f*(42 + 201*f + 580*f**2 + 645*f**3 + 300*f**4 + 50*f**5))/241920.,
    11: lambda f: 0,
    12: lambda f: -(f*(14 + 79*f + 272*f**2 + 318*f**3 + 150*f**4 + 25*f**5))/967680.,
    13: lambda f: 0,
    14: lambda f: (f*(2 + 13*f + 52*f**2 + 63*f**3 + 30*f**4 + 5*f**5))/1.93536e6,
    15: lambda f: 0,
    16: lambda f: -((-2 + f)*f)/9.,
    17: lambda f: (f*(-28 + 11*f))/63.,
    18: lambda f: (f**2*(-5 + 3*f**2))/63.,
    19: lambda f: (f**2*(23 + 6*f - 9*f**2))/126.,
    20: lambda f: (f*(-14 + 3*f + 16*f**2 + 2*f**3 - 10*f**4 - 5*f**5))/504.,
    21: lambda f: (f*(308 - 99*f - 484*f**2 - 210*f**3 + 120*f**4 + 85*f**5))/5544.,
    22: lambda f: (f*(231 + 231*f - 66*f**2 - 263*f**3 - 65*f**4 + 165*f**5 + 140*f**6 + 35*f**7))/24948.,
    23: lambda f: -(f*(24024 + 22737*f - 14586*f**2 - 42913*f**3 - 24340*f**4 + 5415*f**5 + 10150*f**6 + 2905*f**7))/1.297296e6,
    24: lambda f: -(f*(1386 + 2739*f + 2112*f**2 - 3020*f**3 - 3020*f**4 + 1650*f**5 + 2240*f**6 + 560*f**7))/798336.,
    25: lambda f: (f*(36036 + 69927*f + 44616*f**2 - 107068*f**3 - 125440*f**4 + 1770*f**5 + 40600*f**6 + 11620*f**7))/1.0378368e7,
    26: lambda f: (f*(924 + 2673*f + 3960*f**2 - 3115*f**3 - 4600*f**4 + 1650*f**5 + 2800*f**6 + 700*f**7))/3.99168e6,
    27: lambda f: -(f*(48048 + 137709*f + 193050*f**2 - 207415*f**3 - 337000*f**4 - 12150*f**5 + 101500*f**6 + 29050*f**7))/1.0378368e8,
    28: lambda f: -(f*(2310 + 8745*f + 18480*f**2 - 10942*f**3 - 20050*f**4 + 5775*f**5 + 11200*f**6 + 2800*f**7))/9.580032e7,
    29: lambda f: (f*(60060 + 226083*f + 465036*f**2 - 350714*f**3 - 697400*f**4 - 40875*f**5 + 203000*f**6 + 58100*f**7))/1.24540416e9,
    30: lambda f: (f*(99 + 462*f + 1254*f**2 - 622*f**3 - 1315*f**4 + 330*f**5 + 700*f**6 + 175*f**7))/4.790016e7,
    31: lambda f: -(f*(72072 + 335049*f + 894894*f**2 - 543725*f**3 - 1245020*f**4 - 88935*f**5 + 355250*f**6 + 101675*f**7))/1.743565824e10,
}
Q022 = {
    0: lambda f: 0,
    1: lambda f: (-21 - 22*f - 11*f**2)/42.,
    2: lambda f: 0,
    3: lambda f: (21 + 44*f + 58*f**2 + 36*f**3 + 9*f**4)/168.,
    4: lambda f: 0,
    5: lambda f: (-231 - 363*f*(2 + f) - 297*f**2*(2 + f)**2 - 85*f**3*(2 + f)**3)/11088.,
    6: lambda f: 0,
    7: lambda f: (231 + 968*f + 2860*f**2 + 5096*f**3 + 4674*f**4 + 2040*f**5 + 340*f**6)/88704.,
    8: lambda f: 0,
    9: lambda f: (-231 - 605*f*(2 + f) - 990*f**2*(2 + f)**2 - 850*f**3*(2 + f)**3)/887040.,
    10: lambda f: 0,
    11: lambda f: (231 + 726*f*(2 + f) + 1485*f**2*(2 + f)**2 + 1700*f**3*(2 + f)**3)/1.064448e7,
    12: lambda f: 0,
    13: lambda f: (-33 - 242*f - 1309*f**2 - 4588*f**3 - 5397*f**4 - 2550*f**5 - 425*f**6)/2.128896e7,
    14: lambda f: 0,
    15: lambda f: (0.0015625 + (11*f*(2 + f))/1680. + (3*f**2*(2 + f)**2)/160. + (17*f**3*(2 + f)**3)/528.)/16128.,
    16: lambda f: -0.041666666666666664 - (29*f)/630. + (2*f**2)/45.,
    17: lambda f: (-735 + 616*f - 242*f**2)/1764.,
    18: lambda f: (105 + 226*f + 159*f**2 - 84*f**3 - 96*f**4)/5040.,
    19: lambda f: (735 + 154*f - 381*f**2 - 108*f**3 + 198*f**4)/3528.,
    20: lambda f: (-1155 - 3696*f - 6710*f**2 - 3892*f**3 + 1377*f**4 + 2580*f**5 + 880*f**6)/221760.,
    21: lambda f: (-8085 - 10164*f - 2816*f**2 + 7556*f**3 + 3739*f**4 - 2720*f**5 - 1870*f**6)/155232.,
    22: lambda f: (45045 + 191334*f + 532389*f**2 + 696072*f**3 + 395423*f**4 - 84250*f**5 - 260265*f**6 - 145460*f**7 - 29120*f**8)/5.189184e7,
    23: lambda f: (315315 + 726726*f + 923637*f**2 + 183144*f**3 - 601429*f**4 - 426490*f**5 + 125655*f**6 + 220780*f**7 + 63910*f**8)/3.6324288e7,
    24: lambda f: (-45045 - 238524*f - 880308*f**2 - 1729416*f**3 - 1739174*f**4 - 266720*f**5 + 835140*f**6 + 581840*f**7 + 116480*f**8)/4.1513472e8,
    25: lambda f: (-315315 - 1057056*f - 2277990*f**2 - 2315352*f**3 + 1191022*f**4 + 2342440*f**5 - 65040*f**6 - 883120*f**7 - 255640*f**8)/2.90594304e8,
    26: lambda f: (45045 + 285714*f + 1305447*f**2 + 3384420*f**3 + 4410530*f**4 + 1169900*f**5 - 1916250*f**6 - 1454600*f**7 - 291200*f**8)/4.1513472e9,
    27: lambda f: (315315 + 1387386*f + 4172883*f**2 + 7030140*f**3 - 1823740*f**4 - 6386500*f**5 - 202050*f**6 + 2207800*f**7 + 639100*f**8)/2.90594304e9,
    28: lambda f: (-45045 - 332904*f - 1807806*f**2 - 5793684*f**3 - 8842471*f**4 - 2842900*f**5 + 3660900*f**6 + 2909200*f**7 + 582400*f**8)/4.98161664e10,
    29: lambda f: (-315315 - 1717716*f - 6608316*f**2 - 15255708*f**3 + 2408723*f**4 + 13303400*f**5 + 768750*f**6 - 4415600*f**7 - 1278200*f**8)/3.487131648e10,
    30: lambda f: (45045 + 380094*f + 2387385*f**2 + 9089808*f**3 + 15467977*f**4 + 5503330*f**5 - 6226395*f**6 - 5091100*f**7 - 1019200*f**8)/6.974263296e11,
    31: lambda f: (315315 + 2048046*f + 9584289*f**2 + 27920256*f**3 - 2855111*f**4 - 23837870*f**5 - 1728195*f**6 + 7727300*f**7 + 2236850*f**8)/4.8819843072e11,
}

Qa = {
    0: {
        0: { 0: Q000, 2: Q002 },
        2: { 0: Q020, 2: Q022 }
    },
    1: {
        0: { 0: Q100, 2: Q102 },
        2: { 0: Q120, 2: Q122 }
    },
}

"""k's at which to output the power spectrum"""
kout = np.array([ 0.001, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.1, 0.105, 0.11, 0.115, 0.12, 0.125, 0.13, 0.135, 0.14, 0.145, 0.15, 0.155, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3 ])

"""Points at which to calculate the correlation function"""
qopti = np.array([ 5.000e+01,5.245e+01,5.490e+01,5.735e+01,5.980e+01,6.224e+01,6.469e+01,6.714e+01,6.959e+01,7.204e+01,7.449e+01,7.694e+01,7.939e+01,8.184e+01,8.429e+01,8.673e+01,8.918e+01,9.163e+01,9.408e+01,9.653e+01,9.898e+01,1.014e+02,1.039e+02,1.063e+02,1.088e+02,1.112e+02,1.137e+02,1.161e+02,1.186e+02,1.210e+02,1.235e+02,1.259e+02,1.284e+02,1.308e+02,1.333e+02,1.357e+02,1.382e+02,1.406e+02,1.431e+02,1.455e+02,1.480e+02,1.504e+02,1.529e+02,1.553e+02,1.578e+02,1.602e+02,1.627e+02,1.651e+02,1.676e+02,1.700e+02 ])

qEH = np.array([  1.000e+00,1.124e+00,1.264e+00,1.421e+00,1.597e+00,1.796e+00,2.019e+00,2.270e+00,2.551e+00,2.868e+00,3.225e+00,3.625e+00,4.075e+00,4.582e+00,5.151e+00,5.790e+00,6.510e+00,7.318e+00,8.227e+00,9.249e+00,1.040e+01,1.169e+01,1.314e+01,1.477e+01,1.661e+01,1.867e+01,2.099e+01,2.360e+01,2.653e+01,2.982e+01,3.353e+01,3.769e+01,4.238e+01,4.764e+01,5.356e+01,6.000e+01,6.021e+01,6.526e+01,6.769e+01,7.053e+01,7.579e+01,7.609e+01,8.105e+01,8.555e+01,8.632e+01,9.158e+01,9.617e+01,9.684e+01,1.021e+02,1.074e+02,1.081e+02,1.126e+02,1.179e+02,1.215e+02,1.232e+02,1.284e+02,1.337e+02,1.366e+02,1.389e+02,1.442e+02,1.495e+02,1.536e+02,1.547e+02,1.600e+02,1.727e+02,1.941e+02,2.183e+02,2.454e+02,2.759e+02,3.101e+02,3.486e+02,3.919e+02,4.406e+02,4.954e+02,5.569e+02,6.261e+02,7.038e+02,7.912e+02,8.895e+02,1.000e+03,1.000e+04])


class Common(object):
    """
    A class to share data among different objects

    Attributes
    ----------
    Nl : int
        The maximum multipole to calculate (default 2)
    kout : array
        The k's at which to calculate the power spectrum
    kIR : float
        An IR cutoff above which we resum
    """
    def __init__(self, Nl=2, kout=kout, kIR=0.02):
        
        self.Nl = Nl
        self.N11 = 3  # number of terms in the linear power spectrum
        self.Nct = 6  # number of counterterms
        self.N22 = 28  # number of 22-loops
        self.N13 = 10  # number of 13-loops
        self.Nloop = 12 # number of bias-independent loops
        self.k = kout
        self.Nk = (self.k).shape[0]
        self.s = qEH  # points at which to evaluate the correlation function
        self.Ns = (self.s).shape[0]
        self.kr = self.k[self.k >= kIR]  # points to resum, above an IR cutoff
        self.Nkr = self.kr.shape[0]
        self.Nklow = self.Nk-self.Nkr  # number of non-resummed points
        # The following are arrays where we store the terms
        self.l11 = np.empty(shape=(self.Nl, self.N11))
        self.lct = np.empty(shape=(self.Nl, self.Nct))
        self.l22 = np.empty(shape=(self.Nl, self.N22))
        self.l13 = np.empty(shape=(self.Nl, self.N13))
        
        for i in range(self.Nl):
            l = 2 * i
            self.l11[i] = np.array([ mu[0][l], mu[2][l], mu[4][l] ])
            self.lct[i] = np.array([ mu[0][l], mu[2][l], mu[4][l], mu[2][l], mu[4][l], mu[6][l] ])
            self.l22[i] = np.array([ 6*[mu[0][l]] + 7*[mu[2][l]] + [mu[4][l], mu[2][l], mu[4][l], mu[2][l], mu[4][l], mu[2][l]] + 3*[mu[4][l]] + [mu[6][l], mu[4][l], mu[6][l], mu[4][l], mu[6][l], mu[8][l]] ])
            self.l13[i] = np.array([ 2*[mu[0][l]] + 4*[mu[2][l]] + 3*[mu[4][l]] + [mu[6][l]] ])
        
        # Points between which we extract the BAO peak in the correlation function
        # self.sLow = 50.
        # self.sHigh = 190.
        # self.idlow = np.where(self.s>self.sLow)[0][0]
        # self.idhigh = np.where(self.s>self.sHigh)[0][0]
        # self.sbao = self.s[self.idlow:self.idhigh]
        # self.snobao = np.concatenate([self.s[:self.idlow], self.s[self.idhigh:]])
        # self.sr = self.sbao
        self.sr = self.s
        
common = Common()


class Bird(object):
    """
    A class used to calculate the nonlinear resummed power spectrum

    Attributes
    ----------
    kin : array
        The k array where the linear power spectrum is evaluated
    Plin : array
        The linear power spectrum
    Omega_m : float
        The value of Omega_m, today
    z : float
        The redshift of interest
    full : bool
        Resums or not the power spectrum (default True)
    co : class
        An object of type Common() used to share data

    Methods
    -------
    setBias(bs)
        Given an array of biases, calculates the different terms for loops and multipoles.
    
    setPs(bs)
        Given an array of biases, calculates the halo power spectrum.
    
    setCf(bs)
        Given an array of biases, calculates the halo correlation function.
    
    setPsCf(bs)
        Given an array of biases, calculates the halo power spectrum and correlation function.
    
    setfullPS()
        Sums up the halo power spectrum
    
    setPsCfl():
        Goes from real space to multipole space
    
    setPslb(bs):
        Given an array of biases, calculates the halo power spectrum in multipole space.
    
    def reducePsCfl():
        Gives the different terms of the halo power spectrum and correlation function in multipole space
    """
    def __init__(self, kin, Plin, Omega_m, z, full=True, co=common):
        self.full = full
        self.f = fN(Omega_m, z)
        self.kin = kin
        self.Pin = Plin
        self.co = co
        self.Plin = interp1d(kin, Plin, kind='cubic')
        self.P11 = self.Plin(co.k)
        self.P22 = np.empty(shape=(co.N22, co.Nk))
        self.P13 = np.empty(shape=(co.N13, co.Nk))
        
        if full is False:
            self.Ploopl = np.empty(shape=(co.Nl, co.Nloop, co.Nk))
            self.Cloopl = np.empty(shape=(co.Nl, co.Nloop, co.Ns))
            self.P11l = np.empty(shape=(co.Nl, co.N11, co.Nk))
            self.Pctl = np.empty(shape=(co.Nl, co.Nct, co.Nk))
            self.P22l = np.empty(shape=(co.Nl, co.N22, co.Nk))
            self.P13l = np.empty(shape=(co.Nl, co.N13, co.Nk))
        
        self.C11 = np.empty(shape=(co.Nl, co.Ns))
        self.C22 = np.empty(shape=(co.Nl, co.N22, co.Ns))
        self.C13 = np.empty(shape=(co.Nl, co.N13, co.Ns))
        self.Cct = np.empty(shape=(co.Nl, co.Ns))
        self.b11 = np.empty(shape=(co.Nl))
        self.b13 = np.empty(shape=(co.Nl, co.N13))
        self.b22 = np.empty(shape=(co.Nl, co.N22))
        self.bct = np.empty(shape=(co.Nl))
        self.Ps = np.empty(shape=(2, co.Nl, co.Nk))
        self.Cf = np.empty(shape=(2, co.Nl, co.Ns))
        self.fullPs = np.empty(shape=(co.Nl, co.Nk))
    
    def setBias(self, bs):
        """Given an array of biases, calculates the different terms for loops and multipoles.

        Parameters
        ----------
        bs : array
            An array of 7 biases
        """
        b1, b2, b3, b4, b5, b6, b7 = bs
        f = self.f
        for i in range(self.co.Nl):
            l = 2*i
            self.b11[i] = b1**2*mu[0][l] + 2.*b1*f*mu[2][l] + f**2*mu[4][l]
            self.b22[i] = np.array([ b1**2*mu[0][l], b1*b2*mu[0][l], b1*b4*mu[0][l], b2**2*mu[0][l], b2*b4*mu[0][l], b4**2*mu[0][l], b1**2*f*mu[2][l], b1*b2*f*mu[2][l], b1*b4*f*mu[2][l], b1*f*mu[2][l], b2*f*mu[2][l], b4*f*mu[2][l], b1**2*f**2*mu[2][l], b1**2*f**2*mu[4][l], b1*f**2*mu[2][l], b1*f**2*mu[4][l], b2*f**2*mu[2][l], b2*f**2*mu[4][l], b4*f**2*mu[2][l], b4*f**2*mu[4][l], f**2*mu[4][l], b1*f**3*mu[4][l], b1*f**3*mu[6][l], f**3*mu[4][l], f**3*mu[6][l], f**4*mu[4][l], f**4*mu[6][l], f**4*mu[8][l] ])
            self.b13[i] = np.array([ b1**2*mu[0][l], b1*b3*mu[0][l], b1**2*f*mu[2][l], b1*f*mu[2][l], b3*f*mu[2][l], b1*f**2*mu[2][l], b1*f**2*mu[4][l], f**2*mu[4][l], f**3*mu[4][l], f**3*mu[6][l] ])
            self.bct[i] = 2.*b1*(b5*mu[0][l]+b6*mu[2][l]+b7*mu[4][l]) + 2.*f*(b5*mu[2][l]+b6*mu[4][l]+b7*mu[6][l])
            
    def setPs(self, bs):
        """Given an array of biases, calculates the halo power spectrum.

        Parameters
        ----------
        bs : array
            An array of 7 biases
        """
        self.setBias(bs)
        self.Ps[0] = np.einsum('l,x->lx', self.b11, self.P11)
        self.Ps[1] = np.einsum('lb,bx->lx', self.b22, self.P22)
        # We subtract the first value so the P22 starts from 0
        for l in range(self.co.Nl):
            self.Ps[1,l] -= self.Ps[1,l,0]
        self.Ps[1] += np.einsum('lb,bx->lx', self.b13, self.P13) + np.einsum('l,x,x->lx', self.bct, self.co.k**2, self.P11)
    
    def setCf(self, bs):
        """Given an array of biases, calculates the halo correlation function.

        Parameters
        ----------
        bs : array
            An array of 7 biases
        """
        self.setBias(bs)
        self.Cf[0] = np.einsum('l,lx->lx', self.b11, self.C11)
        self.Cf[1] = np.einsum('lb,lbx->lx', self.b22, self.C22) + np.einsum('lb,lbx->lx', self.b13, self.C13) + np.einsum('l,lx->lx', self.bct, self.Cct)

    def setPsCf(self, bs):
        """Given an array of biases, calculates the halo power spectrum and correlation function.

        Parameters
        ----------
        bs : array
            An array of 7 biases
        """
        self.setBias(bs)
        self.Ps[0] = np.einsum('l,x->lx', self.b11, self.P11)
        self.Ps[1] = np.einsum('lb,bx->lx', self.b22, self.P22)
        for l in range(self.co.Nl):
            self.Ps[1,l] -= self.Ps[1,l,0]
        self.Ps[1] += np.einsum('lb,bx->lx', self.b13, self.P13) + np.einsum('l,x,x->lx', self.bct, self.co.k**2, self.P11)
        self.Cf[0] = np.einsum('l,lx->lx', self.b11, self.C11)
        self.Cf[1] = np.einsum('lb,lbx->lx', self.b22, self.C22) + np.einsum('lb,lbx->lx', self.b13, self.C13) + np.einsum('l,lx->lx', self.bct, self.Cct)
    
    def setfullPs(self, cf=False):
        """Sums up the halo power spectrum"""
        self.fullPs = np.sum(self.Ps, axis=0)
        if cf is True:
            return np.sum(self.Cf, axis=0)
    
    def setPsCfl(self):
        """Goes from real space to multipole space"""
        self.P11l = np.einsum('x,ln->lnx', self.P11, self.co.l11)
        self.Pctl = np.einsum('x,x,ln->lnx', self.co.k**2, self.P11, self.co.lct)
        self.P22l = np.einsum('nx,ln->lnx', self.P22, self.co.l22)
        self.P13l = np.einsum('nx,ln->lnx', self.P13, self.co.l13)
    
        self.C22 = np.einsum('lnx,ln->lnx', self.C22, self.co.l22)
        self.C13 = np.einsum('lnx,ln->lnx', self.C13, self.co.l13)
        
        self.reducePsCfl()
    
    def setPslb(self, bs):
        """Given an array of biases, calculates the halo power spectrum in multipole space.

        Parameters
        ----------
        bs : array
            An array of 7 biases
        """
        b1, b2, b3, b4, b5, b6, b7 = bs
        f = self.f
        
        b11 = np.array([ b1**2, 2.*b1*f, f**2 ])
        bct = np.array([ 2.*b1*b5, 2.*b1*b6, 2.*b1*b7, 2.*f*b5, 2.*f*b6, 2.*f*b7 ])
        b22 = np.array([ b1**2, b1*b2, b1*b4, b2**2, b2*b4, b4**2, b1**2*f, b1*b2*f, b1*b4*f, b1*f, b2*f, b4*f, b1**2*f**2, b1**2*f**2, b1*f**2, b1*f**2, b2*f**2, b2*f**2, b4*f**2, b4*f**2, f**2, b1*f**3, b1*f**3, f**3, f**3, f**4, f**4, f**4 ])
        b13 = np.array([ b1**2, b1*b3, b1**2*f, b1*f, b3*f, b1*f**2, b1*f**2, f**2, f**3, f**3 ])
        
        self.Ps[0] = np.einsum('b,lbx->lx', b11, self.P11l)
        self.Ps[1] = np.einsum('b,lbx->lx', b22, self.P22l)
        for l in range(self.co.Nl):
            self.Ps[1,l] -= self.Ps[1,l,0]
        self.Ps[1] += np.einsum('b,lbx->lx', b13, self.P13l) + np.einsum('b,lbx->lx', bct, self.Pctl)
        self.setfullPs()
        
    def reducePsCfl(self):
        """Gives the different terms of the halo power spectrum and correlation function in multipole space"""
        f1 = self.f
        
        self.Ploopl[:,0] = f1**2*self.P22l[:,20] + f1**3*self.P22l[:,23] + f1**3*self.P22l[:,24] + f1**4*self.P22l[:,25] + f1**4*self.P22l[:,26] + f1**4*self.P22l[:,27] + f1**2*self.P13l[:,7] + f1**3*self.P13l[:,8] + f1**3*self.P13l[:,9] # *1
        self.Ploopl[:,1] = f1*self.P22l[:,9] + f1**2*self.P22l[:,14] + f1**2*self.P22l[:,15] + f1**3*self.P22l[:,21] + f1**3*self.P22l[:,22] + f1*self.P13l[:,3] + f1**2*self.P13l[:,5] + f1**2*self.P13l[:,6] # *b1
        self.Ploopl[:,2] = f1*self.P22l[:,10] + f1**2*self.P22l[:,16] + f1**2*self.P22l[:,17] # *b2
        self.Ploopl[:,3] = f1*self.P13l[:,4] # *b3
        self.Ploopl[:,4] = f1*self.P22l[:,11] + f1**2*self.P22l[:,18] + f1**2*self.P22l[:,19] # *b4
        self.Ploopl[:,5] = self.P22l[:,0] + f1*self.P22l[:,6] + f1**2*self.P22l[:,12] + f1**2*self.P22l[:,13] + self.P13l[:,0] + f1*self.P13l[:,2] # *b1*b1
        self.Ploopl[:,6] = self.P22l[:,1] + f1*self.P22l[:,7] # *b1*b2
        self.Ploopl[:,7] = self.P13l[:,1] # *b1*b3
        self.Ploopl[:,8] = self.P22l[:,2] + f1*self.P22l[:,8] # *b1*b4
        self.Ploopl[:,9] = self.P22l[:,3] # *b2*b2
        self.Ploopl[:,10] = self.P22l[:,4] # *b2*b4
        self.Ploopl[:,11] = self.P22l[:,5] # *b4*b4
        
        self.Cloopl[:,0] = f1**2*self.C22[:,20] + f1**3*self.C22[:,23] + f1**3*self.C22[:,24] + f1**4*self.C22[:,25] + f1**4*self.C22[:,26] + f1**4*self.C22[:,27] + f1**2*self.C13[:,7] + f1**3*self.C13[:,8] + f1**3*self.C13[:,9] # *1
        self.Cloopl[:,1] = f1*self.C22[:,9] + f1**2*self.C22[:,14] + f1**2*self.C22[:,15] + f1**3*self.C22[:,21] + f1**3*self.C22[:,22] + f1*self.C13[:,3] + f1**2*self.C13[:,5] + f1**2*self.C13[:,6] # *b1
        self.Cloopl[:,2] = f1*self.C22[:,10] + f1**2*self.C22[:,16] + f1**2*self.C22[:,17] # *b2
        self.Cloopl[:,3] = f1*self.C13[:,4] # *b3
        self.Cloopl[:,4] = f1*self.C22[:,11] + f1**2*self.C22[:,18] + f1**2*self.C22[:,19] # *b4
        self.Cloopl[:,5] = self.C22[:,0] + f1*self.C22[:,6] + f1**2*self.C22[:,12] + f1**2*self.C22[:,13] + self.C13[:,0] + f1*self.C13[:,2] # *b1*b1
        self.Cloopl[:,6] = self.C22[:,1] + f1*self.C22[:,7] # *b1*b2
        self.Cloopl[:,7] = self.C13[:,1] # *b1*b3
        self.Cloopl[:,8] = self.C22[:,2] + f1*self.C22[:,8] # *b1*b4
        self.Cloopl[:,9] = self.C22[:,3] # *b2*b2
        self.Cloopl[:,10] = self.C22[:,4] # *b2*b4
        self.Cloopl[:,11] = self.C22[:,5] # *b4*b4
        
    def setreducePslb(self, bs):
        b1, b2, b3, b4, b5, b6, b7 = bs
        f = self.f
        
        b11 = np.array([ b1**2, 2.*b1*f, f**2 ])
        bct = np.array([ 2.*b1*b5, 2.*b1*b6, 2.*b1*b7, 2.*f*b5, 2.*f*b6, 2.*f*b7 ])
        bloop = np.array([ 1., b1, b2, b3, b4, b1*b1, b1*b2, b1*b3, b1*b4, b2*b2, b2*b4, b4*b4 ])
        self.Ps[0] = np.einsum('b,lbx->lx', b11, self.P11l)
        self.Ps[1] = np.einsum('b,lbx->lx', bloop, self.Ploopl)
        for l in range(self.co.Nl):
            self.Ps[1,l] -= self.Ps[1,l,0]
        self.Ps[1] += np.einsum('b,lbx->lx', bct, self.Pctl)
        self.setfullPs()

    def subtractShotNoise(self):
        """Subtracts the shotnoise estimated as the first entry of the power spectrum"""
        #TODO Remove the for loops
        for l in range(self.co.Nl): 
            for n in range(self.co.Nloop):
                shotnoise = self.Ploopl[l,n,0]
                self.Ploopl[l,n] -= shotnoise

    def formatTaylor(self):
        """Puts the power spectrum in a more convenient format for the driver program"""
        allk = np.concatenate([self.co.k, self.co.k]).reshape(-1,1)
        Plin = np.flip(np.einsum('n,lnk->lnk', np.array([1., 2.*self.f, self.f**2]), self.P11l), axis=1) 
        Plin = np.concatenate( np.einsum('lnk->lkn', Plin) , axis=0)
        Plin = np.hstack(( allk, Plin )) 
        Ploop1 = np.concatenate( np.einsum('lnk->lkn', self.Ploopl) , axis=0)
        Ploop2 = np.einsum('n,lnk->lnk', np.array([2., 2., 2., 2.*self.f, 2.*self.f, 2.*self.f]), self.Pctl)
        Ploop2 = np.concatenate( np.einsum('lnk->lkn', Ploop2) , axis=0)
        Ploop = np.hstack((allk, Ploop1, Ploop2))
        return Plin, Ploop

def CoefWindow(N, window=1):
    """Gives a window function that sends the function to 0 at the edges"""
    n = np.arange(-N//2, N//2 + 1)
    if window == 1:
        n_cut = N//2
    else:
        n_cut = int(window * N//2)
    
    n_right = n[-1] - n_cut
    n_left = n[0]+ n_cut 

    n_r = n[n[:] > n_right] 
    n_l = n[n[:] < n_left]

    theta_right = (n[-1] - n_r) / float(n[-1] - n_right - 1) 
    theta_left = (n_l - n[0]) / float(n_left - n[0] - 1)

    W = np.ones(n.size)
    W[n[:] > n_right] = theta_right - sin(2 * pi * theta_right) / (2 * pi)
    W[n[:] < n_left] = theta_left - sin(2 * pi * theta_left) / (2 * pi)
    return W


class FFTLog(object):
    """
    A class implementing the FFTLog algorithm.
    
    Attributes
    ----------
    Nmax : int, optional
        maximum number of points used to discretize the function
    xmin : float, optional
        minimum of the function to transform
    xmax : float, optional
        maximum of the function to transform
    bias : float, optional
        power by which we modify the function as x**bias * f
    
    Methods
    -------
    setx()
        Calculates the discrete x points for the transform
    
    setPow()
        Calculates the power in front of the function
    
    Coef()
        Calculates the single coefficients

    sumCoefxPow(xin, f, x, window=1)
        Puts the fft together
    """
    def __init__(self, **kwargs):
        self.Nmax = kwargs['Nmax']
        self.xmin = kwargs['xmin']
        self.xmax = kwargs['xmax']
        self.bias = kwargs['bias']
        self.dx = np.log(self.xmax/self.xmin) / (self.Nmax-1.)
        self.setx()
        self.setPow()
    
    def setx(self):
        self.x = np.empty(self.Nmax)
        for i in range(self.Nmax): 
            self.x[i] = self.xmin * np.exp(i*self.dx)
    
    def setPow(self):
        self.Pow = np.empty(self.Nmax+1, dtype = complex)
        for i in range(self.Nmax+1): 
            self.Pow[i] = self.bias + 1j * 2.*np.pi / (self.Nmax*self.dx) * (i - self.Nmax/2.)
    
    def Coef(self, xin, f, extrap='extrap', window=1, co=common):
        #TODO Maybe put the arguments in the init?
        interpfunc = interp1d(xin, f, kind='cubic')
        
        fx = np.empty(self.Nmax)
        tmp = np.empty(int(self.Nmax/2+1), dtype = complex)
        Coef = np.empty(self.Nmax+1, dtype = complex)
        
        if extrap is 'extrap':
            if xin[0] > self.x[0]:
                #print ('low extrapolation')
                nslow = (log(f[1])-log(f[0])) / (log(xin[1])-log(xin[0]))
                Aslow = f[0] / xin[0]**nslow
            if xin[-1] < self.x[-1]:
                #print ('high extrapolation')
                nshigh = (log(f[-1])-log(f[-2])) / (log(xin[-1])-log(xin[-2]))
                Ashigh = f[-1] / xin[-1]**nshigh
                
            for i in range(self.Nmax): 
                if xin[0] > self.x[i]: fx[i] = Aslow * self.x[i]**nslow * np.exp(-self.bias*i*self.dx)
                elif xin[-1] < self.x[i]: fx[i] = Ashigh * self.x[i]**nshigh * np.exp(-self.bias*i*self.dx)
                else: fx[i] = interpfunc(self.x[i]) * np.exp(-self.bias*i*self.dx)
        
        elif extrap is'padding':
            for i in range(self.Nmax): 
                if xin[0] > self.x[i]: fx[i] = 0.
                elif xin[-1] < self.x[i]: fx[i] = 0.
                else: fx[i] = interpfunc(self.x[i]) * np.exp(-self.bias*i*self.dx)
                
        #tmp = rfft(fx) ### numpy
        tmp = rfft(fx, planner_effort='FFTW_ESTIMATE')() ### pyfftw
        
        for i in range(self.Nmax+1):
            if (i < self.Nmax/2): Coef[i] = np.conj(tmp[int(self.Nmax/2-i)]) * self.xmin**(-self.Pow[i]) / float(self.Nmax)
            else: Coef[i] = tmp[int(i-self.Nmax/2)] * self.xmin**(-self.Pow[i]) / float(self.Nmax)
        
        if window is not None: Coef = Coef*CoefWindow(self.Nmax, window=window)
        else:
            Coef[0] /= 2.
            Coef[self.Nmax] /= 2.
        
        return Coef
        #return self.x, 
    
    def sumCoefxPow(self, xin, f, x, window=1):    
        Coef = self.Coef(xin, f, window=window)
        fFFT = np.empty_like(x)
        for i, xi in enumerate(x):
            fFFT[i] = np.real( np.sum(Coef * xi**self.Pow) )
        return fFFT

class NonLinear(object):
    def __init__(self, co=common, load=True, save=True, path='./', NFFT=256):
        
        #self.fftsettings = dict(Nmax=NFFT, xmin=1.5e-5, xmax=1000., bias=-1.6)
        self.fftsettings = dict(Nmax=NFFT, xmin=1.5e-5, xmax=20., bias=-1.6)

        self.fft = FFTLog(**self.fftsettings)
        self.co = co
        
        if load is True:
            try:
                L = np.load(os.path.join(path, 'pyegg%s.npz')%NFFT)
                if (self.fft.Pow-L['Pow']).any():
                    print ('Loaded loop matrices do not correspond to asked FFTLog configuration. \n Computing new matrices.')
                    load = False
                else:
                    self.M22, self.M13, self.Mcf11, self.Mcf22, self.Mcf13, self.Mcfct = L['M22'], L['M13'], L['Mcf11'], L['Mcf22'], L['Mcf13'], L['Mcfct']
            except:
                print ('Can\'t load loop matrices at %s. \n Computing new matrices.'%path)
                load = False
        
        if load is False:
            self.setM22()
            self.setM13()
            self.setMl()
            self.setMcf11()
            self.setMcf22()
            self.setMcf13()
            self.setMcfct()
        
        if save is True:
            try:
                np.savez( os.path.join(path, 'pyegg%s.npz')%NFFT, Pow=self.fft.Pow,
                     M22=self.M22, M13=self.M13, Mcf11=self.Mcf11, Mcf22=self.Mcf22, Mcf13=self.Mcf13, Mcfct=self.Mcfct )
            except:
                print ('Can\'t save loop matrices at %s.'%path)
        
        self.setkPow(co=co)
        self.setsPow(co=co)
        
        # To speed-up matrix multiplication:
        self.optipathP22 = np.einsum_path('nk,mk,bnm->bk', self.kPow, self.kPow, self.M22, optimize='optimal')[0]
        self.optipathC13 = np.einsum_path('ns,ms,blnm->bls', self.sPow, self.sPow, self.Mcf22, optimize='optimal')[0]
        self.optipathC22 = np.einsum_path('ns,ms,blnm->bls', self.sPow, self.sPow, self.Mcf13, optimize='optimal')[0]
        
    def setM22(self):
        self.M22 = np.empty(shape=(self.co.N22, self.fft.Pow.shape[0], self.fft.Pow.shape[0]), dtype='complex')
        # common piece of M22
        Ma = np.empty(shape=(self.fft.Pow.shape[0], self.fft.Pow.shape[0]), dtype='complex')
        for u, n1 in enumerate(-0.5*self.fft.Pow):
            for v, n2 in enumerate(-0.5*self.fft.Pow):
                Ma[u,v] = M22a(n1, n2)
        for i in range(self.co.N22): 
            # singular piece of M22
            Mb = np.empty(shape=(self.fft.Pow.shape[0], self.fft.Pow.shape[0]), dtype='complex')
            for u, n1 in enumerate(-0.5*self.fft.Pow):
                for v, n2 in enumerate(-0.5*self.fft.Pow):
                    Mb[u,v] = M22b[i](n1, n2)
            self.M22[i] = Ma*Mb
    
    def setM13(self):
        self.M13 = np.empty(shape=(self.co.N13, self.fft.Pow.shape[0]), dtype='complex')
        Ma = M13a(-0.5*self.fft.Pow)
        for i in range(self.co.N13): self.M13[i] = Ma*M13b[i](-0.5*self.fft.Pow)
    
    def setMcf11(self):
        self.Mcf11 = np.empty(shape=(self.co.Nl, self.fft.Pow.shape[0]), dtype='complex')
        for l in range(self.co.Nl):
            for u, n1 in enumerate(-0.5*self.fft.Pow):
                self.Mcf11[l,u] = MPC(2*l, n1)
        
    def setMl(self):
        self.Ml = np.empty(shape=(self.co.Nl, self.fft.Pow.shape[0], self.fft.Pow.shape[0]), dtype='complex')
        for l in range(self.co.Nl):
            for u, n1 in enumerate(-0.5*self.fft.Pow):
                for v, n2 in enumerate(-0.5*self.fft.Pow):
                    self.Ml[l,u,v] = MPC(2*l, n1+n2-1.5)
    
    def setMcf22(self):
        self.Mcf22 = np.einsum('lnm,bnm->blnm', self.Ml, self.M22)
    
    def setMcf13(self):
        self.Mcf13 = np.einsum('lnm,bn->blnm', self.Ml, self.M13)
    
    def setMcfct(self):
        self.Mcfct = np.empty(shape=(self.co.Nl, self.fft.Pow.shape[0]), dtype='complex')
        for l in range(self.co.Nl):
            for u, n1 in enumerate(-0.5*self.fft.Pow-1.):
                self.Mcfct[l,u] = MPC(2*l, n1)
    
    def setkPow(self):
        self.kPow = np.exp(np.einsum('n,k->nk', self.fft.Pow, np.log(self.co.k)))
    
    def setsPow(self):
        self.sPow = np.exp(np.einsum('n,s->ns', -self.fft.Pow-3., np.log(self.co.s)))
    
    def CoefkPow(self, Coef):
        return np.einsum('n,nk->nk', Coef, self.kPow )
    
    def CoefsPow(self, Coef):
        return np.einsum('n,ns->ns', Coef, self.sPow )
    
    def makeP22(self, CoefkPow, bird):
        bird.P22 = self.co.k**3 * np.real( np.einsum('nk,mk,bnm->bk', CoefkPow, CoefkPow, self.M22, optimize=self.optipathP22 ) )
    
    def makeP13(self, CoefkPow, bird):
        bird.P13 = self.co.k**3 * bird.P11 * np.real( np.einsum('nk,bn->bk', CoefkPow, self.M13) )
        
    def makeC11(self, CoefsPow, bird):
        bird.C11 = np.real( np.einsum('ns,ln->ls', CoefsPow, self.Mcf11) )
    
    def makeCct(self, CoefsPow, bird):
        bird.Cct = self.co.s**-2 * np.real( np.einsum('ns,ln->ls', CoefsPow, self.Mcfct) )
        
    def makeC22(self, CoefsPow, bird):
        bird.C22 = np.real( np.einsum('ns,ms,blnm->lbs', CoefsPow, CoefsPow, self.Mcf22, optimize=self.optipathC22 ) )
    
    def makeC13(self, CoefsPow, bird):
        bird.C13 = np.real( np.einsum('ns,ms,blnm->lbs', CoefsPow, CoefsPow, self.Mcf13, optimize=self.optipathC13 ) )
        
    def Coef(self, bird, window=None):
#         if bird.kin[0] > self.fftsettings['xmin']: print ('Please provide a linear power spectrum with kmin < %s'%self.fftsettings['xmin'])
#         if bird.kin[-1] < self.fftsettings['xmax']: print ('Please provide a linear power spectrum with kmax > %s'%self.fftsettings['xmax'])
        return self.fft.Coef(bird.kin, bird.Pin, window=window)
    
    def Ps(self, bird, window=None):
        coef = self.Coef(bird, window=window)
        coefkPow = self.CoefkPow(coef)
        self.makeP22(coefkPow, bird)
        self.makeP13(coefkPow, bird)
        
    def Cf(self, bird, window=None): 
        coef = self.Coef(bird, window=window)
        coefsPow = self.CoefsPow(coef)
        self.makeC11(coefsPow, bird)
        self.makeCct(coefsPow, bird)
        self.makeC22(coefsPow, bird)
        self.makeC13(coefsPow, bird)
    
    def PsCf(self, bird, window=None):
        coef = self.Coef(bird, window=window)
        coefkPow = self.CoefkPow(coef)
        coefsPow = self.CoefsPow(coef)
        self.makeP22(coefkPow, bird)
        self.makeP13(coefkPow, bird)
        self.makeC11(coefsPow, bird)
        self.makeCct(coefsPow, bird)
        self.makeC22(coefsPow, bird)
        self.makeC13(coefsPow, bird)


class Resum(object):
    """Class that resums the power spectrum
    
    Attributes
    ----------
    co : class
        Object of type Common(), used to share data
    
    LambdaIR : float
        
    """
    def __init__(self, co=common, LambdaIR=.066, NFFT=256):
        
        self.LambdaIR = LambdaIR
        self.co = co
        self.NIR = 32  #-4
        #TODO can put these to empty
        self.Q = np.zeros(shape=(2, co.Nl, co.Nl, self.NIR))
        self.IRcorr = np.zeros(shape=(2, co.Nl, self.NIR, co.Nkr))
        self.IR11 = np.zeros(shape=(co.Nl, self.NIR, co.Nkr))
        self.IRct = np.zeros(shape=(co.Nl, self.NIR, co.Nkr))
        self.IRloop = np.zeros(shape=(co.Nl, co.Nloop, self.NIR, co.Nkr))
        
        # keep these to zeros for padding zeros at low k
        self.IRresum = np.zeros(shape=(2, co.Nl, co.Nk))
        self.IR11resum = np.zeros(shape=(co.Nl, co.N11, co.Nk))
        self.IRctresum = np.zeros(shape=(co.Nl, co.Nct, co.Nk))
        self.IRloopresum = np.zeros(shape=(co.Nl, co.Nloop, co.Nk))
        #TODO Check these settings
        self.fftsettings = dict(Nmax=NFFT, xmin=0.1, xmax=10000., bias=-0.6)
        
        self.fft = FFTLog(**self.fftsettings)
        self.setM(co=co)
        self.setkPow()
        
        self.Xfftsettings = dict(Nmax=32, xmin=1.5e-5, xmax=10., bias=-2.6)
        self.Xfft = FFTLog(**self.Xfftsettings)
        self.setXM()
        self.setXsPow()
        
        k2pi = np.array([co.kr**2, co.kr**4, co.kr**6, co.kr**8, co.kr**10, co.kr**12, co.kr**14, co.kr**16])
        self.k2p = np.concatenate((k2pi, k2pi))
        
        self.alllpr = [
            [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], 
            [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], 
        ]

        keh, Peh = np.loadtxt('eh/EH_pk.dat', unpack = True)
        Omega_m_eh, z_eh = ((0.1189109449 + 0.02214)/0.6777**2, 0.5617)
        self.EH = Bird(keh, Peh, Omega_m_eh, z_eh, full=False)
        nonlinearEH = NonLinear(load=True,save=False)
        nonlinearEH.PsCf(self.EH)
        self.EH.setPsCfl()
    
    def setXsPow(self):
        self.XsPow = np.exp(np.einsum('n,s->ns', -self.Xfft.Pow-3., np.log(self.co.sr)))
    
    def setkPow(self):
        self.kPow = np.exp(np.einsum('n,s->ns', -self.fft.Pow-3., np.log(self.co.kr)))
    
    def setXM(self):
        self.XM = np.empty(shape=(2, self.Xfft.Pow.shape[0]), dtype='complex')
        for l in range(2):
            self.XM[l] = MPC(2*l, -0.5*self.Xfft.Pow)
        
    def IRFilters(self, bird, soffset=1., LambdaIR=None, RescaleIR=2.6, window=None):
        if LambdaIR is None:
            LambdaIR = self.LambdaIR
        Coef = self.Xfft.Coef(bird.kin, bird.Pin * np.exp(-bird.kin**2/LambdaIR**2)/bird.kin**2, window=window)
        CoefsPow = np.einsum('n,ns->ns', Coef, self.XsPow )
        X02 = np.real( np.einsum('ns,ln->ls', CoefsPow, self.XM) )
        X0offset = np.real( np.einsum('n,n->', np.einsum('n,n->n', Coef, soffset**(-self.Xfft.Pow-3.)), self.XM[0]) )
        X02[0] = X0offset-X02[0]
        X = RescaleIR * 2./3. * (X02[0] - X02[1])
        Y = 2. * X02[1]
        return X, Y
        
    def setM(self):
        self.M = np.empty(shape=(self.co.Nl, self.fft.Pow.shape[0]), dtype='complex')
        for l in range(self.co.Nl):
            self.M[l] = 8.*pi**3*MPC(2*l, -0.5*self.fft.Pow)
        
    def IRCorrection(self, XpYpC, k2p, lpr=None, window=None):
        Coef = self.fft.Coef(self.co.sr, XpYpC, extrap='padding', window=window)
        CoefkPow = np.einsum('n,nk->nk', Coef, self.kPow)
        return k2p * np.real( np.einsum('nk,ln->lk', CoefkPow, self.M) )
    
    def makeQ(self, f):
        for a in range(2):
            for l in range(self.co.Nl):
                for lpr in range(self.co.Nl):
                    for u in range(self.NIR):
                        self.Q[a][l][lpr][u] = Qa[1-a][2*l][2*lpr][u](f)
    
    def extractBAO(self, cf):
        return cf
    
    def Ps(self, bird, full=True, window=None):
        self.makeQ(bird.f)

        X, Y = self.IRFilters(bird)
        Xp = np.array([X, X**2, X**3, X**4, X**5, X**6, X**7, X**8])
        XpY = np.array([Y, X*Y, X**2*Y, X**3*Y, X**4*Y, X**5*Y, X**6*Y, X**7*Y])
        
        XpYp = np.concatenate((Xp, XpY))
        
        if full is True:
            for a, cf in enumerate(bird.Cf - self.EH.Cf):
                cbao = self.extractBAO(cf)
                for l, cl in enumerate(cbao):
                    u = 0
                    for j, (xy, k2pj, lpr) in enumerate(zip(XpYp, self.k2p, self.alllpr)):
                        IRcorrUnsorted = self.IRCorrection(xy*cl, k2pj, lpr=lpr, window=window)
                        for v in range(len(lpr)):
                            self.IRcorr[a,l,u+v] = IRcorrUnsorted[v]
                        u += len(lpr)
            self.IRresum[...,self.co.Nklow:] = np.einsum('alpn,apnk->alk', self.Q, self.IRcorr)
            bird.Ps += self.IRresum
            bird.setfullPs()
        else:
            cbao = self.extractBAO(bird.C11 - self.EH.C11)
            for l, cl in enumerate(cbao):
                u = 0
                for j, (xy, k2pj, lpr) in enumerate(zip(XpYp, self.k2p, self.alllpr)):
                    IRcorrUnsorted = self.IRCorrection(xy*cl, k2pj, lpr=lpr, window=window)
                    for v in range(len(lpr)): self.IR11[l,u+v] = IRcorrUnsorted[v]
                    u += len(lpr)
            cbao = self.extractBAO(bird.Cct - self.EH.Cct)
            for l, cl in enumerate(cbao):
                u = 0
                for j, (xy, k2pj, lpr) in enumerate(zip(XpYp, self.k2p, self.alllpr)):
                    IRcorrUnsorted = self.IRCorrection(xy*cl, k2pj, lpr=lpr, window=window)
                    for v in range(len(lpr)): self.IRct[l,u+v] = IRcorrUnsorted[v]
                    u += len(lpr)
            cbao = self.extractBAO(bird.Cloopl - self.EH.Cloopl)
            for l, cl in enumerate(cbao):
                for i, cli in enumerate(cl):
                    u = 0
                    for j, (xy, k2pj, lpr) in enumerate(zip(XpYp, self.k2p, self.alllpr)):
                        IRcorrUnsorted = self.IRCorrection(xy*cli, k2pj, lpr=lpr, window=window)
                        for v in range(len(lpr)): self.IRloop[l,i,u+v] = IRcorrUnsorted[v]
                        u += len(lpr)
            self.IR11resum[..., self.co.Nklow:] = np.einsum('lpn,pnk,pi->lik', self.Q[0], self.IR11, self.co.l11)
            self.IRctresum[..., self.co.Nklow:] = np.einsum('lpn,pnk,pi->lik', self.Q[1], self.IRct, self.co.lct)
            self.IRloopresum[..., self.co.Nklow:] = np.einsum('lpn,pink->lik', self.Q[1], self.IRloop)
            bird.P11l += self.IR11resum
            bird.Pctl += self.IRctresum
            bird.Ploopl += self.IRloopresum
