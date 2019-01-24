import numpy as np

"""
Calculate the shear modulus of a material from the
Young's modulus and poissons ratio. 
"""
def calcShearModulus( E, v ):
    return E / (2*(1+v))

"""
Calculate the Young's modulus of a material from the
shear modulus and poissons ratio. 
"""
def calcYoungsModulus( G, v ):
    return 2*G*(1+v)
    
"""
Calculate the poissons ratio of a material from the
Young's and shear modulii. 
"""
def calcPoissonsRatio( E, G ):
    return (E/(2*G)) - 1
    