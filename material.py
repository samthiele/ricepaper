"""
A class for defining/storing the material properties of individual nodes and then generating node packings
to fill a domain with particles (at this stage only by gravity settling, though it would be straight-forward to 
implement a sphere-packing algorithm).

Sam Thiele 2018
"""

import numpy as np
from ricepaper import RicePaper
from ricepaper.utils import *
class NodeSet:
    
    """
    Initialise a DEM material by defining a set of node properties
    
    **Arguments**:
     -name = the name of this nodeset
     -radii = list defining radii of particles in this material (in meters). Must be a list (as the length of this defines how many particle types to define).
     -density = list defining the density of partices in this material (in kg/m^3). If a single value is passed, this is used for all particle types.
     -E = list defining the Young's moduli of each particle (determines repulsive forces during particle interactions; in Pa). If a single value is passed, this is used for all particle types. 
     -v = list defining the poissons ratio of each particle (determines the lateral force during particle interactions; dimensionless). If a single value is passed, this is used for all particle types.
     -frict = list of friction coefficients (determines the shear force during particle interactions; dimensionless). If a single value is passed, this is used for all particle types.
    """
    def __init__( self, name, radii, density, E, v ):
        self.name = name
        self.ntypes = len(radii)
        
        if self.ntypes > 20:
            print("Warning: Riceball can only define a max of 20 particle types. This model will probably crash...")
        
        if not isinstance(radii,list):
            radii = [radii]*self.ntypes
        if not isinstance(density,list):
            density = [density]*self.ntypes
        if not isinstance(E,list):
            E = [E]*self.ntypes
        if not isinstance(v,list):
            v = [v]*self.ntypes
            
        assert self.ntypes == len(density) and self.ntypes == len(E) and self.ntypes == len(v), "Error: lists of particle properties must be the same length."
        
        self.radii = np.array(radii)
        self.density = np.array(density)
        self.E = np.array(E)
        self.v = np.array(v)
        
        #init interaction tables to zero
        self.friction = np.zeros([self.ntypes,self.ntypes])
        self.cohesion = np.zeros([self.ntypes,self.ntypes])
        self.nStiff = np.zeros([self.ntypes,self.ntypes])
        self.sStiff = np.zeros([self.ntypes,self.ntypes])
        self.tStrength = np.zeros([self.ntypes,self.ntypes])
        self.sStrength = np.zeros([self.ntypes,self.ntypes])
        
    """
    Define properties of frictional contacts between particles.
    
    **Arguments**:
        -friction = [n x n] array where element [i,k] contains the friction coefficient between particles of type i and k If a single value is passed this is value is applied to all possible contacts.
        -cohesion = [n x n] array where element [i,k] contains the cohesion (in newtons) between particles of type i and k. If a single value is passed this is value is applied to all possible contacts.
    """
    def setFriction( self, friction, cohesion ):
        if not isinstance(friction,list):
            friction = np.zeros( [self.ntypes,self.ntypes] ) + friction
        if not isinstance(cohesion,list):
            cohesion = np.zeros( [self.ntypes,self.ntypes] ) + cohesion
    
        assert friction.shape == self.friction.shape, "Invalid friction array: required dimensions [%d x %d]" % (self.ntypes,self.ntypes)
        assert cohesion.shape == self.cohesion.shape, "Invalid cohesion array: required dimensions [%d x %d]" % (self.ntypes,self.ntypes)
        
        self.friction = np.array(friction)
        self.cohesion = np.array(cohesion)
    
    """
    Define bond stiffness between particles. 
    
    **Arguments**:
        -nStiff = [n x n] array where element [i,k] contains the normal-stiffness of bonds between particles of type i and k. If a single value is passed this is value is applied to all possible contacts. 
        -sStiff = [n x n] array where element [i,k] contains the shear-stiffness of bonds between particles of type i and k. If a single value is passed this is value is applied to all possible contacts.
        -tStrength = [n x n] array where element [i,k] contains the tensile strength of bonds between particles of type i and k. If a single value is passed this is value is applied to all possible contacts.
        -sStrength = [n x n] array where element [i,k] contains the shear strength of bonds between particles of type i and k (at zero normal stress). If a single value is passed this is value is applied to all possible contacts.
    """
    def setBonding( self, nStiff, sStiff, tStrength, sStrength ):
        if not isinstance(nStiff,list):
            nStiff = np.zeros( [self.ntypes,self.ntypes] ) + nStiff
        if not isinstance(sStiff,list):
            sStiff = np.zeros( [self.ntypes,self.ntypes] ) + sStiff
        if not isinstance(tStrength,list):
            tStrength = np.zeros( [self.ntypes,self.ntypes] ) + tStrength
        if not isinstance(sStrength,list):
            sStrength = np.zeros( [self.ntypes,self.ntypes] ) + sStrength
            
            
        assert nStiff.shape == self.nStiff.shape, "Invalid nStiff array: required dimensions [%d x %d]" % (self.ntypes,self.ntypes)
        assert sStiff.shape == self.sStiff.shape, "Invalid sStiff array: required dimensions [%d x %d]" % (self.ntypes,self.ntypes)
        assert tStrength.shape == self.tStrength.shape, "Invalid tStrength array: required dimensions [%d x %d]" % (self.ntypes,self.ntypes)
        assert sStrength.shape == self.sStrength.shape, "Invalid sStrength array: required dimensions [%d x %d]" % (self.ntypes,self.ntypes)
        
        self.nStiff = np.array(nStiff)
        self.sStiff = np.array(sStiff)
        self.tStrength = np.array(tStrength)
        self.sStrength = np.array(sStrength)
    
    """
    Internal function for writing material properties into a riceball model definition file.
    
    **Arguments**:
     -R = a ricepaper object used to interface with riceball. 
    """
    def _bindPropsToModel( self, R):
        #write material name as commment
        R.custom("\n*****************************")
        R.custom("** define material '%s' " % self.name )
        R.custom("*****************************")
        #define in order of radii for convenience
        for i in range(self.ntypes):
            #write node as comment
            R.custom("\n** particle type %d" % (i+1))
            
            #write particle params
            R.setRadius(i+1,self.radii[i])
            R.setDensity(i+1,self.density[i])
            R.setHertzian(i+1, calcShearModulus(self.E[i],self.v[i]), self.v[i])
            
        #write inter-particle params
        R.custom("\n** particle interactions")
        for i in range(self.ntypes):
            for n in range(self.ntypes):
                if n <= i: #only need to write interactions once (p1 interact p2 is the same as p2 interact p1)
                    R.setFrictionItc(i+1,n+1,self.friction[i,n])
                    R.setLinItc(i+1,n+1,self.nStiff[i,n],self.sStiff[i,n])
                    R.setCohesion(i+1,n+1,self.cohesion[i,n])
                    R.setBond(i+1,n+1, self.nStiff[i,n], self.sStiff[i,n], self.tStrength[i,n], self.sStrength[i,n] )
                    R.custom("") #newline