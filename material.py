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
        
        #placeholder for ricepaper interface
        self.R = None
        
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
    Deposit particles of this material type and let it settle until it reaches equilibrium. 
    
    **Arguments**:
        -n = the total number of particles to generate
        -l = the length of the domain to loosly fill with particles before gravity settling. 
        -h = the height of the domain to fill with particles before gravity settling.
    **Keywords**:
        -frequency = a list containing the proportion of each particle type to populate with. This list
                    is normalised such that it adds to 1. To give each layer different particle frequencies pass 
                    a list of length *nLayers* such that frequency[i] = [fN1,fN2,fN3]. 
        -pEff = the packing efficiency used to convert total particle area to domain area. Default is 0.6.
                This is used to calculate the number of particles to generate using n = pEff * l * h * pa, where pa is the 
                average particle area weighted by its frequency. Numbers closer to 1 will generate more particles.  
        -base = the width of the base. If this is larger than l, conical piles will be generated. Default is base = l. 
        -tStep = the timestep to use for the simulation. Default is 0.005 seconds. 
        -stepSize = the model time (in seconds) to run the model for before checking if it has reached stability. Default is 60. 
        -nLayers = the number of layers (different colours) to generate. Default is 4. 
        -pcolor = primary layer colour. Default is green.
        -acolor = alternate layer colour. Default is blue. 
        -verbose = True if output is printed to console. Default is true.
        -suppress = If True, the model is not actually executed (i.e. it is assumed that output has been
                   generated on a previous run. Useful for debugging quickly. Default is False. 
    **Returns**:
        - A list of riceball objects as the particles settle, the last of which *should* be in an 
          equillibrium state. 
    """
    def gravityDeposit(self,l,h,walltype,**kwds):
        
        #get and validate input
        verbose = kwds.get("verbose",True)
        nLayers = kwds.get("nlayers",4)
        pcolor = kwds.get("pcolor",'b')
        acolor = kwds.get("acolor",'g')
        stepSize = kwds.get("stepSize",60)
        
        base = kwds.get("base",l)
        assert base >= l, "Error - base must be greater or equal to l."
        
        frequency = np.array(kwds.get("frequency",[1/self.ntypes]*self.ntypes))
        if not isinstance(frequency[0],list): #frequency is the same for each layer 
            frequency = [frequency] * nLayers 
        assert len(frequency) == nLayers, "Error - particle frequencies need to be specified for each layer."
        for i,freq in enumerate(frequency):
            assert len(freq) == self.ntypes, "Error - a frequency has not been provided for all particle types."
            frequency[i] = freq / np.sum(freq) #normalise such that sum of frequencies = 1 for each layer
        
        self.walltype = walltype
        assert self.walltype <= self.ntypes, "Error - the specified particle type does not exist."
        
        pEff = kwds.get("pEff",0.6)
        assert pEff < 1 and pEff > 0, "Error - pEff must be between 0 and 1."
        
        tstep = kwds.get("tstep",0.005)
        assert tstep > 1e-9 and tstep < 1, "Error - stop passing stupid timesteps..."
        
        
        
        #start model
        name = "/gravity_settle"
        if base > l: #this is a pile, not a settle....
            name = "/gravity_pile"
        
        self.R = RicePaper(self.name + name)
        
        #setup and store bounds
        self.width = int(base+np.max(self.radii))
        self.height = int(2*h)
        self.depth = int(2.1*np.max(self.radii))
        self.R.setBounds(self.width,self.height,self.depth)
        
        #setup numerics
        self.R.setDamping(viscous_fraction=0.01) #1% of velocity is attenuated in each step. Helps model converge faster
        self.R.setNumericalProperties(timestep=tstep)
        self.R.setGravity((0,-9.8,0)) #set gravity

        #bind materials
        self._bindPropsToModel()
        
        #setup domain and generate walls
        domH = h*1.25
        if base > l: #reduce height for piles
            domH = base / 3 #crude but kinda works
        pR = self.radii[self.walltype-1] #wall particle radius
        self.R.genLine(( pR, pR, pR ),  #minX,minY,minZ
                  (base-pR,pR,pR), #maxX, maxY, maxZ
                  pR / 4,  #gap
                  self.walltype, self.walltype ) #particle type
        
        self.R.genLine(( pR, 2.25*pR, pR ),  #start point
                  ( pR,domH,pR), #end point
                  pR / 4,  #gap
                  self.walltype, self.walltype ) #particle type
                  
        self.R.genLine(( base-pR, 2.25*pR, pR ),  #start point
                  ( base-pR,domH,pR), #end point
                  pR / 4,  #gap
                  self.walltype, self.walltype ) #particle type
        
        #fix DOF on walls to keep them in place
        self.R.fixDOFAll(True,True,True) 
        
        #generate particles
        layerHeight = h * 1.5 / nLayers #height of each layer
        initHeight = 0.25 * h #start first layer at 25% of height
        idx = np.argsort(self.radii)[::-1] #generate balls from largest radius to smallest (optimises packing)
        for q in range(nLayers):
            if q % 2 == 0: #set primary color
                c = pcolor
            else: #set alternate colour
                c = acolor
            
            #calculate number of particles
            pa = np.sum(np.pi * self.radii**2 * frequency[q]) #weighted average particle area
            n = pEff * l * layerHeight / pa
            if verbose:
                print("Generating %d particles in layer %d" % (n,q))
            
            #set domain to generate particles in
            xmin = 0.5 * base - 0.5 * l + np.max(self.radii) * 2
            xmax = 0.5 * base + 0.5 * l - np.max(self.radii) * 2
            self.R.setDomain( xmin, xmax,  #xmin,xmax
                          initHeight+layerHeight*q, #layer base
                          initHeight+layerHeight*(q+1), #layer top
                          0, np.min(self.radii)) #zmin,zmax
            
            #generate particles
            for i in range(self.ntypes):
                self.R.genBalls( n * frequency[q][idx[i]], idx[i]+1, idx[i]+1, c )
        
        #do simulation
        avgK = 100000
        while avgK > 1: #while average particle kinetic energy > 1 J
            self.R.cycle(stepSize) #run for period of time (default is 1 min)
            self.R.execute(suppress=kwds.get("suppress",True))
            M = self.R.loadLastOutput() #load last output to check if model has reached stability
            id,K = M.getAttributes(["kin"]) #get average kinetic energy
            avgK = np.mean(K)
            print ("Average kinetic energy = %E" % avgK)
            
        print("Model equilibrated")
        return self.R #return ricepaper instance
    
    
    #private functions
    """
    Internal function for writing material properties into a riceball model definition file.
    
    **Arguments**:
     -R = a ricepaper object used to interface with riceball. 
    """
    def _bindPropsToModel( self ):
        #write material name as commment
        R = self.R
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