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
        -l = the length of the domain to fill with particles
        -h = the approximate height of particles when stacked. This is used to calculate the number
             of particles to generate using n = 0.75 * l * h * pa, where pa is the average particle 
             area weighted by its frequency. 
    **Keywords**:
        -walltype = the id of the particle type to use as the wall. Otherwise a soft, friction free
                    boundary is used.
        -frequency = a list containing the proportion of each particle type to populate with. This list
                    is normalised such that it adds to 1. 
        -pEff = the packing efficiency used to convert total particle area to domain area. Default is 0.75.
                Numbers closer to 1 will generate more particles. 
        -tStep = the timestep to use for the simulation. Default is 0.005 seconds. 
        -stepSize = the time (in seconds) to run the model for before checking if it has reached stability. Default is 60. 
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
    def gravityDeposit(self,l,h,**kwds):
        
        #validate input
        frequency = np.array(kwds.get("frequency",[1/self.ntypes]*self.ntypes))
        assert len(frequency) == self.ntypes, "Error - a frequency has not been provided for all particle types."
        frequency = frequency / np.sum(frequency) #normalise such that sum = 1
        
        self.walltype = kwds.get("walltype",None)
        if not self.walltype is None:
            assert self.walltype <= self.ntypes, "Error - the specified particle type does not exist."
        
        pEff = kwds.get("pEff",0.75)
        assert pEff < 1 and pEff > 0, "Error - pEff must be between 0 and 1."
        
        tstep = kwds.get("tstep",0.005)
        assert tstep > 1e-9 and tstep < 1, "Error - stop passing stupid timesteps..."
        
        verbose = kwds.get("verbose",True)
        nLayers = kwds.get("nlayers",4)
        pcolor = kwds.get("pcolor",'b')
        acolor = kwds.get("acolor",'g')
        stepSize = kwds.get("stepSize",60)
        
        #calculate number of particles
        pa = np.sum(np.pi * self.radii**2 * frequency) #weighted average particle area
        n = pEff * l * h / pa
        if verbose:
            print("Generating %d particles (excluding walls)" % n)
        
        #setup wall type particle if need be
        if self.walltype is None:
            walltype = len(self.radii) + 2 #plus two rather than 1 as indices for materials start with 1
            self.ntypes += 1
            self.radii.append( np.min(self.radii) )
            self.density.append( np.min(self.density) ) #density doesn't really matter here tbh
            self.E.append( 1e6 ) #use really soft walls to damp bounces
            self.v.append( 0.1 ) #low poissons ratio to damp bounces
            self.friction = np.hstack(self.friction,np.zeros(self.ntypes-1,1))
            self.friction = np.vstack(self.friction, np.zeros(self.ntypes))
            self.cohesion = np.hstack(self.cohesion,np.zeros(self.ntypes-1,1))
            self.cohesion = np.vstack(self.cohesion, np.zeros(self.ntypes))
            self.nStiff = np.hstack(self.nStiff,np.zeros(self.ntypes-1,1))
            self.nStiff = np.vstack(self.nStiff, np.zeros(self.ntypes))
            self.tStrength = np.hstack(self.tStrength,np.zeros(self.ntypes-1,1))
            self.tStrength = np.vstack(self.tStrength, np.zeros(self.ntypes))
            self.sStrength = np.hstack(self.sStrength,np.zeros(self.ntypes-1,1))
            self.sStrength = np.vstack(self.sStrength, np.zeros(self.ntypes))
        
        #start model
        self.R = RicePaper(self.name + "/gravity_settle")
        
        #setup and store bounds
        self.width = int(l+np.max(self.radii))
        self.height = int(2*h)
        self.depth = int(2.1*np.max(self.radii))
        self.R.setBounds(self.width,self.height,self.depth)
        
        #setup numerics
        self.R.setDamping(viscous_fraction=0.01) #1% of velocity is attenuated in each step. Helps model converge faster
        self.R.setNumericalProperties(timestep=tstep)
        self.R.setGravity((0,-9.8,0)) #set gravity

        #bind materials
        self._bindPropsToModel( self.R )
        
        #setup domain and generate walls
        pR = self.radii[self.walltype-1] #wall particle radius
        self.R.genLine(( pR, pR, pR ),  #minX,minY,minZ
                  (l-pR,pR,pR), #maxX, maxY, maxZ
                  pR / 4,  #gap
                  self.walltype, self.walltype ) #particle type
        
        self.R.genLine(( pR, 2.25*pR, pR ),  #start point
                  ( pR,h*1.25,pR), #end point
                  pR / 4,  #gap
                  self.walltype, self.walltype ) #particle type
                  
        self.R.genLine(( l-pR, 2.25*pR, pR ),  #start point
                  ( l-pR,h*1.25,pR), #end point
                  pR / 4,  #gap
                  self.walltype, self.walltype ) #particle type
        
        #fix DOF on walls to keep them in place
        self.R.fixDOFAll(True,True,True) 
        
        #generate particles
        layerHeight = h * 1.5 / nLayers #height of each layer
        initHeight = 0.25 * h #start first layer at 25% of height
        for i in range(nLayers):
            if i % 2 == 0: #set primary color
                c = pcolor
            else: #set alternate colour
                c = acolor
            
            #set domain to generate particles in
            self.R.setDomain( np.max(self.radii) * 2, l - np.max(self.radii) * 2,  #xmin,xmax
                          initHeight+layerHeight*i, #layer base
                          initHeight+layerHeight*(i+1), #layer top
                          0, np.min(self.radii)) #zmin,zmax
            
            #generate particles
            for i,f in enumerate(frequency):
                self.R.genBalls( (n/nLayers) * f, i+1, i+1, c )
        
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
    
    """
    Do a pure shear test to measure the bulk cohesion of this material. 

    **Arguments**:
        -lower = the height of the base of the domain to shear
        -upper = the height of the top of the domain to shear
        -velocity = the velocity of the upper plate (lower plate velocity = 0). 
        
    **Keywords**:
        -normalStress = the normal (confining) stress applied during the shear experiment. Default is 0. 
        -tStep = the timestep to use for the simulation. Default is 0.005 seconds. 
        -stepSize = the time (in seconds) to run the model for before writing output. Default is 1.
        -nSteps = the number of steps to run the model for. Default is 250. 
        -verbose = True if output is printed to console. Default is true.
        -suppress = If True, the model is not actually executed (i.e. it is assumed that output has been
                   generated on a previous run. Useful for debugging quickly. Default is False. 
    **Returns**:
        -stress = the average pressure on the upper domain of the shear box at each step.
        -strain = the bulk shear strain at each step.
    """
    def shearTest(self,lower,upper,velocity,**kwds):
            assert self.R != None, "Please generate a material (e.g. using gravityDeposit(...)) first."
            assert upper - lower > np.max(self.radii), "Error - Lower surface must be (significantly) below upper surface..."
            
            #get args
            tstep = kwds.get("tstep",0.005)
            assert tstep > 1e-9 and tstep < 1, "Error - stop passing stupid timesteps..."
            
            verbose = kwds.get("verbose",True)
            stepSize = kwds.get("stepSize",1)
            nSteps = kwds.get("nSteps",250)
            normalStress = kwds.get("normalStress",0)
            
            #clone state so we don't modify it
            R2 = self.R.clone("%s/shear_test" % self.name)
            m = self.R.loadLastOutput()
            minx,maxx,miny,maxy = m.getBounds()
            
            #remove wall type particles
            R2.custom("**Delete walls created during gravity deposition**")
            R2.setDomain( minx, maxx, miny, maxy, 0,  self.depth )
            R2.custom("PRP 1 EROd %d\n" % self.walltype)
            
            #turn of gravity
            #R2.setGravity((0,0,0))
            R2.setNumericalProperties(timestep=tstep)

            #setup shear test
            R2.custom("**Setup shear test**")
            R2.custom("Wall Pos %.3f %.3f" % (upper,lower)) #set position of platens
            R2.custom("Wall Xvel %.3f 0" % (velocity)) #set shear velocity
            R2.custom("Wall Nstress %.3f" % (-normalStress)) #set normal stress
            #R2.custom("Wall Gain %.3f" % (1.0) ) #set gain of servo responsible for maintining normal stress
            #run test
            steps = []
            for i in range(nSteps):
                steps.append(R2.cycle(stepSize))
            R2.execute(suppress=kwds.get("suppress",False))
            
            #load steps
            M = R2.loadSteps(steps)
            
            #calculate shear strain at each step (from displacement)
            t = np.linspace(stepSize,stepSize*nSteps,nSteps) #number of seconds of shearing applied to each model
            strain = np.rad2deg(np.arctan((t*velocity)/(upper-lower)))
            
            #estimate overall shear stress at each step (from forces on fixed particles)
            stress = []
            for m in M:
                pid,F = m.getAttributes(["force"], onlyFixed=True, gravity=np.array([0,0,0])) 
                fsum = np.array([0,0,0]) #sum of all forces on contacts
                for Flist in F:
                    fsum =  fsum + np.sum(Flist,axis=0)
                stress.append(-fsum / (self.width * self.depth)) #append total force divided by area over which it is applied

            #load output
            return np.array(stress),np.array(strain), M
        #private functions
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