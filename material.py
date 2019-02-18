"""
A class for defining/storing the material properties of individual nodes and then generating node packings
to fill a domain with particles (at this stage only by gravity settling, though it would be straight-forward to 
implement a sphere-packing algorithm).

Sam Thiele 2018
"""

import numpy as np
import ricepaper
from ricepaper import RicePaper
from ricepaper.utils import *
import matplotlib.pyplot as plt

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
    Define frictional of contacts between particles.
    
    **Arguments**:
        -friction = [n x n] array where element [i,k] contains the friction coefficient between particles of type i and k If a single value is passed this is value is applied to all possible contacts.
    """
    def setFriction( self, friction ):
        if not isinstance(friction,list):
            friction = np.zeros( [self.ntypes,self.ntypes] ) + friction
    
        assert friction.shape == self.friction.shape, "Invalid friction array: required dimensions [%d x %d]" % (self.ntypes,self.ntypes)
        self.friction = np.array(friction)

    """
    Define cohesion of contacts between particles.
    
    **Arguments**:
        -cohesion = [n x n] array where element [i,k] contains the cohesion (in newtons) between particles of type i and k. If a single value is passed this is value is applied to all possible contacts.
    """
    def setCohesion( self, cohesion ):
        if not isinstance(cohesion,list):
            cohesion = np.zeros( [self.ntypes,self.ntypes] ) + cohesion
        assert cohesion.shape == self.cohesion.shape, "Invalid cohesion array: required dimensions [%d x %d]" % (self.ntypes,self.ntypes)
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
    Constructs bonds between particles based on their (current) location.
    
    **Keywords**:
     -radius = the radius to search for neighbouring particles to bond with. Defualt is 1.25 * the maximum particle radius. 
     -moments = true if bonds that transmit moments should be created. Default is False. 
    """
    def bond( self, **kwds ):
        
        r = kwds.get("radius",np.max(self.radii)*1.25)
        self.R.makeBonds(radius = r, moments = kwds.get("moments",False))
        
    """
    Deposit particles of this material type and let it settle until it reaches equilibrium. 
    
    **Arguments**:
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
        -nlayers = the number of layers (different colours) to generate. Default is 4. 
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
        
        frequency = kwds.get("frequency",[1/self.ntypes]*self.ntypes)
        if not isinstance(frequency[0],list): #frequency is the same for each layer 
            frequency = [frequency] * nLayers 
        for i,freq in enumerate(frequency):
            assert len(freq) == self.ntypes, "Error - a frequency has not been provided for all %d particle types." % self.ntypes
            frequency[i] = np.array(freq) / np.sum(freq) #normalise such that sum of frequencies = 1 for each layer
        
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
        
        #run simulation to stability
        self.R.executeToStability(stepSize,thresh=1.0,**kwds)
                
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
                

"""
Class for constructing mechanical tests , executing them (in serial or in parallel), processing 
and visualising the results. 
"""
class MechLab:
    
    """
    Utility class for storing test properties. 
    """
    class Test:
        """
        Init test properties class and populate basic properties. 
        
        **Arguments**:
         - testName = the name of this test. Should reflect the location of the output.
         - testType = the type of this test. Should be "shear" or "compression" or "extension".
         - lab = the MechLab object that created this test.        
         - R = the ricepaper object that implements this test. 
        """
        def __init__( self, testName, testType, lab, R ):
            self.name = testName
            self.type = testType
            self.lab = lab
            self.R = R
            self.steps = [] #execution steps will be stored here. 
            
            #define default test params (these will be overwritten when changed)\
            self.xvel = 0.5
            self.yval = 0.5
            self.confining = 0 #confining/normal stress applied to top platen during shear and compression tests
            self.initSteps = 0
            self.testSteps = 0
            self.initial_height = lab.upper - lab.lower
            self.initial_width = lab.init_xmax - lab.init_xmin
            
            self.models = None
            
            
        """
        Load test results and do stress/strain calcs etc. 
        """
        def load_results(self):
            if not self.models is None:
                return #no need to recalculate
            #calculate time at each step and xvelocity at each step
            self.time = np.linspace(0,(self.initSteps+self.testSteps)*self.lab.stepSize,self.initSteps+self.testSteps)
            
            #calculate shear strain at each step (from displacement)
            t = np.linspace(0,self.testSteps*self.lab.tstep,self.testSteps)
            self.shear_strain = np.append([0]*self.initSteps, 
                        np.rad2deg(np.arctan((t*self.xvel)/self.initial_height)))

            #loop through steps and calculate stresses and strains
            self.shear_stress = []
            self.normal_stress = []
            self.compressive_stress = []
            self.sample_stress = []
            self.sample_width = [] #change in sample width will be meaningless for shear tests.
            self.sample_height = []
            
            self.models = self.R.loadSteps(self.steps)
            for m in self.models:
                
                #***************************
                #get sample dimensions
                #***************************
                xmin,xmax,ymin,ymax = m.getBounds(dynamic=True)
                self.sample_width.append(xmax-xmin) #n.b. this will be invalid for shear-tests
                self.sample_height.append(ymax-ymin)
                
                #****************************
                #Estimate boundary stress
                #****************************
                pid,F = m.getAttributes(["force"],onlyFixed=True,gravity=np.array([0,0,0]))
                f_top = np.array([0.0,0.0,0.0])
                f_side = np.array([0.0,0.0,0.0])
                for i,f in enumerate(F):
                    #sum over top platen (for shear tests)
                    if m.pos[pid[i]][1]>self.lab.upper-np.max(self.lab.mat.radii)*4 and int(m.G.nodes[pid[i]]["MTYPE"]) != self.lab.mat.walltype:
                        f_top += f
                    #sum over right platen (for compression tests)
                    if m.pos[pid[i]][0] > xmax - np.max(self.lab.mat.radii)*4 and int(m.G.nodes[pid[i]]["MTYPE"]) == self.lab.mat.walltype:
                        f_side += f
                
                #store shear-box stresses
                self.shear_stress.append( -f_top[0] / self.lab.platen_area)
                self.normal_stress.append( f_top[1] / self.lab.platen_area)
                
                #store compression test stressess
                self.compressive_stress.append( f_side[0] / ((self.lab.upper - self.lab.lower)*2*np.max(self.lab.mat.radii) ))
                
                #***************************
                #Estimate sample stress
                #***************************
                #n.b. all tests are conducted on the international space station, so there is no gravity :-) 
                pid,S,vol = m.getAttributes(["stress","volume"],ignoreFixed=True,gravity=np.array([0,0,0]))
                S_sum = np.zeros((2,2))
                for i,s in enumerate(S):
                    S_sum += s*vol[i]
                if self.type == "shear":
                    self.sample_stress.append(S_sum / (self.lab.platen_area*(ymax-ymin)))
                else:
                    self.sample_stress.append(S_sum / ((xmax-xmin)*(ymax-ymin)*2*np.max(self.lab.mat.radii))) #will be wrong under shear as sample no longer rectangular
            
            #store data as np arrays
            self.shear_stress = np.array(self.shear_stress)
            self.normal_stress = np.array(self.normal_stress)
            self.sample_width = np.array(self.sample_width) #change in sample width will be meaningless for shear tests.
            self.sample_height = np.array(self.sample_height)
            self.compressive_stress = np.array(self.compressive_stress)
    """
    Initialise this test and specify initial platen positions
    
    **Arguments**:
     - mat = the NodeSet making the material being tested. A particle assembalage must have been
                  already constructed using this (with "gravityDeposit").
    
    **Keywords**:
     -lower = the height of the lower platen above the base of the material domain. Default is 10%. 
     -upper = the height of the top platen above the base of the material domain. Default is 75%.
     -tStep = the timestep to use for the simulation. Default is 0.005 seconds. 
     -stepSize = the time (in seconds) to run the model for at each increment. Default is 5 seconds.
     -initTime = number of seconds to run the model before shearing (to reach target normal stress). Default is 10 seconds.
     -testTime = time to run the experiment for (excluding the initialisation!). Default is 50 seconds. 
    """
    def __init__( self, mat, **kwds ):
        self.mat = mat
        assert mat.R != None, "Please generate a material (e.g. using gravityDeposit(...)) first."
        
        #bind material (for reference largely, and in case properties have been changed)
        self.mat._bindPropsToModel()
            
            
        #get platen positions
        self.upper = kwds.get("upper",mat.height / 2)
        self.lower = kwds.get("lower",mat.height / 20)
        assert self.upper - self.lower > np.max(mat.radii), "Error - Lower surface must be (significantly) below upper surface..."
        
        #get step properties
        self.stepSize = kwds.get("stepSize",5)
        self.initTime = kwds.get("initTime",10)
        self.testTime = kwds.get("testTime",50)
        
        #get timestep
        self.tstep = kwds.get("tstep",0.005)
        assert self.tstep > 1e-9 and self.tstep < 1, "Error - stop passing stupid timesteps..."
        
        #get initial sample bounds (n.b. the height is the total sample, not just the bit between the platens)
        m = mat.R.loadLastOutput()
        self.init_xmin,self.init_xmax,self.init_ymin,self.init_ymax = m.getBounds(dynamic=True)
        assert self.init_ymin < self.lower, "Error - lower platen does not intersect particles."
        assert self.init_ymax > self.upper, "Error - upper platen does not intersect particles."
        
        #calculate platen area
        self.platen_area = (self.init_xmax - self.init_xmin) * 2 * np.max(self.mat.radii)
        
        #initialise test queue
        self.tests = []
    
    """
    Create a shearTest and add it to the execution queue.  
    
    **Arguments**:
     - normal_stress = the confining (normal) stress to apply during the shear box test.
     - shear_velocity = the velocity of the upper plate of the shear experiment. The lower plate is fixed. 
    
    **Keywords**:
     -yvel = the maximum velocity of the upper platen in the y-directions (limits response to changes 
              in normal stress. Default is 0.5 m/sec. 
     -bond = True if bonds should be generated before executing the test. Default is True. 
    """
    def addShearTest(self, normal_stress, shear_velocity, **kwds  ):
        #clone ricepaper interface
        name = "%s/shear_test/%d_Pa" % (self.mat.name,normal_stress)
        R = self.mat.R.clone(name)
        
        #create new test
        T = MechLab.Test( name, "shear", self, R)
        
        #store test params
        T.confining = normal_stress
        T.xvel = shear_velocity
        T.yvel = kwds.get("yvel",0.5)
        
        ##############################################
        #remove wall and base particles from model
        ##############################################
        R.custom("\n\n**Delete walls created during gravity deposition**")
        walld=self.mat.radii[self.mat.walltype-1]*2
        
        #delete left wall
        R.setDomain( 0, self.init_xmin, 0, self.mat.height, 0,  self.mat.depth )
        R.custom("PRP 1 EROd %d\n" % self.mat.walltype)
        
        #delete right wall
        R.setDomain( self.init_xmax, self.mat.width, 0, self.mat.height, 0,  self.mat.depth )
        R.custom("PRP 1 EROd %d\n" % self.mat.walltype)
        
        #delete base
        R.setDomain( 0, self.mat.width, 0, self.init_ymin, 0,  self.mat.depth )
        R.custom("PRP 1 EROd %d\n" % self.mat.walltype)
        
        ##############################
        #turn of gravity and bond
        ##############################
        R.setGravity((0,0,0))
        R.setNumericalProperties(timestep=self.tstep)
        if kwds.get("bond",True):
            R.makeBonds(np.max(self.mat.radii)*1.25,moments=kwds.get("moments",False))
        
        ##############################
        #setup shear box
        ##############################
        R.custom("\n\n********************")
        R.custom("**Setup shear test**")
        R.custom("********************")
        R.custom("Wall Pos %.3f %.3f" % (self.upper,self.lower)) #set position of platens
        R.custom("Wall Nstress %.3f" % (normal_stress)) #set normal stress
        R.custom("Wall Gain %E" % (1e-5)) #set large number for servo gain (we use velocity as the limit)
        R.custom("Wall Vel %.3f" % T.yvel) #set maximum normal velocity 
        
        #################################################################
        #setup init phase (compress until we reach desired normal stress)
        #################################################################
        R.custom("Wall Xvel 0 0") #zero shear velocity
        T.initSteps = int(self.initTime / self.stepSize)
        for i in range(T.initSteps):
            T.steps.append( R.cycle(self.stepSize) )
        
        ##################
        #setup shear phase
        ##################
        R.custom("Wall Xvel %.3f 0" % (shear_velocity)) #set shear velocity
        T.testSteps = int(self.testTime/self.stepSize)
        for i in range(T.testSteps):
            T.steps.append( R.cycle(self.stepSize) )
            
        #store test and return 
        self.tests.append( T )
        
        return T
    
    """
    Plot the results of the shearbox experiments.
    
    **Arguments**:
     - detailed = If true, detailed stress plots for each sample are plotted, as well as the overall mohr-coloumb failure envelope.
                  If false, only the failure envelope is plotted. 
    """
    def plotShearResults( self, detailed = True, **kwds ):
        maxS = [] #maximum shear stress
        NS = [] #applied normal stresses (75th percentile, though should be quite constant)...
        for T in self.tests:
            if T.type != "shear":
                continue
            
            #make sure results have been loaded
            T.load_results()
            
            #extract maxima
            maxS.append(np.max(T.shear_stress))
            NS.append(np.percentile(T.normal_stress,75))
            
            #setup fig
            fig,ax = plt.subplots(1,3,figsize=(20,5))

            #plot time-strain curve
            #time = np.array(list(range(0,len(strain))))*kwds["stepSize"]
            ax[0].plot(T.time,T.shear_stress/1e6,color="blue",label="shear stress")
            ax[0].plot(T.time,T.normal_stress/1e6,color="red",label="normal stress")
            ax[0].axhline(NS[-1]/1e6,color='red')
            ax[0].set_title("Platen Stresses")
            ax[0].set_xlabel("Time (seconds)")
            ax[0].set_ylabel("Stress (MPa)")
            ax[0].legend()

            #get stress in sample
            smax = []
            nmax = []
            for s in T.sample_stress:
                #calculate principal stresses
                eigval, eigvec = np.linalg.eig(s)
                idx = eigval.argsort()[::-1]
                eigval = eigval[idx]
                eigvec = eigvec[:,idx]

                smax.append(0.5 * np.abs(eigval[0]-eigval[1]))
                nmax.append(0.5 * np.abs(eigval[0] + eigval[1]))
                
            ax[1].plot(T.time,np.array(smax)/1e6,color='blue',label="shear stress")
            ax[1].plot(T.time,np.array(nmax)/1e6,color='red', label="normal stress")
            ax[1].set_title("Sample Stresses")
            ax[1].set_xlabel("Time (seconds)")
            ax[1].set_ylabel("Stress (MPa)")
            ax[1].legend()

            #plot broken bonds
            x = list(range(0,len(T.models)))
            tensile = []
            shear = []
            for m in T.models:
                tcount = 0
                scount = 0
                for b in m.brk:
                    if b[0] == 'Tensile Failure':
                        tcount+=1
                    elif b[0] == 'Shear Failure':
                        scount+=1
                    elif b[0] == 'Tens-shear Failure':
                        scount+=1
                        tcount+=1
                    else:
                        print(b)
                        assert False
                tensile.append(tcount)
                shear.append(scount)
            ax[2].plot(T.time,tensile,color='b',label="tensile")
            ax[2].set_ylabel("tensile bond failures",color="b")
            ax2 = ax[2].twinx()
            ax2.plot(T.time,shear,color='g',label="shear")
            ax2.set_ylabel("shear bond failures",color="g")
            ax[2].set_title("Broken Bonds")
            ax[2].set_xlabel("Time (seconds)")
            fig.suptitle("Shear test at %d MPa normal stress" % (T.confining/1e6))
            fig.show()

        #plot Coloumb failure line
        if len(NS) > 1: #can't do for only one test..
            plt.figure()
            x = np.array(NS) / 1e6
            y = np.array(maxS) / 1e6
            plt.scatter(x,y,color='b')
            k,m = np.polyfit(x,y,1)
            plt.plot([0,np.max(x)+5],[m,(np.max(x)+5)*k+m],color='k')
            plt.text(0.95,0.1,"cohesion=%.2f MPa" % m,transform=plt.gca().transAxes,horizontalalignment='right')
            plt.text(0.95,0.05,"friction angle=%.2f°" % np.rad2deg(np.arctan(k)),transform=plt.gca().transAxes,horizontalalignment='right')

            plt.xlim(0,np.max(x)+5)
            plt.ylim(0,np.max(y)+5)
            plt.ylabel("Shear Stress (MPa)")
            plt.xlabel("Normal Stress (MPa)")
            plt.show()

    def writeToVTK(self):
        #calculate strain and export to VTK
        for T in self.tests:
            T.load_results()
            
            #calc strain and export to vtk
            for t in range(1,len(T.models)):
                T.models[t].computeStrain2D(T.models[t-1]) #compute strain
                T.models[t].averageAttr("stress",n=5) #average stress field
                T.models[t].averageAttr("strain",n=2)
                T.models[t].writeVTK("%s/STEP_%03d.vtk" % (T.name,t+1)) #export 
        
        
        
    """
    Run all tests.
    
    **Arguments**:
     -multiThreaded = True if tests should be executed in separate threads. Default is true. 
    
    **Keywords**: All keywords are passed to RicePaper execute(...). 
    """
    def runTests(self, multiThreaded=True, **kwds):
        #gather list
        q = []
        for T in self.tests:
            if not multiThreaded:
                T.R.execute(**kwds)
            else:
                q.append(T.R)
                
        if multiThreaded:
            ricepaper.multiThreadExecute(q,**kwds)                
