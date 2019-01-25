"""
A simple interface for generating and running riceball models. 

Sam Thiele 2018
"""
import sys, os
import numpy as np
import subprocess
import shutil

from .reader import RiceBall

"""
A wrapper class for RiceBall. Manages the creation of model definition files, gives them to RiceBall and
loads the output into python (using RiceBallReader) for analysis.

Sam Thiele 2018
"""
class RicePaper:

    """
    Creates a RicePaper instance.
    
    
    **Arguments**:
    -dir = the directory in which RiceBall files (both input and output) will be written.
    """
    def __init__(self,dir):
        self.dir = dir #store output directory
        self.lines = [] #list of RiceBall commands
        self.bounds = [15000,6000,100] #default bounds - xmax, ymax, zmax
        self.nballs = 50000 #default to max 10000 balls 
        self.nboxes = 9000
        self.nwall = 0
        self.plotScale = 40 #length (model units) of 1 inch in figure
        self.forceScale = 5e12
        self.name = "RiceBallSimulation"
        self.radii = {1 : 10 } #default radii
        self.density = {1 : 1000 } #default density
        self.hertz = {1 : (2.9e9,0.2)}
        self.seed = 1 #seed for random number generators
        self.step = 0 #index for counting timesteps
        self.file = 0 #index for counting files created
        self.restart=None
        self.history = [] #list of all commands that lead to this model (rather than just from the last execute)
    def getAllLines(self):
        out = []
        out.append(self._startLines()) #write start lines
        out += self.lines
        return out
        
    """
    Create a model definition file and pass it to the ricebal executable for 
    execution.
    
    **Arguments**:
    -saveState = save the model state at the end of the simulation. Default is True.
    -returnLog = load and return the ricebal log after execution. Default is False.
    -restart = If not None, a restart file is passed to riceball to restart a saved simulation.
    -supress = If true, the model defenition file is written, but riceball is not invoked.
               This is useful if the model files have already been generated and you want to 
               fool ricepaper into thinking they have been freshly minted.
    **Returns**: 0 if riceball executes succesfully, or the log if returnLog is True.
    """
    def execute(self,saveState=True,returnLog=False,restart=None,suppress=False):
        #check output directory exists
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        #cwd = os.getcwd()
        #os.chdir(self.dir)
        
        #check for custom restart file
        if not restart is None:
             self.restart = restart
        if not self.restart is None:
            assert os.path.exists(os.path.join(self.dir,self.restart)), "Invalid (non-existant) restart file."
        
        #get lines
        lines = self.getAllLines()
        
        #update history
        if len(self.history) == 0:
            self.history += lines #include start lines
        else: #this is a restart - just include new lines
            self.history += self.lines

        #add trailing lines
        if (saveState):
            lines.append("SAVE state_%d.sav\n" % self.file)
        lines.append("stop\n")
        
        #write file
        f = open(os.path.join(self.dir,"trubalw.dat"),'w')
        f.writelines(lines)
        f.close()
        
        #run executable
        state = 0
        if not suppress:
            cd = os.getcwd()
            state = subprocess.call(["ricebal5.8a"],shell=True,cwd=self.dir) #call riceball
            os.chdir(cd)
            
        self.clear() #clear lines that have now been run
        self.restart = "state_%d.sav" % self.file #store link to restart file

        if returnLog:
            if os.path.exists("trubalw.out"):
                f=open("trubalw.out","r")
                lines = f.readlines()
                f.close()
                return lines #echo lines
        return state #return state of program (0 for success)
    """
    Clear the cached commands in this RicePaper object. This is also called after every execute() command.
    """
    def clear(self):
        self.lines = []  #clear lines
    
    """
    Clones this ricepaper object and associated riceball project.
    
    **Arguments**
    -name = the name of the new riceball project. Must be different to the previous to avoid overwriting it...
    
    **Returns**:
    -a duplicated ricepaper object.
    """
    def clone(self, newName):
        assert self.name != newName, "Error - cloned ricepaper object must have a different name to the old one."
        
        #create new riceball object with this name
        clone=RicePaper(newName)
        
        #copy across properties (deep copy)
        clone.name = self.name + "_clone"
        clone.lines = list(self.lines)
        clone.bounds = list(self.bounds)
        clone.nballs = self.nballs
        clone.nboxes = self.nboxes
        clone.nwall = self.nwall
        clone.plotScale = self.plotScale
        clone.forceScale = self.forceScale
        clone.radii = self.radii.copy()
        clone.density = self.radii.copy()
        clone.hertz = self.hertz.copy()
        clone.seed = self.seed
        clone.step = self.step
        clone.file = self.file
        clone.restart = self.restart
        clone.history = list(self.history)
        
        if not clone.restart is None:
            #create new directory and copy across restart file
            if not os.path.exists(clone.dir):
                os.makedirs(clone.dir)
            if os.path.exists(os.path.join(clone.dir,self.restart)):
                os.remove(os.path.join(clone.dir,self.restart)) #remove pre-existing restart file
            shutil.copy2(os.path.join(self.dir,self.restart),clone.dir)
        
        return clone
    """
    Loads the last output file created the last time execute() was run. 
    """
    def loadLastOutput(self):
        cwd = os.getcwd()
        os.chdir(self.dir) #move into output dir
        fname = "Step%03d.out" % self.file
        assert os.path.exists(fname), "Error: no ouput file %s found" % fname
        model = RiceBall(fname,radii=self.radii)
        os.chdir(cwd)
        return model
        
    """
    Loads the output associated with a particular step (as returned by the cycle() command) or list of steps.
    
    **Returns**:
    -The requested model or list of models as RiceBall classes.
    """
    def loadSteps(self,steps):
        out = []
        if isinstance(steps,int):
            steps = [steps]
        
        cwd = os.getcwd()
        os.chdir(self.dir) #move into output dir
        
        for s in steps:
            fname = "Step%03d.out" % s
            assert os.path.exists(fname), "Error: no ouput file %s found" % fname
            out.append(RiceBall(fname,radii=self.radii))
        os.chdir(cwd)
        
        if len(out) == 1:
            return out[0]
        else:
            return out
    #these funcitions all change "stored" variables (that we might care about after).
    #stored vars automatically get "prepended" to any model defenition files that we might write.
    
    """
    Set the name of this simulation. This controls the output folder name.
    """
    def setName(self,name):
        self.name = name
    
    """
    Set the bounds of the simulation domain. Note that this only works if called prior to the first execute() command.
    """
    def setBounds(self,xmax,ymax,zmax, updateNBoxes=True, updateScale=True):
        assert isinstance(xmax,int), "xmax must be an integer."
        assert isinstance(ymax,int), "ymax must be an integer."
        assert isinstance(zmax,int), "zmax must be an integer."
        
        #update bounds
        self.bounds[0] = xmax,
        self.bounds[1] = ymax,
        self.bounds[2] = zmax
        
        #update nboxes
        if updateNBoxes:
            self.nboxes = (xmax / zmax) * (ymax / zmax)
        
        #update plotting scale
        if updateScale:
            #scale width to A4 sheet (total width = 8 inches)
            self.plotScale = self.xmax / 8
     
    """
    Set the maximum number of balls that can be created in a this simulation. Note that this only works if called prior to the first execute() command.
    """
    def setMaxBalls(self,nballs):
        assert isinstance(nballs,int), "nballs must be an integer."
        self.nballs = nballs
    
    """
    Set the random seed for reproducable results. Note that this only works if called prior to the first execute() command.
    """
    def setSeed(self,n):
        _validateIdx(n)
        self.seed = n
        self.lines.append("gen seed %d\n" % self.seed)
    
    """
    Define a particle radius class.
    
    **Arguments**:
    -index = the ID of the particle radius calss to define
    -value = the value of the particle radius
    """
    def setRadius(self, index, value ):
        _validateIdx(index)
        self.radii[index] = value
        #self.lines.append(self._defineRadii()) #push to buffer
        self.lines.append("RAD %.2f %d\n" % (value, index))
    """
    Define particle density.
    
    **Arguments**:
    -index = the ID of the material to set the density for
    -density = the density to set 
    """
    def setDensity(self, index, density ):
        _validateIdx(index)
        self.density[index] = density
        #self.lines.append(self._defineDensity()) #push to buffer
        self.lines.append("DENS %d %d\n" % (density,index))
    """
    Define hertzian material properties
    
    **Arguments**:
    -index = the ID of the material to set Hertzian properties for.
    -shearModulus = the shear modulus of the Hertizan bonds.
    -poisson = the poisson ratio of the Hertzian bonds.
    """
    def setHertzian(self, index, shearModulus, poisson ):
        _validateIdx(index)
        self.hertz[index] = (shearModulus,poisson)
        #self.lines.append(self._defineHertzProperties()) #push to buffer
        self.lines.append("HERTZ %E %f %d\n" % (shearModulus,poisson,index))
    
    
    """
    Set the linear interaction properties between particle types (if used).
    
    **Arguments**:
    -index1 = particle type 1 of the interaction
    -index2 = particle type 2 of teh interaction
    -normalStiffness = interaction normal stiffness
    -shearStiffness  = interaction shear stiffness
    """
    #these functions directly map to the output
    def setLinItc(self, index1, index2, normalStiffness, shearStiffness):
        _validateIdx(index1)
        _validateIdx(index2)
        
        #write
        self.lines.append("SH %E %d %d\nNO %E %d %d\n" % (shearStiffness,index1,index2,normalStiffness,index1,index2))
    """
    Set frictional properties for interactions between particle types.
    
    **Arguments**:
    -index1 = the index of the first particle type in the interaction
    -index2 = the index of the second particle type in the interaction.
    -friction_coeff = the friction coefficient (=tan(friction angle)).
    """
    
    """
    Set the properties of any bonds created between the specified particle types. N.B. this does not create any bonds! (use makeBond for that).
    
    **Arguments**:
    -index1 = particle type 1 of the interaction
    -index2 = particle type 2 of teh interaction
    -normalStiffness = bond normal stiffness (Pa)
    -shearStiffness  = bond shear stiffness (Pa)
    -tStrength = tensile strength of the bond (Pa)
    -sStrength = shear strength of the bond (at zero normal stress; Pa). The friction coefficient is used to linearly increase sstrength with increasing normal stress.
    """
    def setBond( self, index1, index2, normalStiffness, shearStiffness, tStrength, sStrength ):
        _validateIdx(index1)
        _validateIdx(index2)
        self.lines.append("BONd %.2E %.2E %.2E %.2E %d %d\n" % (normalStiffness,shearStiffness,tStrength,sStrength,index1,index2))
    
    def setFrictionItc(self, index1, index2, friction_coeff):
        _validateIdx(index1)
        _validateIdx(index2)
        self.lines.append("FRIC %f %d %d\n" % (friction_coeff,index1,index2))
    
    """
    Set the cohesion between two particle types.
    
    **Arguments**:
    -index1 = the index of the first particle type in the interaction.
    -index2 = the index of the second particle type in the interaction.
    -cohesion = the cohesion between the specified particle types.
    
    """
    def setCohesion(self, index1, index2, cohesion):
        _validateIdx(index1)
        _validateIdx(index2)
        self.lines.append("COH %f %d %d\n" % (cohesion,index1,index2))
    
    """
    Set numerical damping properties.
    
    **Arguments**:
    -acceleration_fraction = the fraction of acceleration removed per timestep. Set to None (default) to disable.
    -viscous_fraction = the fraction of velocity to remove per timestep. Set to None to disable. Default is 0.01.
    """
    def setDamping(self,acceleration_fraction=None,viscous_fraction=0.01):
        useAcc = 0
        useVisc = 0
        if acceleration_fraction is None:
            useAcc = 1
            acceleration_fraction = 0.01
        if viscous_fraction is None:
            useVisc = 1
            viscous_fraction = 0.01
        
        self.lines.append("DAMP %f %f %d %d\n" % (acceleration_fraction,viscous_fraction,useAcc,useVisc))
        
        #turn off other damping
        self.lines.append("CDMP 0.7 0.7 1 1\nHDMP 0.2 1.0 1 1\n") #todo - expose these other types of damping??
    
    """
    Set numerical properties for this simulation.
    
    **Arguments**:
    -timestep = the timestep interval (per second). If this is too large, the simulation will be unstable. Default is 0.01.
    -frac = something important? Default is 0.1.
    -tol = something else important? Default is 3.5.
    -xres = something even more important? Default is 1.0.
    """
    def setNumericalProperties(self,timestep=1e-2, frac=0.1,tol=3.5,xres=1.0):
        self.lines.append("FRAC %f\nTOL %f\nXRES %f\nTS %f\n" % (frac,tol,xres,timestep))
    
    """
    Create a line of balls.
    
    **Arguments**:
    -start = the start point for the line as (x,y,z) tuple.
    -end = the end point for the line as (x,y,z) tuple.
    -gap = the distance between each ball in the line.
    -size_class = the size class of these balls.
    -surface_class = the surface/material class of the balls.
    """
    def genLine(self,start,end,gap,size_class,surface_class):
        #get ball radii
        assert size_class in self.radii, "Error - you need to define ball radius before generating."
        assert len(start) == 3 and len(end) ==3, "Error - start and end should be 3D vectors."
        r = self.radii[size_class]
        
        #compute effective diameter (including gap)
        D = r * 2 + gap
        
        #computer number of balls
        n = np.linalg.norm(np.array(start) - np.array(end)) / D
        
        #write
        self.lines.append("LIN %d %f %f %f %f %f %f %d %d\n" % (n,start[0],start[1],start[2],end[0],end[1],end[2],size_class,surface_class))
    
    """
    Play god and change gravity
    
    **Arguments**:
    -gravity: vector as (x,y,z) describing gravitational acceleration.
    """
    def setGravity( self, gravity ):
        assert len(gravity) == 3, "Gravity must be a 3D vector" #gravity is 3D
        self.lines.append("GRAV %f %f %f\n" % gravity)
    
    """
    Fix (or release) the degrees of freedom for all particles in the domain.
    """
    def fixDOFAll( self, tx = False, ty = False, tz = False, rx = True, ry = True, rz = True):
        self.lines.append("FIX %d %d %d %d %d %d all\n" % (tx,ty,tz,rx,ry,rz))
    
    #def fixDOF( self, pos, tx = False, ty = False, tz = False, rx = True, ry = True ,rz = True,e=0.1):
    #    self.setDomain(
    """
    Set the domain in whicn new particles are created or particle properties are changed.
    """
    def setDomain( self, xmin, xmax, ymin, ymax, zmin, zmax ):
        #set domain for creating particles
        self.lines.append("DOM %f %f %f %f %f %f\n" % (xmin,xmax,ymin,ymax,zmin,zmax))
        
        #also set "active" polygon so that this function can also be used for changing properties
        self.lines.append("POLy 1 %d %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n" % (zmin,zmax,
                                                                            xmin,ymin,
                                                                            xmin,ymax,
                                                                            xmax,ymax,
                                                                            xmax,ymin))
    """
    Sets the domain to a specific position (for selecting individual particles).
    
    **Arguments**:
    -pos = an (x,y) tuple specifying the location of the particle to selecting
    -e = the half-width of the domain box to create. Should be > 0 and less than the particle radius. Default is 0.1.
    """
    def setDomainToPos(self, pos, e=0.1):
        self.setDomain(pos[0]-e,pos[0]+e,pos[1]-e,pos[1]+e,0,100)
        
    """
    Add balls to the simulation in the active domain (see setDomain(...)).
    
    **Arguments**:
    -nballs = the number of balls to add.
    -shape_class = the radius class of the balls to add.
    -surface_class = the surface/material class of the balls to add.
    -color = the color of the balls to add. Can be integer < 9 as per RiceBall or matplot lib colors ('r','g','b','y','w','k','grey','c' or 'v').
    """
    def genBalls( self, nballs, shape_class = 1, surface_class = 1, color=0):
    
        #parse matplotlib color style strings
        if color == "g" or color == "green":
            color = 1
        if color == "y" or color == "yellow":
            color = 2
        if color == "r" or color == "red":
            color = 3
        if color == "w" or color == "white":
            color = 4
        if color == "k" or color == "black":
            color = 5
        if color == "grey" or color == "gray":
            color = 6
        if color == "b" or color == "blue":
            color = 7
        if color == "c" or color == "cyan":
            color = 8
        if color == "v" or color == "violet":
            color = 9
        
        if not isinstance(color,int):
            color = 0
        
        self.lines.append("GEN %d %d %d %d\n" % (nballs,shape_class,surface_class,color))
    
    """
    Write a cycle command to the model definition file. This will cause the model to be run for the specified number of timesteps when execute() is called.
    
    **Arguments**:
    -ncycles = the number of cycles to run for.
    -storeOutput = If true, output files are written after the cycles. Default is True.
    -createFig = If true, output figures (postscript files) are written after the cycles. Default is False.
    
    **Returns**:
    -the stepID for this cycle. These can be used for retrieving output (if output has been saved). 
    """
    def cycle(self, ncycles, storeOutput=True, createFig = False):
        #increment trackers of number of steps/number of cycle calls
        self.step += ncycles
        self.file += 1
        
        #open new bondfile
        if storeOutput:
            self.lines.append("BFIL 1 Step%03d.bnd\n" % self.file)
        
        #actual cycle command
        self.lines.append("C %d\n" % ncycles)
        
        #store output?
        if storeOutput:
            self.lines.append("BFIL 0\n") #close bond file
            self.lines.append("PR Step%03d.out G I S B C\n" % self.file)
        if createFig:
            self.lines.append("PL Step%03d_B.ps BO D\n" % self.file)
            self.lines.append("PL Step%03d_F.ps BO F\n" % self.file)
            self.lines.append("PL Step%03d_D.ps BO RDIS\n" % self.file)
            
        return self.file
    
    """
    Delete balls from the model domain.
    
    **Arguments**:
    -balls = a list of ball ids to remove
    -pos = a dictionary matching each ball id (key) to its position (value). 
    -e = the half-width of the selection box used to pick balls. Should be > 0 and < the particle radius. 
    """
    def delBalls( self, balls, pos, e=0.1):
        for b in balls:
            #set domain to only include ball center
            p = pos[b]
            #self.setDomain(p[0]-0.1,p[0]+0.1,p[1]-0.1,p[1]+0.1,0,100)
            #self.lines.append("POLy 1 0 100 %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n" % (p[0]-e,p[1]-e,
            #                                                                p[0]-e,p[1]+e,
            #                                                                p[0]+e,p[1]+e,
            #                                                                p[0]+e,p[1]-e))
            self.setDomainToPos(p,e)
            
            #remove ball
            self.lines.append('PRP 1 EROd x\n')
            
    """
    Make bond between two particles. Note that these particles must be neighbours or 
    unexpected bonds will be created within the bounding-box that joins the two particles.
    
    **Arguments**:
    -pos1 = the position of the first particle to bond.
    -pos2 = the position of the second particle to bond.
    """
    def makeBond(self,pos1,pos2,e=0.1):
        #set bounds
        x = np.array([pos1[0],pos2[0]])
        y = np.array([pos1[1],pos2[1]])
        self.lines.append("DOB %d,%d,%d,%d,0,100\n" % (min(x)-e,
                                                     max(x)+e,
                                                     min(y)-e,
                                                     max(y)+e))
        
        #compute distance between points
        dist = e + np.linalg.norm(x-y)
        
        #make bonds
        self.lines.append("MKbonds %.2f all\n" % dist)
    
    """
    Submit a custom command to the model definition file.
    
    **Arguments**:
    -command = a string containing the custom command to write to the model definition file.
    """
    def custom(self, command):
        self.lines.append("%s\n" % command)
            
    """
    Write a trubalw.dat file that can recreate the most recently executed model without using any stochastic functions. This can help avoid issues associated
    with different random seed procedures on machines with different compilers. Note that this is a fairly crude hack... so may not perfectly reproduce models.
    Also note that some features (such as bonds) will not be reproduced!
    """
    def dumpState(self, file = 'trubalw.dat'):
        model = self.loadLastOutput() #load last state
        
        #get all executed lines, but remove any gen commands
        lines = []
        for l in self.history:
            if not "GEN" in l and not "LIN" in l and not "CR" in l: #not a command that generates balls
                cmd = l.split(" ")[0]
                if not cmd == "C" and not cmd == "PL" and not cmd == "PR" and not "BFIL " in l: #ignore cycle commands and output commands
                    lines.append(l)
        
        #loop through and create nodes
        for N in model.G.nodes:
            #get data
            x,y,z = float(model.G.nodes[N]['U.x']),float(model.G.nodes[N]['U.y']),float(model.G.nodes[N]['U.z']) #postion
            sType = int(model.G.nodes[N]['STYPE']) #size type
            mType = int(model.G.nodes[N]['MTYPE']) #material type
            lines.append("CReate %f %f %f %d %d\n" % (x,y,z,sType,mType)) #create this particle 
        #
        #lines.append("C 1") #this seems to be needed for us to then change particle properties
        #assign properties
        #for N in model.G.nodes:
            x,y,z = float(model.G.nodes[N]['U.x']),float(model.G.nodes[N]['U.y']),float(model.G.nodes[N]['U.z']) #postion
            vx,vy,vz = float(model.G.nodes[N]['UDOT.x']),float(model.G.nodes[N]['UDOT.y']),float(model.G.nodes[N]['UDOT.z']) #velocity
            tx,ty,tz = float(model.G.nodes[N]['THETA.x']),float(model.G.nodes[N]['THETA.y']),float(model.G.nodes[N]['THETA.z']) #rotation
            dtx,dty,dtz = float(model.G.nodes[N]['TDOT.x']),float(model.G.nodes[N]['TDOT.y']),float(model.G.nodes[N]['TDOT.z']) #angular velocity
            c = int(model.G.nodes[N]['COL']) #color
            dofT = model.G.nodes[N]['TFIXED'] #translation DOF
            dofR = model.G.nodes[N]['RFIXED'] #rotation DOF

            #and set its properties
            e = 0.5
            #lines.append("DOM %f %f %f %f %f %f\n" % (x-e,x+e,y-e,y+e,z-e,z+e))
            lines.append("POLy 1 %d %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n" % (0,100,
                                                                            x-e,y-e,
                                                                            x-e,y+e,
                                                                            x+e,y+e,
                                                                            x+e,y-e))
            lines.append("PRP 1 COL x %d\n" % c) #set color
            lines.append("PRP 1 VEL %f %f %f\n" % (vx,vy,vz)) #set velocity
            lines.append("PRP 1 FIX %s %s %s %s %s %s\n" % (dofT[0],dofT[1],dofT[2], #fix DOFs
                                                          dofR[0],dofR[1],dofR[2]))
            #lines.append("PRP 1 FIX 1 1 1 1 1 1\n")
            #TODO - set particle rotations and angular velocities??
        
        #add any outstanding lines (note - we assume no further GEN commands here!)
        lines += self.lines
        
        #add trailing lines
        lines.append("c 1\n") #not sure if this is needed, but cycle breifly to (possibly) apply changes
        lines.append("SAVE state_%d.sav\n" % self.file) #save the file state - so that following analyses can pick up on the output!
        lines.append("stop\n")
        
        #write file
        f = open(file,'w')
        f.writelines(lines)
        f.close()
        
    
    
#   "Private" functions for writing properties in this class to definition files. These should not be called from outside this class...

    """
    Create header lines for a new (as opposed to re-loaded) simulation.
    """
    def _startLines(self):
        if self.restart is None:
            line = "START %d %d %d %d %d %d LOG\n" % (self.bounds[0],self.bounds[1],self.bounds[2],self.nboxes,self.nballs,self.nwall)
            line += "P000 - %s\n" % self.name
            line += "2-D\n"
        else:
            line = "RESTART %s\n" % self.restart
        line += ("DMX %d\nFMX %.0E\n" % (self.plotScale,self.forceScale)).replace("+","")
        return line
    """
    Write the radii dictionary to the model definition file.
    """
    def _defineRadii(self):
        line = ""
        for key,value in self.radii.items():
            line += "RAD %.2f %d\n" % (value, key)
        return line
    """
    Write the density dictionary to the model definition file. 
    """
    def _defineDensity(self):
        line = ""
        for key,value in self.density.items():
            line+= "DENS %d %d\n" % (value,key)
        return line
    """
    Write the hertzian properties dictionary to the model defenintion file. 
    """
    def _defineHertzProperties(self):
        line = ""
        for key,value in self.hertz.items():
            line+= "HERTZ %E %f %d\n" % (value[0],value[1],key)
        return line
 
"""
Quick and dirty multithreading tool for executing multiple riceball projects at once.

**Arguments**:
-projects = a list of the projects to run in parallel. These must all operate from different projects. 
-saveState = Should each project save it's state at the end of the job? Default is True.
-verbose = True if a printout should be given each time a thread is finished.
-supress = If true, the model defenition file is written, but riceball is not invoked.
               This is useful if the model files have already been generated and you want to 
               fool ricepaper into thinking they have been freshly minted. 
"""
def multiThreadExecute( projects,saveState=True,verbose=True,suppress=False):
    import multiprocessing
    from threading import Thread
    from time import sleep
    
    nCores = multiprocessing.cpu_count() - 1 #number of cores avaliable for processing (this controls how many threads we launch)
    waiting = set(projects)
    active = set()
    while not (len(waiting) == 0 and len(active) == 0): #While there are jobs still to run/running?
        if len(active) < nCores and len(waiting) > 0: #are there free cores? Jobs still to run?
            job=waiting.pop() #get next job (n.b. this automatically removes it from the set)
            T = Thread(target=job.execute,args=(saveState,False,None,suppress)) #create thread for next job #args = saveState=True,returnLog=False,restart=None,suppress=False
            T.start() #spawn thread
            active.add(T) #add to active thread list
            
            if verbose:
                print("Launching job %d of %d." % ((len(projects) - len(waiting)),len(projects)))
            sleep(0.1) #give the computer a chance to launch this thread succesfully (avoids issues when python is asked to be in multiple working dirs concurrently)
            continue #skip to next loop, in case active set isn't full yet

        #active set is full - have any thread's finished?
        dead = []
        for T in active:
            if not T.isAlive(): #thread has finished
                dead.append(T)
        if len(dead) != 0:
            for d in dead:
                active.remove(d)
            continue #skip the sleep and launch threads!

        #wait a while until something finishes
        sleep(1) #sleep for 1 second

"""
Util function for asserting that an index is valid.
"""
def _validateIdx(index):
    assert isinstance(index,int), "index must be an integer."
    assert index >= 1, "Index must be >= 1"