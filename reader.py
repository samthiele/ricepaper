"""
A class for loading and containing output data from riceball. Each object
maps each ball in the simulation to a networkx graph, attaches associated
physical properties and adds graph-edges for each of the interactions
(attributed with the relevant forces etc.).

Sam Thiele 2018
"""

import sys, os
import glob
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#vtk stuff
from pyevtk.hl import pointsToVTK
from pyevtk.hl import polyLinesToVTK

"""
A class to encapsulate a riceball model state, as loaded from
output files.
"""
class RiceBall:
    """
    Load a riceball model from the provided "OUT" file.
    
    **Arguments**:
    - file = the filepath (string) of the output file (*.OUT) to load
    - radii = A dictionary defining the radius of each ball is defined and each node given a radius. 
    - density = A dictionary defining the density of each ball is defined and each node given a mass.
    """
    def __init__(self,file,radii,density):
        self.G = nx.Graph() #create a graph to store particles & interactions in
        self.pos = {} #dictionary allowing quick look-up of node positions given its id
        self.stress_computed = False
        self.file = file #store file path
        self.radii = radii #store radii dict
        self.density = density #store density dict
        
        #check the file exists...
        assert os.path.exists(file), "Error - could not find file %s" % file
        
        #load lines in file
        f = open(file,'r')
        data = f.readlines()
        f.close()
        lineNum = 0
        
        #skip to section containing particle data (TODO -  read data at header of file)
        while not "ADDRESS" in data[lineNum]:
            lineNum+=1
            
        #function for reading space delimited line
        def splitLine(line):
            l = line.split(' ')
            l = list(filter(None,l))
            return l

        #function for processing header line of tabulated data
        def getHeader(line):
            #processes header
            header = splitLine(line)
            header = [x.replace("(1)",".x") for x in header]
            header = [x.replace("(2)",".y") for x in header]
            header = [x.replace("(3)",".z") for x in header]
            return header
        
        #get header for first chunk of node data
        header = getHeader(data[lineNum])
        
        #now get node data
        lineNum +=1
        while not "ADDRESS" in data[lineNum]:
            dataLine = splitLine(data[lineNum])
            lineNum += 1 #this line has been read
            
            #skip invalid lines
            if len(dataLine) != len(header):
                continue
                
            #build key-value attribute dict
            attributes = {}
            for i,column in enumerate(header):
                attributes[column] = dataLine[i]
            
            #add networkx node
            nodeID = int(dataLine[0])
            self.G.add_node(nodeID,**attributes)
            
            #add node to "position" dictionary
            self.pos[nodeID] = (float(self.G.nodes[nodeID]['U.x']),
                                float(self.G.nodes[nodeID]['U.y']))
        
        #now get second chunk of node data
        header = getHeader(data[lineNum])
        header[10] = 'TFIXED' #translational degrees of freedom
        header.insert(11,'RFIXED') #we need to add another header for the rotational degrees of freedom due to quirky file format...
        
        lineNum +=1
        while not "ADDRESS" in data[lineNum]:
            dataLine = splitLine(data[lineNum])
            lineNum += 1 #this line has been read
            
            #skip invalid lines
            if len(dataLine) != len(header):
                continue
              
            #get node ID
            nID = int(dataLine[0])
            #add attributes
            for i,column in enumerate(header):
                assert(nID in self.G.nodes), "Error - could no find all data for node %d" % nID
                self.G.nodes[nID][column] = dataLine[i]
            
            #get node radius and volume
            if not self.radii is None:
                sID = int(self.G.nodes[nID]["STYPE"])
                assert sID in self.radii, "Error - unexpected STYPE found. Check %d is in radii dict." % sID
                self.G.nodes[nID]["radius"] = self.radii[sID]
                self.G.nodes[nID]["volume"] = (self.radii[sID]**3) * (4/3) * np.pi
            #get node mass and volume 
            if not self.density is None:
                mID = int(self.G.nodes[nID]["MTYPE"])
                if mID == 0: #special case -> contains wall nodes
                    self.density[mID] = 2500 #assign arbitrary density
                assert mID in self.density, "Error - unexpected MTYPE found. Check %d is in radii dict." % mID
                self.G.nodes[nID]["mass"] = self.density[mID]*self.G.nodes[nID]["volume"]
                
        #finally, add edge data
        header = getHeader(data[lineNum])
        
        lineNum += 1
        while lineNum < len(data): #loop to eof
            dataLine = splitLine(data[lineNum])
            lineNum += 1 #this line has been read
            
            #skip invalid lines
            if len(dataLine) != len(header):
                continue

            #build key-value attribute dict
            attributes = {}
            for i,column in enumerate(header):
                attributes[column] = dataLine[i]
                
            #add attribute describing magnitude of shear stress
            attributes["Fs"] = np.linalg.norm( np.array( [float(attributes["FS.x"]),float(attributes["FS.y"])]))
            
            
            #add edge
            self.G.add_edge(int(attributes["BALL1"]),int(attributes["BALL2"]),**attributes)
            
            #calculate magnintude of shear stress?
            
            
        #is there associated bond breakage data?
        bfile = os.path.splitext(file)[0] + ".bnd"
        if os.path.isfile(bfile):
            self.brk = [] #store breakage data such that each row corresponds to a breakage event
            f = open(bfile,"r")
            lines = f.readlines()
            header = getHeader(lines[0])
            for l in lines[1:]: #first line is header: Action  Cycle  Styp1  Styp2  Rad1  Rad2  x1  y1  z1      x2  y2  z2  Fn  Fs  Fsmax
                data = splitLine(l)
                data[1] = "%s %s" % (data[0],data[1]) #concatenate first two lines
                #store data
                self.brk.append(data[1:]) #store

        
    """
    Returns the number of balls in this model step 
    """
    def nBalls(self):
        return len(self.G)
        
    """
    Returns the number of interactions in this model step
    """
    def nInteractions(self):
        return self.G.number_of_edges()
    
    
    """
    Quicky extract per-particle attributes.
    
    **Arguments**:
     - attr = A list of the attributes to get. Options are: 
                - "position" = the position of each particle (m)
                - "velocity" = the velocity vector of each particle (m/sec)
                - "speed" = the speed (scalar) of each particle (m/sec)
                - "mass" = the mass of each particle (kg)
                - "density" = the density of each particle (kg/m^3)
                - "radius" = the radius of each particle  (m)
                - "volume" = the volume of each particle (m^3)
                - "kinetic" = the kinetic energy of each particle (J)
                - "momentum" = the momentum of each particle (kg m / sec )
                - "force" = the net force acting on each particle.
                - "torque" = the net torque acting on each particle. 
                - "acceleration" = the linear accleration that would result from the net force on this particle.
                - "stress" = a 3x3 stress per-particle stress tensor
    **Keywords**:
     - recalc = if true, all per-particle parameters are recalculated rather than just retrieved. Default is False. 
     - ignoreFixed = if true, velocities of fixed particles are ignored. Default is true. 
     - onlyFixed = if true, velocity of non-fixed particles are ignored. Useful for tests involving walls. Default is false.
     - gravity = gravitational acceleration vector used for force calculations (only). Default is [0,-9.8,0].
     - nodes = a list of node IDs to return attributes for (otherwise data from all nodes are returned). 
    Returns:
     - V = a list such that the first row contains particle IDs and subsequent rows contain 
           the requested attributes (in the same order as in attr). 
    """
    def getAttributes(self,attr,**kwds):
        
        #check attr is a list
        if not isinstance(attr,list):
            attr = [attr]
        
        #get kewords
        recalc = kwds.get("recalc",False)
        ignoreFixed = kwds.get("ignoreFixed",True)
        onlyFixed = kwds.get("onlyFixed",False)
        if onlyFixed: #can't have both as true...
            ignoreFixed = False
            
        gravity = kwds.get("gravity",np.array([0,-9.8,0]))
        nodes = kwds.get("nodes",self.G.nodes)
        
        #init output array
        out = [[] for i in range(len(attr) + 1)]
        
        #force recalc?
        if recalc:
            self.computeAttributes(gravity=gravity)
        
        #loop through particles
        for N in nodes:
            #ignore fixed/non-fixed particles depending on input
            if ignoreFixed and '111' in self.G.nodes[N]["TFIXED"]:
                continue
            if onlyFixed and not '111' in self.G.nodes[N]["TFIXED"]:
                continue
                
            #loop through and get requested attributes
            out[0].append(N) #store PID
            for i,a in enumerate(attr):
                a = a.lower() #conver to lower case
                if "pos" in a: #get position
                    out[i+1].append(np.array([float(self.G.nodes[N]["U.x"]),
                                       float(self.G.nodes[N]["U.y"]),
                                       float(self.G.nodes[N]["U.z"])]))
                elif "vel" in a: #get velocity
                    out[i+1].append(np.array([float(self.G.nodes[N]["UDOT.x"]),
                                       float(self.G.nodes[N]["UDOT.y"]),
                                       float(self.G.nodes[N]["UDOT.z"])]))
                elif "spe" in a: #get speed
                    out[i+1].append(np.linalg.norm(np.array([float(self.G.nodes[N]["UDOT.x"]),
                                                   float(self.G.nodes[N]["UDOT.y"]),
                                                   float(self.G.nodes[N]["UDOT.z"])])))
                elif "mass" in a: #get mass
                    out[i+1].append(self.G.nodes[N]["mass"])
                elif "dens" in a: #get density
                    mID = int(self.G.nodes[N]["MTYPE"])
                    out[i+1].append(self.density[mID])
                elif "rad" in a: #get radius
                    out[i+1].append(self.G.nodes[N]["radius"])
                elif "vol" in a: #get volume
                    out[i+1].append(self.G.nodes[N]["volume"])
                elif "kin" in a: #get kinetic energy
                    if not "kinetic" in self.G.nodes[N]:
                        self.computeAttributes(gravity=gravity)
                    out[i+1].append( self.G.nodes[N]["kinetic"] )
                elif "moment" in a:
                    if not "momentum" in self.G.nodes[N]:
                        self.computeAttributes(gravity=gravity)
                    out[i+1].append( self.G.nodes[N]["momentum"] )
                elif "torque" in a:
                    if not "torque" in self.G.nodes[N]:
                        self.computeAttributes(gravity=gravity)
                    out[i+1].append( self.G.nodes[N]["torque"] )
                elif "force" in a: 
                    if not "force" in self.G.nodes[N]:
                        self.computeAttributes(gravity=gravity)
                    out[i+1].append( self.G.nodes[N]["force"] )
                elif "acc" in a:
                    if not "acc" in self.G.nodes[N]:
                        self.computeAttributes(gravity=gravity)
                    out[i+1].append( self.G.nodes[N]["acc"] )
                elif "stress" in a:
                    if not "stress" in self.G.nodes[N]:
                        self.computeAttributes(gravity=gravity)
                    out[i+1].append( self.G.nodes[N]["stress"] )
                elif a in self.G.nodes[N]: #return any other node attribute
                    out[i+1].append( self.G.nodes[N][a] ) 
        #completo
        return out
    
    """
    Pre-computes per-particle attributes such as stress, resultant force and acclerations. These are stored on the
    node graph for later use (e.g. with getAttributes(...)) or export to vtk. 
    
    **Keywords**:
        - gravity = gravitational acceleration vector used for force calculations (only). Default is [0,-9.8,0].
        
    """
    def computeAttributes(self, **kwds):
        
        gravity = kwds.get("gravity",np.array([0,-9.8,0]))
        
        #loop through particles
        for N in self.G.nodes:
            
            #compute kinetic energy and momentum
            v = np.linalg.norm(np.array([float(self.G.nodes[N]["UDOT.x"]), #velocity
                               float(self.G.nodes[N]["UDOT.y"]),
                               float(self.G.nodes[N]["UDOT.z"])]))
                               
            self.G.nodes[N]["kinetic"] = 0.5*self.G.nodes[N]["mass"]*(v**2) #kinetic energy
            self.G.nodes[N]["momentum"] = self.G.nodes[N]["mass"] * v
            
            #compute resultant force, acceleration and stress
            A = np.array([float(self.G.nodes[N]["U.x"]),  #get position of this particle
                         float(self.G.nodes[N]["U.y"]),
                         float(self.G.nodes[N]["U.z"])])
            
            T = [ np.array([0,0,0]) ] #init torque to zero
            F = [ gravity * self.G.nodes[N]["mass"] ] #init force to gravity
            S = np.zeros([3,3]) #init null stress tensor 
                    
            for e in self.G.edges(N,data=True):
                #get position of second particle
                N2 = int(e[2]["BALL1"])
                dir = 1.0 #shear force in correct orientation 
                if N2 == N:
                    N2 = int(e[2]["BALL2"])
                    dir = -1.0 #shear force needs to be flipped
                B = np.array([float(self.G.nodes[N2]["U.x"]),
                           float(self.G.nodes[N2]["U.y"]),
                           float(self.G.nodes[N2]["U.z"])])
                
                #calculate location force is applied
                AB = B - A
                nn = AB / np.linalg.norm(AB) #contact normal vector
                R = self.G.nodes[N]["radius"] * nn #radius vector (sometimes called the "branch vector"0
                
                #get normal and shear force on contact
                sF = np.array([float(e[2]["FS.x"]),float(e[2]["FS.y"]),float(e[2]["FS.z"])]) #shear force
                nF = -float(e[2]["FN"]) * nn
                
                #flip direction of shear force depending on order of contact
                sF *= dir
                
                #calculate torque vector from this contact
                T.append( np.cross(R,sF) )
                
                #calculate force vector from this contact
                f = nF + sF
                F.append(f)
                
                #accumulate stress tensor
                for u in range(3):
                    for v in range(3):
                        S[u][v] += R[u]*f[v]
            
            #normalise stress tensor and average shear components to ensure symmetry
            if not '111' in self.G.nodes[N]["TFIXED"]:
                S = S / self.G.nodes[N]["volume"]
                
                #make compression positive
                S = S * -1
                
                #average shear stresses
                S[0,1] = np.mean([S[0,1],S[1,0]])
                S[1,0] = S[0,1]
                
                #calculate principal stresses
                eigval, eigvec = np.linalg.eig(S)
                idx = eigval.argsort()[::-1]
                eigval = eigval[idx]
                eigvec = eigvec[:,idx]
                self.G.nodes[N]["sig1"] = eigval[0] * eigvec[:,0]
                self.G.nodes[N]["sig3"] = eigval[1] * eigvec[:,1]
                
                #calculate mean and deviatoric stresses (n.b. 2D case only!)
                Sm = np.mean( [eigval[0],eigval[1]] )
                self.G.nodes[N]["meanStress"] = Sm
                self.G.nodes[N]["deviatoricStress"] = S - np.array( [[Sm,0,0],
                                                                     [0,Sm,0],
                                                                     [0,0,0]] )
            else: #special case for fixed nodes - write null stress tensors (as stress calculation will be garbage)
                S = np.zeros([3,3])
                self.G.nodes[N]["sig1"] = np.zeros(3)
                self.G.nodes[N]["sig3"] = np.zeros(3)
                self.G.nodes[N]["meanStress"] = 0
                self.G.nodes[N]["deviatoricStress"] = S
            
            #calculate resultant force and torque and associated accelerations
            rF = np.sum(F,axis=0)
            rT = np.sum(T,axis=0)
            acc = rF / self.G.nodes[N]["mass"]
            ang = np.rad2deg(rT / (2/5*self.G.nodes[N]["mass"]*self.G.nodes[N]["radius"]**2))
            
            #store all on graph
            self.G.nodes[N]["force"] = rF #net force
            self.G.nodes[N]["torque"] = rT #net torque
            self.G.nodes[N]["acc"] = acc #linear acceleration
            self.G.nodes[N]["ang"] = ang #angular acceleration
            self.G.nodes[N]["stress"] = S #stress tensor
            
    """
    Computes the displacements between matching particles relative to the reference (prior-state) model.
    
    **Arguments**:
    -reference = the reference model representing the initial particle positions. If no match is found, a displacement of (0,0,0) is assigned.
    -storeOnNodes = True if particle displacement vectors are added to the networkX nodes for future use. Default is True.
    """
    def computeParticleDisplacements(self,reference,storeOnNodes=True):
        D = {}
        for N in self.G.nodes: #loop through nodes
            if N in reference.G.nodes: #is there a matching node in the reference?
                D[N] = np.array(self.pos[N]) - np.array(reference.pos[N]) #compute displacement
            else:
                D[N] = np.array([0,0,0]) #no match - assign null displacement
            if storeOnNodes:
                self.G.nodes[N]["disp"] = D[N] #store result
        return D
    
    
    """
    Removes particles on one side of a cutting line.
    
    **Arguments**:
    -x = the x-intercept of the cutting line
    -dip = the dip (degrees) of the cutting line.Positive values are west-dipping.
    -lower = True (default) if points under the line are retained. If false, points above the line are retained.
    -ignoreFixed =  True (default) if fixed points (typically boundary points) are ignored by the cutting operation.
    **Returns**
    - Returns a list of deleted points. Deleted points are also removed from the underlying networkX network in this model instance. 
    """
    def cut(self,x,dip,lower=True, ignoreFixed=True):
        assert abs(dip) <= 90, "Error: invalid dip (%f) when generating dyke." % dip
        
        #convert dip to radians
        dip = np.deg2rad(dip) 
        
        #calculate cutting line direction vector
        dir = np.array([np.cos(dip),np.sin(dip),0])
        if dir[0] < 0: #ensure positive x so that the sign of the cross product makes sense
            dir = dir * -1
            
        deletedList = []
        for N in self.G.nodes:
        
            if ignoreFixed and self.G.node[N]["TFIXED"] == "111":
                continue #ignore this node
            
            #get node position relative to line origin
            p = np.array([self.pos[N][0] - x, self.pos[N][1], 0])
            
            #compute cross product
            cross = np.cross(dir,p)
            
            #test based on sign of z-component (will be positive if point is "above" the cutting line)
            if (cross[2] > 0 and lower) or (cross[2] < 0 and not lower): #wrong side of the line?
                deletedList.append(N)
        
        #delete nodes from graph
        self.G.remove_nodes_from(deletedList)
        
        return deletedList
        
    """
    Returns the 2D bounds of all balls in this model.
    Arguments:
    -dynamic = if set as True, only balls that can move (as opposed to boundary
               balls) are included in the bounds calculation. Default is False.
    Returns:
    -minx,max,miny,maxy = the minimum and maximum positions for each axis
    """
    def getBounds(self,dynamic=False):
        pos = []
        r = []
        for N in self.G.nodes:
            if dynamic and self.G.nodes[N]["TFIXED"] == '111':
                continue #skip static balls
            pos.append(self.pos[N])
            r.append(self.G.nodes[N]["radius"])
        
        #now calculate the bounds of all (valid) positions
        minx = np.min(np.array(pos).T[0] - r)
        maxx = np.max(np.array(pos).T[0] + r)
        miny = np.min(np.array(pos).T[1] - r)
        maxy = np.max(np.array(pos).T[1] + r)
        return minx,maxx,miny,maxy
        
    """
    Writes the nodes and interactions in this model step to VTK file for
    visualisation in ParaView or similar.
    
    **Arguments**:
    -filename = the base filename used to create the vtk files
    """
    def writeVTK(self, filename):
        #gather node data
        x = []
        y = []
        z = []
        r = []
        c = []
        id = []
        tfixed = []
        rfixed = []
        disp = []
        
        if self.stress_computed:
            S = []
            sig1 = []
            sig3 = []
            dif = []

        for N in self.G.nodes:
            x.append(self.pos[N][0])
            y.append(self.pos[N][1])
            z.append(0.0)
            c.append(int(self.G.nodes[N]['COL']))
            id.append(N)
            tfixed.append(int(self.G.nodes[N]['TFIXED']))
            rfixed.append(int(self.G.nodes[N]['RFIXED']))
            if 'radius' in self.G.nodes[N]:
                r.append(self.G.nodes[N]['radius'])
            else:
                r.append(int(self.G.nodes[N]['STYPE']))
            if self.stress_computed:
                S.append( self.G.node[N]["stress"] )
                sig1.append( self.G.node[N]["sigma1"] )
                sig3.append( self.G.node[N]["sigma3"] )
                dif.append( self.G.node[N]["dif"] )
            if "disp" in self.G.node[N]: #displacement has been computed
                disp.append( self.G.node[N]["disp"] )
                
        #build point attribute dictionary
        data = {"id" : np.array(id),
                "radius" : np.array(r), 
                "color" : np.array(c,dtype=int),
                "DOF-T" : np.array(tfixed,dtype=int),
                "DOF-R" : np.array(rfixed,dtype=int)}
        
        if self.stress_computed: #TODO - figure out how to write non-scalar fields to VTK
            data["sig1.magnitude"] = np.sqrt( np.array(sig1)[:,0]**2 + np.array(sig1)[:,1]**2)
            data["sig1.x"] = np.array(sig1)[:,0] #sigma 1 vector (components)
            data["sig1.y"] = np.array(sig1)[:,1]
            data["sig3.magnitude"] = np.sqrt( np.array(sig3)[:,0]**2 + np.array(sig3)[:,1]**2)
            data["sig3.x"] = np.array(sig3)[:,0] #sigma 3 vector (components)
            data["sig3.y"] = np.array(sig3)[:,1]
            data["S.xx"] = np.array(S)[:,0,0] #2D stress tensor
            data["S.xy"] = np.array(S)[:,0,1]
            data["S.yy"] = np.array(S)[:,1,1]
            data["dif"] = np.array(dif)
            
        #export particle displacements
        if len(disp) != 0: 
            data["disp.magnitude"] = np.sqrt( np.array(disp)[:,0]**2 + np.array(disp)[:,1]**2)
            data["disp.x"] = np.array(disp)[:,0]
            data["disp.y"] = np.array(disp)[:,1]
        
        #write nodes
        pointsToVTK(filename + "_balls",np.array(x),np.array(y),np.array(z),
                                    data = data)
        
        #gather edge data
        x = []
        y = []
        z = []
        sID = []
        eID = []
        ppl = [] #number of points per line. Always = 2.
        Fn = []
        Fs = [] #magnitude of shear force
        Fsx = [] #x-component of shear force
        Fsy = [] #y-component of shear force
            
        for e in self.G.edges(data=True):
            x += [self.pos[e[0]][0],self.pos[e[1]][0]]
            y += [self.pos[e[0]][1],self.pos[e[1]][1]]
            z += [0.0,0.0]
            sID.append(e[0])
            eID.append(e[1])
            ppl.append(2) 
            Fn.append(float(e[2]["FN"]))
            Fsx.append(float(e[2]["FS.x"]))
            Fsy.append(float(e[2]["FS.y"]))
            Fs.append(e[2]["Fs"])
            
        #build properties dict
        cellData = {"Point1" : np.array(sID,dtype=int), 
                                         "Point2" : np.array(eID,dtype=int),
                                         "Fn" : np.array(Fn),
                                         "Fs" : np.array(Fs),
                                         "Fs.x" : np.array(Fsx),
                                         "Fs.y" : np.array(Fsy)}
                                         
        #write lines vtk
        polyLinesToVTK(filename + "_interactions",np.array(x),np.array(y),np.array(z),
                              pointsPerLine = np.array(ppl,dtype=int),
                              cellData=cellData)
                              
                              
                              
    """
    Creates a quick matplotlib visualisation of this model.
    
    **Arguments**:
    -nodeSize = Fraction of particle radius used when drawing. Default is 1.0. 
    -nodeList = A list of nodes to be drawn. If None, all nodes are drawn. Default is None.
    -color = The color of the nodes to draw. If None, individual node colors are used.
    -hold = If true, the matplotlib canvas is held for future plot commands. Default is False.
    -figsize = The size of the matplotlib figure as (x,y). Default is (18,5).
    """
    def quickPlot(self,nodeSize=1.0,nodeList=None,color=None,hold=False,figsize=(18,5)):
        #make/get list of nodes
        if nodeList is None:
            nodeList = self.G.nodes #all nodes
        
        #build plot
        plt.figure(figsize=figsize)
        
        #draw edges
        nx.draw(self.G,self.pos,node_size=0.1,hold=True)
        
        #build circles
        circles = []
        for i,n in enumerate(nodeList):
            #get colour
            c = "gray" #default
            if color is None:
                pc = int(self.G.nodes[n]['COL'])
                if pc == 1:
                    c = "g"
                elif pc == 2:
                    c ="y"
                elif pc == 3:
                    c = "r"
                elif pc == 4:
                    c = "w"
                elif pc == 5:
                    c = "k"
                elif pc == 6:
                    c = "gray"
                elif pc == 7:
                    c = "b"
                elif pc == 8:
                    c = "xkcd:cyan"
                elif pc == 9:
                    c = "xkcd:violet"
            else: #use predefined colour
                if isinstance(color,list):
                    c = color[i]
                else:
                    c = color
            
            #build circle patch
            plt.gca().add_artist(matplotlib.patches.Circle(self.pos[n], 
                            radius = self.G.nodes[n]["radius"]*nodeSize,
                            color = c))
        
        #set limits
        minx,maxx,miny,maxy = self.getBounds()
        plt.gca().set_aspect('equal')
        plt.xlim(minx,maxx)
        plt.ylim(miny,maxy)
        
        #draw figure?
        if not hold:
            plt.show() 

    """
    Plots the contact network and any particles that are subject to unbalanced forces that cause accelerations greater
    than the specified threshold. Useful for identifying parts of an assemblage that are not static. 
    
    **Arguments**:
     - minAcc = the acceleration threshold below which particles will not be plotted.
     
    **Keywords**:
     - keywords are passed to RiceBall.getAttributes(...). Use to set gravity, ignore particle types etc. 
    """
    def plotUnbalanced( self, minAcc = 0.1, **kwds ):
        pid, acc = self.getAttributes(["acc"],**kwds)
        a = np.linalg.norm(acc,axis=1) #get magnitude of acceleration vectors
        self.quickPlot(nodeList=np.array(pid)[a>minAcc]) #plot unbalanced particles
    
    """
    Plot particles in the model coloured by stress magnitude and with a tick showing the direction of sigma 1. 
    
    **Keywords**:
        - type = The stress scalar to colour particles by. Default is "mag". Options are:
            -"sig1" = use principal compressive stress magnitude
            -"sig2" = use least compressive stress magnitude
            -"mean" = use the mean or hydrostatic stress
            -"mag" = the magnitude of the stress tensor
            -"diff" or "dve" = the magnitude of the differential or deviatoric stress tensor
            -"sx" = the shear stress in the x-direction
            -"sy" = the shear stress in the y-direction
            -"nx" = the normal stress in the x-direction
            -"ny" = the normal stress in the y-direction
        - cmap =  the matplotlib colour map to draw stress magnitudes with. Default is "magma".
        - norm = a norm object to use for the colour mapping. Otherwise min to max is used. 
        - vmin = min value of the norm object for colour mapping (alternative to passing norm object directly).
        - vmax = max value of the norm object for colour mapping (alternative to passing norm object directly).
        - tickLength = the length of sigma1 ticks as a fraction of particle radius. Default is 0.75.
        - tickThickness = the thickness of sigma1 ticks in points. Default is 1.0.
        - linecolor = the colour of the ticks. Default is 'white'. 
        - nodeList = a list of nodes to plot. Default is all nodes. 
        - nodeSize = Fraction of particle radius used when drawing. Default is 1.0. 
        - figSize = the size of the figure. 
        - ignoreFixed = True if only dynamic (i.e. non-fixed) particles should be plotted. Default is True as stress tensors for
                        fixed particles will be incorrect. 
        - title = the title of the plot. 
    **Returns**:
     - fig, ax = the figure that has been plotted.
    """
    def plotStress(self,**kwds):
        
        #get kwds
        cmap = kwds.get("cmap","plasma")
        if isinstance(cmap,str):
            cmap = plt.get_cmap(cmap)
        
        type = kwds.get("type","mag")
        tl = kwds.get("tickLength",0.75)
        tt = kwds.get("tickThickness",1.0)
        nodes = kwds.get("nodeList", self.G.nodes)
        figSize = kwds.get("figSize", (10,10))
        nodeSize = kwds.get("nodeSize", 1.0)
        ignoreFixed = kwds.get("ignoreFixed", True)
        title = kwds.get("title",type)
        
        #gather stress scalars
        S = []
        for N in nodes:
        
            if not "stress" in self.G.nodes[N]:
                self.computeAttributes()
            
            #ignore fixed
            if ignoreFixed and '111' in self.G.nodes[N]["TFIXED"]:
                S.append(0)
                continue
                
            if "sig1" in type:
                S.append( np.linalg.norm(self.G.nodes[N]["sig1"]) )
            elif "sig3" in type:
                S.append( np.linalg.norm(self.G.nodes[N]["sig3"]) )
            elif "mean" in type or "hyd" in type:
                S.append( self.G.nodes[N]["meanStress"] )
            elif "mag" in type:
                S.append( np.linalg.norm( self.G.nodes[N]["stress"] ) )
            elif "diff" in type or "dev" in type:
                S.append( np.linalg.norm( self.G.nodes[N]["deviatoricStress"] ) )
            elif "sx" in type:
                S.append( self.G.nodes[N]["stress"][0,1] )
            elif "sy" in type:
                S.append( self.G.nodes[N]["stress"][1,0] )
            elif "nx" in type:
                S.append( self.G.nodes[N]["stress"][0,0] )
            elif "ny" in type:
                S.append( self.G.nodes[N]["stress"][1,1] )
            
        #build norm object
        vmin = kwds.get("vmin",np.min(S))
        vmax = kwds.get("vmax",np.max(S))
        norm = kwds.get("norm",matplotlib.colors.Normalize( vmin=vmin, vmax=vmax ))
        
        #build plot
        gs = matplotlib.gridspec.GridSpec(5,2, width_ratios=[25,1])
        fig = plt.figure( figsize=figSize )
        ax = [ plt.subplot( gs[:,0] ), plt.subplot( gs[2,1] ) ]
        
        #build circles
        circles = []
        for i,N in enumerate(nodes):
            
            #skip fixed
            if ignoreFixed and '111' in self.G.nodes[N]["TFIXED"]:
                continue
                
            #get colour
            c = cmap( norm(S[i]) )
            
            #build circle patch
            ax[0].add_artist(matplotlib.patches.Circle(self.pos[N], 
                            radius = self.G.nodes[N]["radius"]*nodeSize,
                            color = c))
            
            #build sigma1 tick
            sig1M = np.linalg.norm(self.G.nodes[N]["sig1"])
            if sig1M > 0:
                sig1 = self.G.nodes[N]["sig1"] / sig1M
                top = self.pos[N] + sig1[:2] * self.G.nodes[N]["radius"] * tl
                bottom = self.pos[N] - sig1[:2] * self.G.nodes[N]["radius"] * tl
                ax[0].add_artist(matplotlib.patches.Polygon(np.array([[top[0],top[1]],[bottom[0],bottom[1]]]),
                                            closed=False,
                                            color=kwds.get('linecolor','w'),
                                            linewidth=tt ) )
            
        #set limits
        minx,maxx,miny,maxy = self.getBounds()
        ax[0].set_aspect('equal')
        ax[0].set_xlim(minx,maxx)
        ax[0].set_ylim(miny,maxy)
        ax[0].set_title(title)
        
        #colorbar
        cb1 = matplotlib.colorbar.ColorbarBase(ax[1], cmap=cmap,
                                norm=norm,
                                orientation='vertical')
        cb1.set_label('Stress (Pa)')
        fig.tight_layout()
        
        fig.show()
        return fig,ax
    """
    Plot a histogram showing the net particle accelerations (net forces / particle mass).
    
    **Keywords**:
     - keywords are passed to RiceBall.getAttributes(...). Use to set gravity, ignore particle types etc. 
    """
    def plotNetAcc( self, **kwds ):
    
        #get data
        pid, force, torque = self.getAttributes(["force","torque"],**kwds)
        
        r = [] #resultant force
        t = [] #net torque
        for i,F in enumerate(force):
            r.append( np.linalg.norm(sum(F)) / self.G.nodes[pid[i]]["mass"] ) #divide mass
            t.append( np.rad2deg(np.linalg.norm(sum(torque[i])) / (2/5 * self.G.nodes[pid[i]]["mass"] * self.G.nodes[pid[i]]["radius"]**2 ))) #divide by moment of intertia


        fig,ax = plt.subplots(1,2,figsize=(20,5))

        ax[0].hist(r,bins=75,alpha=0.75,range=(0,np.percentile(r,90)))
        ax[0].set_xlabel("Net linear acceleration (m/sec)")
        ax[0].set_ylabel("# Particles")

        ax[1].hist(t,bins=75,alpha=0.75,range=(0,np.percentile(r,90)))
        ax[1].set_xlabel("Net angular acceleration (degrees/sec)")
        ax[1].set_ylabel("# Particles")

        fig.suptitle("Particle stability")
        fig.show()
    
"""
Utility function for loading all .OUT files in a given directory.

**Arguments**:
- dir = the directory path (string) containing the output files (*.OUT) to load
- radii = A dictionary defining the radius of each ball is defined and each node given a radius. 
- density = A dictionary defining the density of each ball is defined and each node given a mass.
    
"""
def loadFromDir(dir,radii,density):
    files = glob.glob(os.path.join(dir,"STEP*.out"))
    out = []
    for f in files:
        out.append( RiceBall(f,radii,density) )
    return out

