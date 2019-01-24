"""
A class for loading and containing output data from riceball. Each object
maps each ball in the simulation to a networkx graph, attaches associated
physical properties and adds graph-edges for each of the interactions
(attributed with the relevant forces etc.).

Sam Thiele 2018
"""

import sys, os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

#vtk stuff
from pyevtk.hl import pointsToVTK
from pyevtk.hl import polyLinesToVTK

#import postprocess as pp

"""
A class to encapsulate a riceball model state, as loaded from
output files.
"""
class RiceBall:
    """
    Load a riceball model from the provided "OUT" file.
    
    **Arguments**:
    - file = the filepath (string) of the output file (*.OUT) to load
    
    **Keywords**:
    - radii = a dictionary containing the radius values for each shape ID. 
              If passed, a dictionary defining the radius of each ball is defined.
    """
    def __init__(self,file,**kwds):
        self.G = nx.Graph() #create a graph to store particles & interactions in
        self.pos = {} #dictionary allowing quick look-up of node positions given its id
        self.stress_computed = False
        self.file = file #store file path
        self.radii = None
        if "radii" in kwds:
            self.STYPE = kwds["radii"]
            assert isinstance(self.STYPE,dict), "Radii argument must be a dictionary linking S_TYPE to radius"
            self.radii = {}
        
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
            
            #get node radius?
            if not self.radii is None:
                sID = int(self.G.nodes[nID]["STYPE"])
                assert sID in self.STYPE, "Error - unexpected STYPE found. Check %d is in radii dict." % sID
                self.radii[nID] = self.STYPE[sID]
                
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
    Computes the per-particle stress tensors as per https://doi.org/10.1016/j.compgeo.2016.03.006
    
    **Arguments**:
    -storeOnNodes = True if stress tensor should be written to each node in the underlying
                    networkX graph (for later use?). Default is true.
    **Returns**:
    -a dictionary S such that S[nodeID] = stress tensor compued for nodeID.
    """
    def computeParticleStresses(self,storeOnNodes = True):
    
        if self.stress_computed: #allready done!
            return nx.get_node_attributes(self.G,"stress")
            
        #todo - construct veroni tesselation to get cell volumes

        #create dict to store stress tensors in
        S = {}
        
        #loop through nodes
        for N in self.G.nodes:
            #get edges associated with this node
            edges = self.G.edges(N,data=True)
            
            #initialise null stress tensor
            stress = np.zeros([3,3])
            
            #we also compute the vector sum of the forces - if they don't add to zero the stress tensor will be wrong!
            nsum = np.zeros(3) #normal-force sum
            tsum = np.zeros(3) #torque sum 
            #loop through edges
            for e in edges:
                #get normal and shear force components from model
                sF = np.array([float(e[2]["FS.x"]),float(e[2]["FS.y"]),float(e[2]["FS.z"])]) #shear force
                nF = float(e[2]["FN"]) #normal force
                
                #compute contact normal vector (the normal vector of this contact and the direction joining this particle to the contacting neighbour)
                #cN = np.array([float(self.G.nodes[e[1]]["U.x"]),
                #                   float(self.G.nodes[e[1]]["U.y"]),
                #                   float(self.G.nodes[e[1]]["U.z"])])
                #cN -= np.array([float(self.G.nodes[e[0]]["U.x"]),
                #                   float(self.G.nodes[e[0]]["U.y"]),
                #                   float(self.G.nodes[e[0]]["U.z"])])
                #cN = cN / np.linalg.norm(cN) #normalise to unit vector
                #nn = cN
                
                #cN is perpendicular to sF - calculate using cross product
                #N.B. ONLY WORKS IN 2D! (i.e. when the z-component of all vectors is 0)
                cN = np.cross( sF / np.linalg.norm(sF), np.array([0,0,1]))
                cN = cN / np.linalg.norm(cN)
                
                #transform nF into vector by multiplying with contact normal
                nF = nF * cN
                
                #calculate branch vector (normal vector * the particle radius)
                branch = np.array(cN) #be sure to copy data rather than pointer!
                if not self.radii is None: #are radii defined?
                    branch *= self.radii[N] #multiply contact normal by particle radius
                
                #check normal and shear components are perpendicular... N.B. Due to generally low precision of the output forces, this will be only approximate hence the low level of precision
                assert np.abs(np.dot(sF/np.linalg.norm(sF),nF/np.linalg.norm(nF))) < 1e-2, "Fn [%e,%e,%e] and Fs [%e,%e,%e] not perpendicular (dot=%e) !?!" % (nF[0],nF[1],nF[2],sF[0],sF[1],sF[2],np.dot(sF / np.linalg.norm(sF) ,nF / np.linalg.norm(nF)))
                
                #resolve combined force
                force = nF + sF
                
                #also sum all forces acting on particle (these should sum to zero if the particle is in equillibrium) 
                nsum += nF
                tsum += np.cross(branch,sF)
                
                #add to stress tensor as per:
                #Sij = 1 / volume * Sum(force_i * branch_j)
                stress[0,0] += force[0] * branch[0]
                stress[0,1] += force[0] * branch[1]
                stress[0,2] += force[0] * branch[2]
                stress[1,0] += force[1] * branch[0]
                stress[1,1] += force[1] * branch[1]
                stress[1,2] += force[1] * branch[2]
                stress[2,0] += force[2] * branch[0]
                stress[2,1] += force[2] * branch[1]
                stress[2,2] += force[2] * branch[2]
            
            if (stress[0,0] > 1000):
                print("uncorrected symmetric terms are: = %E, %E" % (stress[1,0],stress[0,1]))
                
            #compute residual force and apply to stress tensor to ensure symmetry
            #n.b. - we need to treat normal and shear forces differently here????
            if len(edges) > 0:
                nResidual = np.linalg.norm(nsum)
                tResidual = np.linalg.norm(tsum)
                
                if nResidual > 0:
                    print( "normal residual = %E, shear residual = %E" % (nResidual,tResidual) )
                    
                    #calculate point to apply force (normal & shear) to cancel residual forces
                    branch = ( nsum / nResidual ) #direction of residual normal force
                    if not self.radii is None:
                         branch *= self.radii[N]
                         
                    #normal force is simply the inverse of the residual normal source
                    nF = -nsum 
                    
                    #calculate shear force that cancels residual torque
                    sF = np.cross(tsum,branch) / (np.linalg.norm(branch) ** 2)
                    
                    #calculate combined force vector
                    force = nF + sF
                    
                    stress[0,0] += force[0] * branch[0]
                    stress[0,1] += force[0] * branch[1]
                    stress[0,2] += force[0] * branch[2]
                    stress[1,0] += force[1] * branch[0]
                    stress[1,1] += force[1] * branch[1]
                    stress[1,2] += force[1] * branch[2]
                    stress[2,0] += force[2] * branch[0]
                    stress[2,1] += force[2] * branch[1]
                    stress[2,2] += force[2] * branch[2]

            #normalise to node volume (TODO - replace ball volume with veroni cell volume)
            V = 1
            if not self.radii is None:
                V = 4./3. * np.pi * self.radii[N]**3
            stress = stress / V
            
            if (stress[0,0] > 1000):
                print("symmetric terms are: = %E, %E" % (stress[1,0],stress[0,1]))
            
            #store stress tensor
            S[N] = stress
            
            if storeOnNodes:
                self.G.node[N]["stress"] = stress
                self.stress_computed = True
                #compute and store principal stresses
                eigval,eigvec = np.linalg.eig(stress)
                
                #avoid complex numbers from null stresses
                if True in np.iscomplex(eigval):
                    eigval = np.zeros([3])
                    eigvec = np.zeros([3,3])
                
                #sort eigens
                idx = eigval.argsort()[::-1] #sort
                eigval = eigval[idx]
                eigvec = eigvec[:,idx].T #transpose so rows are vectors
                
                #store principal stresses
                self.G.node[N]["sigma1"] = eigval[0] * eigvec[0]
                #self.G.node[N]["sigma2"] = eigval[1] * eigvec[1] #meaningless in 2D
                self.G.node[N]["sigma3"] = eigval[1] * eigvec[1]
                self.G.node[N]["dif"] = eigval[0] - eigval[1] #differential stress
                
        return S
    
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
        pos = list(self.pos.values())
        if dynamic:
            #filter out static nodes
            pos = []
            for N in self.G.nodes:
                if self.G.nodes[N]["TFIXED"] != '111': #non-static ball
                    pos.append(self.pos[N])
        
        #now calculate the bounds of all (valid) positions
        minx = np.min(np.array(pos).T[0])
        maxx = np.max(np.array(pos).T[0])
        miny = np.min(np.array(pos).T[1])
        maxy = np.max(np.array(pos).T[1])
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
            if self.radii is None:
                r.append(int(self.G.nodes[N]['STYPE']))
            else:
                r.append(self.radii[N])
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
    -nodeSize = The size of the nodes in this figure. Default is 10.
    -nodeList = A list of nodes to be drawn. If None, all nodes are drawn. Default is None.
    -color = The color of the nodes to draw. If None, individual node colors are used.
    -hold = If true, the matplotlib canvas is held for future plot commands. Default is False.
    -figsize = The size of the matplotlib figure as (x,y). Default is (18,5).
    """
    def quickPlot(self,nodeSize=10,nodeList=None,color=None,hold=False,figsize=(18,5)):
        #make/get list of nodes
        if nodeList is None:
            nodeList = self.G.nodes #all nodes
        
        #build plot
        plt.figure(figsize=figsize)
        if not color is None: #specified draw color
            nx.draw(self.G,self.pos,node_size=nodeSize,nodelist=nodeList,node_color=color,hold=True)
        else:
            #draw edges
            nx.draw(self.G,self.pos,node_size=0.1,hold=True)
            
            colDict = {}
            for n in nodeList:
                #get node color
                c = int(self.G.nodes[n]['COL'])
                
                #parse to matplotlib color
                color = 'gray'
                if c == 1:
                    color = "g"
                if c == 2:
                    color = "y"
                if c == 3:
                    color = "r"
                if c == 4:
                    color = "w"
                if c == 5:
                    color = "k"
                if c == 6:
                    color = "gray"
                if c == 7:
                    color = "b"
                if c == 8:
                    color = "xkcd:cyan"
                if c == 9:
                    color = "xkcd:violet"
                
                if color in colDict:
                    colDict[color].append(n) #add to existing list
                else:
                    colDict[color] = [n] #create new list
                
                #plot nodes for each color group
            for color,nodeList in colDict.items():
                nx.draw(self.G,self.pos,node_size=nodeSize,nodelist=nodeList,edgelist=[],node_color=color,hold=True)

        #draw figure?
        if not hold:
            plt.show() 





