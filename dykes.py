"""
Series of functions for extracting chains of balls from sandpiles so that
bonds between them can be stiffened to form imposed force-chains that simulate
the behaviour of dykes in volcanic edifices.

Sam Thiele 2018
"""
import numpy as np
import networkx as nx

"""
A factory class for creating dyke injections in a sandpile.
"""
class DykeFarm:
    """
    Create a dykeFarm object.
    
    **Arguments**:
    -model = the model data (RiceBall class) that dykes are build in
    -p_arrest = the probability that a dyke will arrest at each step. Default is 0.
    """
    def __init__(self, model, p_arrest = 0):
        #store access to model
        self.model = model
        self.p_arrest = p_arrest
        #calculate upper and lower boundaries of volcanic pile
        #(this is where dykes start and end)
        self.lower,self.upper = self.getBoundaryNodes()
        
    """
    Compute the heighteset and lowest nodes in the pile along the x-axis. These
    represent the bottom and top surfaces of the volcano. Computed by gridding
    in the x-direction and picking nodes with the minimum and maximum elevations.
    
    Returns:
    -baseNodes,surfaceNodes = lists of the nodes on the volcano base and surface (respectively)
    """
    def getBoundaryNodes(self):
        #get bounds of dynamic balls in model
        minx,maxx,miny,maxy = self.model.getBounds(dynamic=True)
        xrange = maxx - minx
        #calculate bin-size from radius
        if self.model.radii is None:
            binsize = xrange / 100 #just use 100 bins as we don't no ball sizes...
        else:
            #set bin-size to twice the largest radius
            binsize = max(self.model.STYPE.values())*2
        
        #calculate number of cells (range / bin size)
        ncells = int((maxx - minx) / binsize) + 1

        #create array that will store the "highest nodes" and "lowest"
        highGrid = np.zeros(ncells) + miny #set to minimum height
        hnodeGrid = np.zeros(ncells,dtype=int) - 1 #N.B. node ID of -1 means not assigned
        lowGrid = np.zeros(ncells) + maxy #set to max height
        lnodeGrid = np.zeros(ncells,dtype=int) - 1

        #loop through nodes
        for N in self.model.G.nodes.keys():
            #ignore fixed wall nodes
            #if self.model.G.nodes[N]['TFIXED'] == '111': 
            #    continue
            
            #compute bin this node falls in and check it is within bounds
            xbin = int( (self.model.pos[N][0] - minx) / binsize)
            if xbin < 0 or xbin > ncells - 1:
                continue #this will be an edge-ball outside of the dynamic range - ignore
            
            #get node height
            y = self.model.pos[N][1]
            
            #check if this node represents a new extreme
            if y > highGrid[xbin]:
                highGrid[xbin] = y #store max height in this bin (so far!)
                hnodeGrid[xbin] = N #store node
            if y < lowGrid[xbin]:
                lowGrid[xbin] = y #store min height in this bin (so far!)
                lnodeGrid[xbin] = N #store node

        #extract non-null nodes from nodeGrids
        surfaceNodes = list(hnodeGrid[hnodeGrid != -1])
        baseNodes = list(lnodeGrid[lnodeGrid != -1])
        
        #done
        return baseNodes,surfaceNodes
        
    """
    Create and propagate a dyke who's orientation is controlled by the 
    shear stress on each particle interaction (dyke follows low shear stress).
    
    **Arguments**:
    -startx = the x-coordinate of the location to initiate dyke propagation
    """
    def dykeFromStress(self, startx ):
        #find closest start node
        dist = 1e50
        n = -1
        for N in self.lower:
            d = abs(startx-self.model.pos[N][0])
            if d < dist:
                dist = d
                n = N
        assert n != -1, "Bugger.."
        
        #propagate to surface
        dyke = [n]
        while not n in self.upper:
            #get edges associated with node n
            edges = self.model.G.edges(n,data=True)
            
            assert len(edges) > 0, "Error: invalid start node given for dyke."
            
            #dead-end found, dyke arrested (TODO - avoid this case)
            if len(edges) == 1:
                return dyke
                
            #find next node (the one joined by the interaction with the 
            #smallest shear stress
            minF = 1e50 #init to really big number
            for e in edges:
                #smallest shear stress so far? (and ignore edge back to start)
                if float(e[2]["Fs"]) < minF and not e[1] in dyke:
                    minF = float(e[2]["Fs"]) #new min has been found
                    print(n,e[0],e[1],n == e[1])
                    n = e[1] #update next node
            
            #add new node to dyke
            dyke.append(n)
            
            #check for dead-end (unique case of dyke arrest)
            print (self.model.G.degree(n))
            if self.model.G.degree(n) == 1:
                break #no-where else to go!
            
            #escape long loop
            if len(dyke) > 50:
                return dyke
        
        return dyke
    """
    Create and propagate a dyke with specified orientation.
    
    **Arguments**:
    -startx = the x-coordinate of the location to initiate dyke propagation
    -dip = the dip of the dyke to create. Positive dips are west-dipping, negative
                dips are east dipping. Dip should be given in degrees.
     -useCost = True if edges oriented differently to the specified dip are
                penalised in the routing algorithm. If false, the shortest path
                between the start position and end postion (as defined by the dip)
                is used. Default is False.
                
    **Returns**:
    -dykes = a list of node IDs of the nodes in the dykes
    -bonds = a list of tuples containing pairs of nodes joined by dyke segments
    """
    def dykeFromOri(self, startx, dip, useCost = False):
        assert abs(dip) <= 90, "Error: invalid dip (%f) when generating dyke." % dip
        
        dip = np.deg2rad(dip) #convert to radians
                
        #find closest start node
        dist = 1e50
        n = -1
        for N in self.lower:
            d = abs(startx-self.model.pos[N][0])
            if d < dist:
                #check node is not disconnected!
                if self.model.G.degree(N) > 0:
                    dist = d
                    n = N
        assert n != -1, "Bugger.."
        
        #compute edge cost as difference between edge-dip and desired dip
        if useCost:
            for e in self.model.G.edges(data=True):
                v = np.array(self.model.pos[e[1]]) - np.array(self.model.pos[e[0]])
                
                #calculate edge dip
                theta = np.arctan(v[1]/v[0])
                
                #assign cost as difference between this and desired dip
                e[2]["cost"] = abs(theta-dip)
            
        #calculate points defining line
        p1 = np.array(self.model.pos[n])
        p2 = p1 + np.array([np.cos(dip),np.sin(dip)])
        
        #loop through upper nodes and find the closest to the above line
        minDist = 1e50
        end = -1
        for N in self.upper:
            pN = self.model.pos[N] #get node postion
            
            #calculate distance
            dist = abs((p2[1]-p1[1])*pN[0] - (p2[0]-p1[0])*pN[1] + p2[0]*p1[1] - p2[1]*p1[0])
            if dist < minDist and self.model.G.degree(N) > 0: #ensure node is connected...
                minDist = dist
                end = N
                
        assert end != -1
        
        #solve and return shortest path between these two nodes
        path = []
        try:
            if useCost:
                path =  nx.shortest_path(self.model.G,n,end,"cost")
            else:
                path =  nx.shortest_path(self.model.G,n,end)
        except: #could not route dyke
            print("Warning: could not route dyke between %d and %d. Dyke ignored." % (n,end))
        
        #simulate dyke arrest and change dyke color and extract "bonds"
        bonds = []
        for i,n in enumerate(path):
            #change color
            self.model.G.nodes[n]["COL"] = "3"
            
            #build bond
            if i > 0:
                bonds.append( (path[i-1],path[i]) )
            
            #arrest dyke here?
            if np.random.rand() < self.p_arrest:
                return path[0:i],bonds
                
        #return dyke
        return path, bonds
        