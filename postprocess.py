"""
A series of functions for postprocessing riceball data to extract and plot
quantities such as average stress and strain.

Sam Thiele 2018
"""
import numpy as np
from scipy.stats import binned_statistic_2d as bin2d
import matplotlib.pyplot as plt
import matplotlib as mpl

###################
#GRIDDING FUNCTIONS
###################
"""
Grid a quantity assigned to networkX nodes.

**Arguments**:
 - graph = the networkx graph containing the nodes
 - nodes = a list of the node id's (keys) to extract data from
 - pos = a dictionary defining the position for each node
 - quantity = the key that returns the quantity desired. If this 
              key returns a scalar, the scalar is gridded. If it returns a vector or tensor,
              the vector/tensor is gridded. If a list is passed, each of these are evaluated as scalar field and then
              the result is stacked to form a vector grid.
 - cellsize = the size of each grid-cell (in both the x and y directions)
"""
def gridNodeData(graph, nodes, pos, quantity, cellsize,dynamic=True):
    #load data as [x,y,[d1,d2,d3 ... ]]
    d = [] #list of data objects (scalar, vector, tensor)
    p = [] #position for each data object
    for n in nodes:
        if not quantity in graph.nodes[n]:
            continue #ignore if key doesn't exist for this node
        if graph.nodes[n]["TFIXED"] == '111': #static ball - ignore
            continue
        p.append(pos[n])
        
        q = graph.nodes[n][quantity]
        if not (isinstance(q,float) or isinstance(q,np.ndarray)):
            try:
                q = float(q) #convert strings to float (or throw error if cannot be float)
            except:
                assert False, "Error: Could not grid data of type %s" % type(q)
        d.append(q)
    
    assert len(d) != 0, "Invalid quantity. Please check node keys (key not found)"
    
    p = np.array(p).T
    d = np.array(d)
    
    assert d.shape[0] == p.shape[1], "Error - inconsistent data sizes."
    
    #compute binning parameters
    xrange = np.ptp(p[0])
    yrange = np.ptp(p[1])
   
    nx = int(xrange / cellsize)
    ny = int(yrange / cellsize)
    
    #bin samples, compute averages and return
    if len(d.shape) == 1: #scalar property
        grid = bin2d(p[0],p[1],d,bins=[nx,ny])
        return grid.statistic
    elif len(d.shape) == 2: #vector property
        components = []
        for c in d.T: #loop through each component
            components.append(bin2d(p[0],p[1],c,bins=[nx,ny]).statistic) #grid component
        return components #return list of vector fields
    elif len(d.shape) == 3: #tensor property (probably stress)
        #what to do here?
        return []
"""
Grids a quantity that is assigned to nodes within a riceball model

**Arguments**:
- model = A RiceBall instance defining the model to grid
- quantity = the key that returns the quantity desired. E.g. "sigma1","V.x". If it returns a vector,
              the vector is gridded.
- dynamic = If true, only dynamic particles (those that are allowed to move) are plotted. Default is true.
"""
def gridModel( model, quantity, cellsize, dynamic=True):
    if not isinstance(quantity,list): #single quantity plotted
        return gridNodeData( model.G, model.G.nodes, model.pos, quantity, cellsize, dynamic )
    else: #quantity is a list - stack output
        stack = []
        for q in quantity:
            stack.append(gridNodeData( model.G, model.G.nodes, model.pos, q, cellsize, dynamic ))
        return stack #list of lists.

"""
Quickly plot a grid dataset using matplotlib

**Arguments**:
-figsize = The size tuple of the matplotlib figure. Default is (18,5).
-hold = If true, plt.show() isn't called, so stuff can be added to the plot later. Default is False.
**Keywords**:
Keywards are passed directly to matplotlib.pyplot.imshow(...).
"""
def quickPlot( grid, figsize=(18,5), hold=False, **kwds ):
    plt.figure(figsize=figsize)
    plt.imshow(grid.T, origin="lower",**kwds)
    if "cmap" in kwds:
        plt.colorbar()
    if not hold:
        plt.show()
        
"""
Quickly(ish) plot a quiver plot of a vector field.

**Arguments**:
-grid = the vector field grid as returned by gridNodeData(...). Should be of the form [x,y,c] where c are the vector components and x,y are the position grid coords.
-normed = True if vectors are normalized befeore plotting. This will only plot vector directions and ignore magnitude. Default is True.
-cmap = the colour map to plot the vector magnitude with. Default is "plasma". If None, colours are not used.
-figsize = the size of the figure/plot created. Defautl is (18,5).
-hold = if set to True, the plot is not shown so it can be modified by future pyplot commands.
**Keywords**:
-All keywords are passed directly to matplotlib.pyplot.quiver(...). Refer to the documentation for this function for keywords.
"""
def quickPlotV( grid, normed=True, cmap="plasma", figsize=(18,5),hold=False,**kwds):
    assert len(grid.shape) == 3, "Error - grid is the wrong shape. It should have the format [x,y,c] where c are the vector components and x,y are position coords."
    assert grid.shape[2] >= 2, "Error - please pass a vector field of the format [x,y,c] where c = 0 for u component of vectors and 1 for v component of vectors."
    #extract components
    u = grid[:,:,0].ravel()
    v = grid[:,:,1].ravel()

    l = np.sqrt( u**2 + v**2 )
    if normed:
        u /= l
        v /= l
    
    #build grid of points
    x = np.linspace(0,grid.shape[0],grid.shape[0])
    y = np.linspace(0,grid.shape[1],grid.shape[1])
    X,Y = np.meshgrid(x,y,indexing='ij')
    
    #plot
    f = plt.gcf()
    if f is None:
        f = plt.figure(figsize=figsize)
    f.set_figwidth(figsize[0])
    f.set_figheight(figsize[1])
    if cmap is None:
        plt.quiver(X.ravel(),Y.ravel(),u,v,**kwds)
    else:
        #make color map
        cmap = plt.get_cmap(cmap)
        norm = mpl.colors.Normalize(vmin=min(l),vmax=max(l))
        plt.quiver(X.ravel(),Y.ravel(),u,v,color=cmap(norm(l)),**kwds)

        #and color bar
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(l)
        plt.colorbar(sm)