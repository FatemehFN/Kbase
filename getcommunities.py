
import clusim.sim as csim
import leidenalg
from igraph import Graph
from clusim.clustering import Clustering, print_clustering
import pandas as pd
from infomap import Infomap
import numpy as np
import xnetwork as xn
import graph_tool as gt
import graph_tool.inference as gtInference
from graph_tool import Graph as gtGraph
from pathlib import Path 
from tqdm.auto import tqdm
from multiprocessing import Pool
from functools import partial


networksPath = Path("KNN_Networks")
networksWithCommunitiesPath = Path("KNN_NetworksWithCommunities")
networksWithCommunitiesPath.mkdir(parents=True,exist_ok=True)


def leiden(ROSMAP_graph,weights,resolution=1.0):
    best_q=None
    bestPartition = None
    for i in range(20):
        partition_ROSMAP = leidenalg.find_partition(ROSMAP_graph,leidenalg.RBConfigurationVertexPartition,
                                    resolution_parameter=resolution,weights=weights)
        
        q=partition_ROSMAP.quality()
        if best_q is None or q>best_q:
            bestPartition = partition_ROSMAP
            best_q=q
    
    return [str(entry) for entry in bestPartition.membership]





def infomapApply(g, m_time, weights=None):
    vertexCount = g.vcount()
    if(weights):
        edges = [(e.source, e.target, e[weights]) for e in g.es]
    else:
        edges = g.get_edgelist()

    if(g.is_directed()):
        extraOptions = "-d"
    else:
        extraOptions = ""
    im = Infomap("%s -N 10 --silent --seed %d" %
                (extraOptions, np.random.randint(4294967296)),markov_time=m_time)
    
    im.setVerbosity(0)
    for nodeIndex in range(0, vertexCount):
        im.add_node(nodeIndex)
    for edge in edges:
        if(len(edge) > 2):
            if(edge[2]>0):
                im.addLink(edge[0], edge[1], edge[2])
            im.add_link(edge[0], edge[1], weight=edge[2])
        else:
            im.add_link(edge[0], edge[1])

    im.run()
    membership = [":".join([str(a) for a in membership])
                for index, membership in im.get_multilevel_modules().items()]

    levelMembership = []

    #print(max([len(element.split(":")) for element in membership]))
    levelCount = max([len(element.split(":")) for element in membership])
    for level in range(levelCount):
        levelMembership.append(
            [":".join(element.split(":")[:(level+1)]) for element in membership]
        )
    return levelMembership





resolutions = [0.001,0.005,0.01,0.1,1.0,0.5,5.0,10.0,20.0,50,100]
markovTimes = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,5,10]


def calculateMetrics(g,propertyName):
    fromCluster = Clustering()
    fromCluster.from_membership_list(g.vs["CellType"])
    toCluster = Clustering()
    toCluster.from_membership_list(g.vs[propertyName])
    adjRandIndex = csim.adjrand_index(fromCluster,toCluster)
    adjMutualInfo = csim.adj_mi(fromCluster,toCluster)
    return {
        "ARI": adjRandIndex,
        "AMI": adjMutualInfo
        }

def processnetwork(networkFile):
    #print(networkFile)
    g = xn.load(networkFile)



    for resolution in (resolutions):
        g.vs["Leiden_weighted_%f"%resolution] = leiden(g,weights="weight",resolution=resolution)
        g.vs["Leiden_unweighted_%f"%resolution] = leiden(g,weights=None,resolution=resolution)




    # for markovTime in markovTimes:
    #      levelInfomapWeighted = infomapApply(g,markovTime,weights="weight")
    #      for levelIndex,levelData in enumerate(levelInfomapWeighted):
    #          g.vs["Infomap_weighted_%f_%d"%(markovTime,levelIndex)] = levelData

    #      levelInfomapUnweighted = infomapApply(g,markovTime,weights=None)
    #      for levelIndex,levelData in enumerate(levelInfomapUnweighted):
    #          g.vs["Infomap_unweighted_%f_%d"%(markovTime,levelIndex)] = levelData


    xn.save(g,networksWithCommunitiesPath/networkFile.name)

    del g
    
    



if __name__ == "__main__":



    networkFiles = list(networksPath.glob("*.xnet"))
    with Pool(16) as p:
        for _ in tqdm(p.imap_unordered(processnetwork, networkFiles), total=len(networkFiles)):
            pass

