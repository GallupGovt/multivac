
from semantic import Clust, ArgClust, Part
from syntax.Relations import ArgType

import json
import os

class MLN(object):
    '''
    Class for simply outputting the MLN structure parsed from source 
    documents.

    NEEDS FILE OUTPUT COMPONENT STILL

    '''
    def __init__(self):
        return None

    def printModel(path):
        clustering = MLN.printClustering(path)
        mln = MLN.printMLN(path)
        prse = MLN.printParse(path)

        if path is None:
            return clustering, mln, prse
        else:
            return None

    def printClustering(path=None):
        out_str = "=== Clustering ===\n"

        for ci in Clust.clusts:
            cl = Clust.getClust(ci)
            if len(cl._relTypeIdx_cnt) > 1:
                out_str += cl.toString() + "\n"

        if path is not None:
            dst = "{}/{}.clustering".format(path, 
                                            os.path.basename(os.path.dirname(path)))
            with open(dst, 'w') as f:
                f.write(out_str)

            return None
        else:
            return out_str

    def saveMLN(path=None):
        if path is not None:
            dst = "{}/{}.mln.json".format(path, 
                                     os.path.basename(os.path.dirname(path)))
            with open(dst, 'w') as f:
                pickle.dump(Clust.clusts,f)

        return None


    def printMLN(path=None):
        out_str = ""

        for ci in Clust.clusts:
            cl = Clust.getClust(ci)
            out_str += "{}\t{}\n".format(cl._clustIdx,cl)

            for aci in cl._argClusts:
                ac = cl._argClusts[aci]
                out_str += "\t{}: ".format(aci)

                out_str += "\t".join(["{}: {}".format(k, v) 
                                      for k, v in ac._argNum_cnt])
                out_str += "\n\t"
                out_str += "\t".join(["{}: {}".format(k, v) 
                                      for k, v in ac._argTypeIdx_cnt])
                out_str += "\n\t"
                out_str += "\t".join(["{}: {}: {}".format(k, Clust.getClust(k), v) 
                                      for k, v in ac._chdClustIdx_cnt])
                out_str += "\n"

        if path is not None:
            dst = "{}/{}.mln".format(path, 
                                     os.path.basename(os.path.dirname(path)))
            with open(dst, 'w') as f:
                f.write(out_str)

            return None
        else:
            return out_str


    def printParse(path=None):
        out_str = "=== Parse ===\n"

        for rnid, pt in Part._rootNodeId_part.items():
            out_str += "{}\t{}\n".format(rnid, pt._relTreeRoot.getTreeStr())
            out_str += "\t{}: {}\n".format(pt._clustIdx,
                                           Clust.getClust(pt._clustIdx).toString())

            if pt._parPart is None:
                out_str += "\t\n\t\n"
            else:
                arg = pt._parPart.getArgument(pt._parArgIdx)
                out_str += "\t{}: {}: {}\n".format(pt._parPart._relTreeRoot.getId(),
                                                   pt._parPart._clustIdx,
                                                   Clust.getClust(pt._parPart._clustIdx))
                out_str += "\t{}: {}: {}\n".format(pt._parPart.getArgClust(pt._parArgIdx),
                                                   arg._path.getArgType(),
                                                   ArgType.getArgType(arg._path.getArgType()))

        if path is not None:
            dst = "{}/{}.parse".format(path, 
                                       os.path.basename(os.path.dirname(path)))
            with open(dst, 'w') as f:
                f.write(out_str)

            return None
        else:
            return out_str

