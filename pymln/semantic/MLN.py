
from semantic import Clust, ArgClust, Part
from syntax.Relations import ArgType, RelType

import json
import pickle
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

        for ci, clust in Clust.clusts.items():
            # if len(clust._relTypeIdx_cnt) > 1:
            out_str += str(ci) + " " + clust.toString() + "\n"
            for aci, ac in clust._argClusts.items():
                out_str += "\t{}\t{}\t{}\n".format(aci, ac.toString(), ac._ttlArgCnt)

        if path is not None:
            dst = "{}/{}.clustering".format(path, 
                                            os.path.basename(os.path.dirname(path)))
            with open(dst, 'w') as f:
                f.write(out_str)

            return None
        else:
            return out_str

    def save_mln(path):
        '''
            Save all objects necessary to recreate the MLN knowledgebase
        '''
        with open(path, 'wb') as f:
            pickle.dump({'clusts': Clust.clusts,
                         'relTypeIdx_clustIdx': Clust.relTypeIdx_clustIdx,
                         'relTypes': RelType.relTypes,
                         'relTypeStr_idx': RelType.relTypeStr_idx,
                         'argTypes': ArgType.argTypes,
                         'argTypeStr_idx': ArgType.argTypeStr_idx,
                         'rootNodeId_part': Part.rootNodeId_part,
                         'clustIdx_partRootNodeIds': Part.clustIdx_partRootNodeIds,
                         'pairClustIdxs_pairPartRootNodeIds': Part.pairClustIdxs_pairPartRootNodeIds}, 
                        f)

        return None

    def load_mln(path):
        with open(path, 'rb') as f:
            mln = pickle.load(f)

        try:
            _ = len(Clust.clusts)
            _ = len(ArgType.argTypes)
            _ = len(RelType.relTypes)
            _ = len(Part.rootNodeId_part)
        except NameError:
            from semantic import Clust, Part
            from syntax.Relations import ArgType, RelType
        finally:
            Clust.clusts = mln['clusts']
            Clust.relTypeIdx_clustIdx = mln['relTypeIdx_clustIdx']
            RelType.relTypes = mln['relTypes']
            RelType.relTypeStr_idx = mln['relTypeStr_idx']
            ArgType.argTypes = mln['argTypes']
            ArgType.argTypeStr_idx = mln['argTypeStr_idx']
            Part.rootNodeId_part = mln['rootNodeId_part']
            Part.clustIdx_partRootNodeIds = mln['clustIdx_partRootNodeIds']
            Part.pairClustIdxs_pairPartRootNodeIds = mln['pairClustIdxs_pairPartRootNodeIds']
            Part.clustIdx_pairClustIdxs = mln['clustIdx_pairClustIdxs']

        return None

    def printMLN(path=None):
        if path is not None:
            dst = "{}/{}.mln".format(path, 
                                     os.path.basename(os.path.dirname(path)))
            with open(dst, 'w') as f:
                for ci in Clust.clusts:
                    cl = Clust.getClust(ci)
                    out_str = "{}\t{}\n".format(cl._clustIdx,cl)

                    for aci in cl._argClusts:
                        ac = cl._argClusts[aci]
                        out_str += "\t{}: ".format(aci)

                        out_str += "\t".join(["{}: {}".format(k, v) 
                                              for k, v in ac._argNum_cnt.items()])
                        out_str += "\n\t"
                        out_str += "\t".join(["{}: {}: {}".format(k, ArgType.getArgType(k).toString(), v) 
                                              for k, v in ac._argTypeIdx_cnt.items()])
                        out_str += "\n\t"
                        out_str += "\t".join(["{}: {}: {}".format(k, Clust.getClust(k), v) 
                                              for k, v in ac._chdClustIdx_cnt.items()])
                        out_str += "\n"

                    f.write(out_str)

            return None
        else:
            return out_str


    def printParse(path=None):
        out_str = ""

        for rnid, pt in Part.rootNodeId_part.items():
            out_str += "{}\t{}\n".format(rnid, pt._relTreeRoot.getTreeStr())
            out_str += "\t{}: {}\n".format(pt._clustIdx,
                                           Clust.getClust(pt._clustIdx).toString())

            if pt._parPart is None:
                out_str += "\t\n\t\n"
            else:
                arg = pt._parPart.getArgument(pt._parArgIdx)
                out_str += "\t{}\t{}\t{}\n".format(pt._parPart._relTreeRoot.getId(),
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

