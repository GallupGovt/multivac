
from semantic import SearchOp, Clust, Part, Scorer
from syntax.Relations import RelType
from utils import java_iter

class Executor(object):
    def __init__(self, parse):
        self._parse = parse

    def executeOp(self, op):
        if op._op == SearchOp.OP_MERGE_CLUST:
            nci = self.execMC(op)
        elif op._op == SearchOp.OP_COMPOSE:
            nci = self.execCompose(op)
        else:
            nci = -1

        return nci

    def execMC(self, op):
        clustIdx1 = op._clustIdx1
        clustIdx2 = op._clustIdx2

        scorer = self._parse._scorer
        nci = clustIdx1

        cl1 = Clust.getClust(clustIdx1)
        cl2 = Clust.getClust(clustIdx2)

        if cl1 is None or cl2 is None:
            return -1

        clx1 = cl1
        clx2 = cl2

        if len(cl1._argClusts) < len(cl2._argClusts):
            clx1 = cl2
            clx2 = cl1
            nci = clustIdx2

        aci2_aci1 = dict()
        scorer.scoreMCForAlign(clx1, clx2, aci2_aci1)

        for aci2 in clx2._argClusts:
            if aci2 not in aci2_aci1:
                ac2 = clx2._argClusts[aci2]

                for ati in ac2._argTypeIdx_cnt:
                    aci1 = clx1.createArgClust(ati)
                    aci2_aci1[aci2] = aci1
                    break

        pids2 = set()
        pids2.update(Part.getPartRootNodeIds(clx2.getId()))

        for rnId in pids2:
            pt = Part.getPartRootNodeId(rnId)

            for argIdx in pt.getArguments():
                pt.getArgument(argIdx)._argPart.unsetParent()

            pt.changeCluster(clx1.getId(), aci2_aci1)

            for argIdx in pt.getArguments():
                pt.getArgument(argIdx)._argPart.setParent(pt, argIdx)

        Clust.removeClust(clx2)

    return nci

    def execCompose(self, op):
        parClustIdx = op._parClustIdx
        chdClustIdx = op._chdClustIdx
        nci = -1

        pcl = Clust.getClust(parClustIdx)
        ccl = Clust.getClust(chdClustIdx)

        if pcl is None or ccl is None:
            return -1

        ncl = None
        prnids = set()
        prnids.update(Part.pairClustIdxs_pairPartRootNodeIds[(parClustIdx, 
                                                              chdClustIdx)])

        for rnids in prnids:
            pid, cid = rnids

            pp = Part.getPartRootNodeId(pid)
            cp = Part.getPartRootNodeId(cid)
            dep = pp.getArguments()[cp._parArgidx]._path.getDep()
            pp._relTreeRoot.addChild(dep, cp._relTreeRoot)
            nrti = RelType.getRelType(pp._relTreeRoot)

            if ncl is None:
                # on first loop
                if Clust.getClustsWithRelType(nrti) is None:
                    ncl = Clust.getClust(Clust.createClust(nrti))
                elif len(Clust.getClustsWithRelType(nrti)) > 1:
                    raise MultipleClustersSameRelType Exception
                else:
                    ncl = Clust.getClust(Clust.getClustsWithRelType(nrti).next())

                nci = ncl.getId()

            pp.removeArgument(cp._parArgidx)

            if pp.getId() != nci:
                for argIdx in pp.getArguments():
                    pp.unsetArgClust(argIdx)
                    arg = pp.getArgument(argIdx)
                    arg._argPart.unsetParent()

                pp.changeCluster(nci, nrti)

                for argIdx, arg in pp.getArguments().items():
                    ati = arg._path.getArgType()
                    aci = -1

                    if ati not in ncl._argTypeIdx_argClustIdxs:
                        aci = ncl.createArgClust(ati)
                    elif len(ncl._argTypeIdx_argClustIdxs[ati]) == 0:
                        aci = ncl.createArgClust(ati)
                    else:
                        aci = ncl._argTypeIdx_argClustIdxs[ati].next()

                    arg._argPart.setParent(pp, argIdx)
                    pp.setArgClust(argIdx, aci)

                pp.setRelType(nrti)
            else:
                pp.unsetRelType()
                pp.setRelType(nrti)

            for argIdx, arg in cp.getArguments():
                ati = arg._path.getArgType()
                aci = -1

                if ati not in ncl._argTypeIdx_argClustIdxs:
                    aci = ncl.createArgClust(ati)
                elif len(ncl._argTypeIdx_argClustIdxs[ati]) == 0:
                    aci = ncl.createArgClust(ati)
                else:
                    aci = ncl._argTypeIdx_argClustIdxs[ati].next()

                cp.unsetArgClust(argIdx)
                pp.setArgClust(pp.addArgument(arg), aci)
                arg._argPart.setParent(pp, pp.addArgument(arg))

            cp.destroy()

        Part.clustIdx_pairClustIdxs[parClustIdx].remove(pci)
        Part.clustIdx_pairClustIdxs[chdClustIdx].remove(pci)
        del Part.pairClustIdxs_pairPartRootNodeIds[pci]

        return nci

    def execComposePart(self, pp, cp):
        parClustIdx = pp._clustIdx
        chdClustIdx = cp._clustIdx
        pcl = Clust.getClust(parClustIdx)
        ccl = Clust.getClust(chdClustIdx)
        dep = pp.getArguments()[cp._parArgidx]._path.getDep()
        pp._relTreeRoot.addChild(dep, cp._relTreeRoot)
        nrti = RelType.getRelType(pp._relTreeRoot)

        ncl = Clust.getClust(Clust.getClustsWithRelType(nrti).next())
        nci = ncl.getId()

        pp.removeArgumentOnly(cp._parArgidx)

        for argIdx, arg in pp.getArguments().items():
            pp.unsetArgClust(argIdx)
            arg._argPart.unsetParent()

        pp.changeClusterOnly(nci, nrti)

        for argIdx, arg in pp.getArguments().items():
            ati = arg._path.getArgType()
            aci = -1

            if ati not in ncl._argTypeIdx_argClustIdxs:
                aci = ncl.createArgClust(ati)
            elif len(ncl._argTypeIdx_argClustIdxs[ati]) == 0:
                aci = ncl.createArgClust(ati)
            else:
                aci = ncl._argTypeIdx_argClustIdxs[ati].next()

            arg._argPart.setParent(pp, argIdx)
            pp.setArgClustOnly(argIdx, aci)

        pp.setRelType(nrti)

        for argIdx, arg in cp.getArguments():
            ati = arg._path.getArgType()
            aci = -1

            if ati not in ncl._argTypeIdx_argClustIdxs:
                aci = ncl.createArgClust(ati)
            elif len(ncl._argTypeIdx_argClustIdxs[ati]) == 0:
                aci = ncl.createArgClust(ati)
            else:
                aci = ncl._argTypeIdx_argClustIdxs[ati].next()

            cp.unsetArgClustOnly(argIdx)
            pp.setArgClust(pp.addArgument(arg), aci)
            arg._argPart.setParent(pp, pp.addArgument(arg))

        cp.destroy()

        return None

    def mergeArg(self, clust, aci1, aci2):
        ac1, ac2 = [clust._argClusts[x] for x in (aci1, aci2)]
        ids = java_iter(ac2._partRootTreeNodeIds)

        while ids.hasnext():
            idx = ids.next()
            p = Part.getPartRootNodeId(idx)
            iit = java_iter(p._argIdx_argClustIdx.keys())

            while iit.hasnext():
                ai = iit.next()
                acix = p._argIdx_argClustIdx[ai]

                if acix == aci2:
                    p.setArgClust(ai, aci1)

        return None































