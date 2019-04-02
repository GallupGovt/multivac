
from multivac.pymln.semantic import SearchOp, Clust, Part, Scorer
from multivac.pymln.syntax.Relations import RelType

class Executor(object):
    def __init__(self, parse):
        self._parse = parse

    def executeOp(self, op):
        if op._op == SearchOp.OP_MERGE_CLUST:
            new_clust_id = self.execMC(op)
        elif op._op == SearchOp.OP_COMPOSE:
            new_clust_id = self.execCompose(op)
        else:
            new_clust_id = -1

        return new_clust_id

    def execMC(self, op):
        #
        # Get clusters associated with our op
        #

        cluster1 = Clust.getClust(op._clustIdx1)
        cluster2 = Clust.getClust(op._clustIdx2)

        if cluster1 is None or cluster2 is None:
            return -1

        #
        # If cluster 1 has fewer argument clusters than cluster 2, swap them.
        # We merge the "smaller" cluster into the larger one.
        #

        if len(cluster1._argClusts) < len(cluster2._argClusts):
            clust_swap = cluster2
            cluster2 = cluster1
            cluster1 = clust_swap

        #
        # Align the argument clusters based on scores, and then map over
        # any remaining argument clusters from cluster 2 to cluster 1.
        #

        aci2_aci1 = dict()
        scorer = self._parse.scorer
        _, aci2_aci1 = scorer.scoreMCForAlign(cluster1, cluster2, aci2_aci1)

        for arg_clust_id2 in cluster2._argClusts:
            if arg_clust_id2 not in aci2_aci1:
                arg_clust = cluster2._argClusts[arg_clust_id2]

                for arg_type in arg_clust._argTypeIdx_cnt:
                    arg_clust_ids = cluster1.getArgClustIdxs(arg_type)

                    if arg_clust_ids is None:
                        arg_clust_id1 = cluster1.createArgClust(arg_type)
                    else:
                        arg_clust_id1 = next(iter(arg_clust_ids))

                    aci2_aci1[arg_clust_id2] = arg_clust_id1
                    break

        #
        # Finally, remap the Parts in cluster 2 to cluster 1 as well.
        #

        part_ids = set()
        part_ids.update(Part.getPartRootNodeIds(cluster2.getId()))

        for part_id in part_ids:
            pt = Part.getPartByRootNodeId(part_id)

            for arg in pt.getArguments().values():
                arg._argPart.unsetParent()

            pt.changeClustRemap(cluster1.getId(), aci2_aci1)

            for argIdx, arg in pt.getArguments().items():
                arg._argPart.setParent(pt, argIdx)

        Clust.removeClust(cluster2)

        return cluster1.getId()

    def execCompose(self, op):
        parClustIdx = op._parClustIdx
        chdClustIdx = op._chdClustIdx
        new_clust_id = -1

        #
        # If either cluster are None, return -1
        #
        if Clust.getClust(parClustIdx) is None or Clust.getClust(chdClustIdx) is None:
            return -1

        new_clust = None
        parent_child_pair = (parClustIdx, chdClustIdx)
        part_ids = set()
        part_ids.update(Part.pairClustIdxs_pairPartRootNodeIds[parent_child_pair])

        deleted_parts = []

        for parent_id, child_id in part_ids:
            if parent_id in deleted_parts or child_id in deleted_parts:
                continue

            parent_part = Part.getPartByRootNodeId(parent_id)
            child_part = Part.getPartByRootNodeId(child_id)
            dep = parent_part.getArguments()[child_part._parArgIdx]._path.getDep()
            parent_part._relTreeRoot.addChild(dep, child_part._relTreeRoot)
            nrti = RelType.getRelType(parent_part._relTreeRoot)

            if new_clust is None:
                # on first loop
                rel_clusts = Clust.getClustsWithRelType(nrti)
                if rel_clusts is None:
                    new_clust = Clust.getClust(Clust.createClust(nrti))
                elif len(rel_clusts) > 1:
                    raise Exception
                else:
                    new_clust = Clust.getClust(next(iter(rel_clusts)))

                new_clust_id = new_clust.getId()

            parent_part.removeArgument(child_part._parArgIdx)

            if parent_part.getClustIdx() != new_clust_id:
                for argIdx in parent_part.getArguments():
                    parent_part.unsetArgClust(argIdx)
                    arg = parent_part.getArgument(argIdx)
                    arg._argPart.unsetParent()

                parent_part.changeClust(new_clust_id, nrti)

                for argIdx, arg in parent_part.getArguments().items():
                    arg_type = arg._path.getArgType()
                    arg_clust_id = -1

                    if arg_type not in new_clust._argTypeIdx_argClustIdxs:
                        arg_clust_id = new_clust.createArgClust(arg_type)
                    elif len(new_clust._argTypeIdx_argClustIdxs[arg_type]) == 0:
                        arg_clust_id = new_clust.createArgClust(arg_type)
                    else:
                        arg_clust_id = next(iter(new_clust._argTypeIdx_argClustIdxs[arg_type]))

                    arg._argPart.setParent(parent_part, argIdx)
                    parent_part.setArgClust(argIdx, arg_clust_id)

                parent_part.setRelTypeIdx(nrti)
            else:
                parent_part.unsetRelTypeIdx()
                parent_part.setRelTypeIdx(nrti)

            #
            # Connect the child part's arguments directly to the parent part now
            #

            for argIdx, arg in child_part.getArguments().items():
                child_part.unsetArgClust(argIdx)
                arg_type = arg._path.getArgType()
                arg_clust_id = -1

                if arg_type not in new_clust._argTypeIdx_argClustIdxs:
                    arg_clust_id = new_clust.createArgClust(arg_type)
                elif len(new_clust._argTypeIdx_argClustIdxs[arg_type]) == 0:
                    arg_clust_id = new_clust.createArgClust(arg_type)
                else:
                    arg_clust_id = next(iter(new_clust._argTypeIdx_argClustIdxs[arg_type]))

                newArgIdx = parent_part.addArgument(arg)
                arg._argPart.setParent(parent_part, newArgIdx)
                parent_part.setArgClust(newArgIdx, arg_clust_id)

            #
            # Remove the old child part
            #

            deleted_parts.append(child_part.getRelTreeRoot().getId())
            child_part.destroy()

        # Part.clustIdx_pairClustIdxs[parClustIdx].remove(pci)
        # Part.clustIdx_pairClustIdxs[chdClustIdx].remove(pci)
        del Part.pairClustIdxs_pairPartRootNodeIds[parent_child_pair]

        return new_clust_id

    def execComposePart(self, pp, cp):
        parClustIdx = pp._clustIdx
        chdClustIdx = cp._clustIdx
        pcl = Clust.getClust(parClustIdx)
        ccl = Clust.getClust(chdClustIdx)
        dep = pp.getArguments()[cp._parArgIdx]._path.getDep()
        pp._relTreeRoot.addChild(dep, cp._relTreeRoot)
        nrti = RelType.getRelType(pp._relTreeRoot)

        ncl = Clust.getClust(next(iter(Clust.getClustsWithRelType(nrti))))
        nci = ncl.getId()

        pp.removeArgument(cp._parArgIdx, clust_only=True)

        for argIdx, arg in pp.getArguments().items():
            pp.unsetArgClust(argIdx)
            arg._argPart.unsetParent()

        pp.changeClust(nci, nrti, clust_only=True)

        for argIdx, arg in pp.getArguments().items():
            ati = arg._path.getArgType()
            aci = -1

            if ati not in ncl._argTypeIdx_argClustIdxs:
                aci = ncl.createArgClust(ati)
            elif len(ncl._argTypeIdx_argClustIdxs[ati]) == 0:
                aci = ncl.createArgClust(ati)
            else:
                aci = next(iter(ncl._argTypeIdx_argClustIdxs[ati]))

            arg._argPart.setParent(pp, argIdx)
            pp.setArgClustOnly(argIdx, aci)

        pp.setRelTypeIdx(nrti)

        for argIdx, arg in cp.getArguments():
            ati = arg._path.getArgType()
            aci = -1

            if ati not in ncl._argTypeIdx_argClustIdxs:
                aci = ncl.createArgClust(ati)
            elif len(ncl._argTypeIdx_argClustIdxs[ati]) == 0:
                aci = ncl.createArgClust(ati)
            else:
                aci = next(iter(ncl._argTypeIdx_argClustIdxs[ati]))

            cp.unsetArgClustOnly(argIdx)
            pp.setArgClust(pp.addArgument(arg), aci)
            arg._argPart.setParent(pp, pp.addArgument(arg))

        cp.destroy()

        return None

    def mergeArg(self, clust, aci1, aci2):
        ac2 = clust._argClusts[aci2]

        for node_id in ac2._partRootTreeNodeIds.copy():
            part = Part.getPartByRootNodeId(node_id)

            for arg_id, arg_clust_id in part._argIdx_argClustIdx.items():
                if arg_clust_id == aci2:
                    part.setArgClust(arg_id, aci1)

        return None




