

#
# Part class
#
#from collections import OrderedDict
from sortedcontainers import SortedSet, SortedDict
from multivac.pymln.semantic import Clust, Argument, ArgClust
from multivac.pymln.syntax.Relations import RelType

class Part(object):
    # SortedDict mapping {str: Part}
    #  - listing of all Part() objects by rootNodeId
    rootNodeId_part = SortedDict()
    # dictionary mapping {int: SortedSet{str: int}}
    clustIdx_partRootNodeIds = {}
    # dictionary mapping {(int, int): set((str, str))}
    pairClustIdxs_pairPartRootNodeIds = {}
    # dictionary mapping {int: set((int, int))}
    # clustIdx_pairClustIdxs = {}

    def getClustPartRootNodeIds():
        return Part.clustIdx_partRootNodeIds

    def getPairPartRootNodeIds(parClustIdx=None, chdClustIdx=None):
        if parClustIdx is None and chdClustIdx is None:
            return Part.pairClustIdxs_pairPartRootNodeIds
        elif parClustIdx is None:
            return {k: v for k, v in
                        Part.pairClustIdxs_pairPartRootNodeIds.items()
                        if k[1]==chdClustIdx}
        elif chdClustIdx is None:
            return {k: v for k, v in
                        Part.pairClustIdxs_pairPartRootNodeIds.items()
                        if k[0]==parClustIdx}
        else:
            if (parClustIdx, chdClustIdx) in Part.pairClustIdxs_pairPartRootNodeIds:
                return Part.pairClustIdxs_pairPartRootNodeIds[(parClustIdx,
                                                               chdClustIdx)]
            else:
                return None

    def getPartByRootNodeId(rnId):
        if rnId in Part.rootNodeId_part:
            return Part.rootNodeId_part[rnId]
        else:
            return None

    def getPartRootNodeIds(clustIdx):
        if clustIdx in Part.clustIdx_partRootNodeIds:
            return Part.clustIdx_partRootNodeIds[clustIdx]
        else:
            return None


    def __init__(self, relTreeRoot):
        self._relTreeRoot = relTreeRoot # TreeNode
        self._relTypeIdx = RelType.getRelType(self._relTreeRoot)
        self._clustIdx = -1
        self._nxtArgIdx = 0 # Remember next index because _args is OrderedDict

        self._parPart = None
        self._parArgIdx = -1

        # Dictionary mapping {int: Argument}
        self._args = SortedDict()
        # Dictionary mapping {int: int}
        self._argIdx_argClustIdx = {}
        # Dictionary mapping {int: set(int)}
        self._argClustIdx_argIdxs = {}

        Part.rootNodeId_part[self._relTreeRoot.getId()] = self

        return None

    def addArgument(self, arg):
        argIdx = self._nxtArgIdx
        self._nxtArgIdx += 1
        self._args[argIdx] = arg

        return argIdx

    def changeClust(self, newClustIdx, newRelTypeIdx, clust_only=False):
        oldClustIdx = self.getClustIdx()
        rootID = self.getRelTreeRoot().getId()
        Part.clustIdx_partRootNodeIds[oldClustIdx].discard(rootID)

        if clust_only:
            self._relTypeIdx = newRelTypeIdx
        else:
            ocl = Clust.getClust(oldClustIdx)
            ocl.onPartUnsetClust(self)
            self.setRelTypeIdx(newRelTypeIdx)

        self.setClust(newClustIdx, clust_only=clust_only)

        parent = self.getParPart()

        if parent is None:
            if newClustIdx in Clust.clustIdx_rootCnt:
                Clust.clustIdx_rootCnt[newClustIdx] += 1
            else:
                Clust.clustIdx_rootCnt[newClustIdx] = 1
            Clust.clustIdx_rootCnt[newClustIdx] -= 1
        else:
            parent_clust_id = parent.getClustIdx()
            paci = parent.getArgClust(self.getParArgIdx())
            pcl = Clust.getClust(parent_clust_id)
            pac = pcl._argClusts[paci]
            pac._chdClustIdx_cnt[oldClustIdx] -= 1

            if newClustIdx in pac._chdClustIdx_cnt:
                pac._chdClustIdx_cnt[newClustIdx] += 1
            else:
                pac._chdClustIdx_cnt[newClustIdx] = 1

            pa = (parent_clust_id, paci)
            Clust.clustIdx_parArgs[oldClustIdx][pa] -= 1

            if newClustIdx not in Clust.clustIdx_parArgs:
                Clust.clustIdx_parArgs[newClustIdx] = {}

            if pa in Clust.clustIdx_parArgs[newClustIdx]:
                Clust.clustIdx_parArgs[newClustIdx][pa] += 1
            else:
                Clust.clustIdx_parArgs[newClustIdx][pa] = 1

            opci = (parent_clust_id, oldClustIdx)
            npci = (parent_clust_id, newClustIdx)
            ptnid = (parent.getRelTreeRoot().getId(), rootID)

            Part.pairClustIdxs_pairPartRootNodeIds[opci].discard(ptnid)

            if len(Part.pairClustIdxs_pairPartRootNodeIds[opci]) == 0:
                del Part.pairClustIdxs_pairPartRootNodeIds[opci]
                # Part.clustIdx_pairClustIdxs[oldClustIdx].discard(opci)
                # Part.clustIdx_pairClustIdxs[parent_clust_id].discard(opci)

            if npci not in Part.pairClustIdxs_pairPartRootNodeIds:
                Part.pairClustIdxs_pairPartRootNodeIds[npci] = set()

            Part.pairClustIdxs_pairPartRootNodeIds[npci].add(ptnid)

            # Part.clustIdx_pairClustIdxs[parent_clust_id].add(npci)

            # if newClustIdx not in Part.clustIdx_pairClustIdxs:
            #     Part.clustIdx_pairClustIdxs[newClustIdx] = set()

            # Part.clustIdx_pairClustIdxs[newClustIdx].add(npci)

        return None

    def changeClustRemap(self, newClustIdx, argClustIdx_newArgClustIdx, clust_only=False):

        if not clust_only:
            oldClustIdx = self.getClustIdx()
            ocl = Clust.getClust(oldClustIdx)

        self.changeClust(newClustIdx, self.getRelTypeIdx(), clust_only=clust_only)

        argIdx_newArgClustIdx = {}

        for ai, arg in self._args.items():
            oaci = self._argIdx_argClustIdx.pop(ai)
            self._argClustIdx_argIdxs[oaci].remove(ai)

            if len(self._argClustIdx_argIdxs[oaci]) == 0:
                del self._argClustIdx_argIdxs[oaci]

            argIdx_newArgClustIdx[ai] = argClustIdx_newArgClustIdx[oaci]

            if not clust_only:
                ocl.onPartUnsetArg(self, arg, oaci)

        for ai in self._args:
            aci = argIdx_newArgClustIdx[ai]
            self.setArgClust(ai, aci, clust_only=clust_only)

        return None

    def destroy(self):
        tid = self.getRelTreeRoot().getId()
        Part.clustIdx_partRootNodeIds[self._clustIdx].discard(tid)

        if len(Part.clustIdx_partRootNodeIds[self._clustIdx]) == 0:
            del Part.clustIdx_partRootNodeIds[self._clustIdx]

        del Part.rootNodeId_part[tid]

        return None

    def getArgument(self, argIdx):
        return self._args[argIdx]

    def getArguments(self):
        return self._args

    def getArgClust(self, argIdx):
        if argIdx in self._argIdx_argClustIdx:
            return self._argIdx_argClustIdx[argIdx]
        else:
            return None

    def getParArgIdx(self):
        return self._parArgIdx

    def getClustIdx(self):
        return self._clustIdx

    def getParArgIdx(self):
        return self._parArgIdx

    def getParPart(self):
        return self._parPart

    def getRelTreeRoot(self):
        return self._relTreeRoot

    def getRelTypeIdx(self):
        return self._relTypeIdx

    def removeArgument(self, argIdx, clust_only=False):
        arg = self.getArgument(argIdx)

        oldArgClustIdx = self._argIdx_argClustIdx.pop(argIdx)
        self._argClustIdx_argIdxs[oldArgClustIdx].remove(argIdx)

        if len(self._argClustIdx_argIdxs[oldArgClustIdx]) == 0:
            del self._argClustIdx_argIdxs[oldArgClustIdx]

        if not clust_only:
            cl = Clust.getClust(self.getClustIdx())
            cl.onPartUnsetArg(self, arg, oldArgClustIdx)

        del self._args[argIdx]

        return None


    def setArgClust(self, argIdx, argClustIdx, clust_only=False):
        oldArgClustIdx = -1

        if argIdx in self._argIdx_argClustIdx:
            oldArgClustIdx = self.getArgClust(argIdx)

        if oldArgClustIdx != argClustIdx:
            self._argIdx_argClustIdx[argIdx] = argClustIdx

            if argClustIdx not in self._argClustIdx_argIdxs:
                self._argClustIdx_argIdxs[argClustIdx] = set()

            self._argClustIdx_argIdxs[argClustIdx].add(argIdx)
            arg = self.getArgument(argIdx)

            if not clust_only:
                cl = Clust.getClust(self._clustIdx)

            if oldArgClustIdx < 0:
                if not clust_only:
                    cl.onPartSetArg(self, arg, argClustIdx)
            else:
                self._argClustIdx_argIdxs[oldArgClustIdx].remove(argIdx)

                if len(self._argClustIdx_argIdxs[oldArgClustIdx]) == 0:
                    del self._argClustIdx_argIdxs[oldArgClustIdx]

                if not clust_only:
                    cl.onPartSetArg(self, arg, argClustIdx, oldArgClustIdx)

        return None

    def setClust(self, clustIdx, clust_only=False):
        self._clustIdx = clustIdx
        rootID = self.getRelTreeRoot().getId()

        if clustIdx not in Part.clustIdx_partRootNodeIds:
            Part.clustIdx_partRootNodeIds[clustIdx] = SortedSet()

        Part.clustIdx_partRootNodeIds[clustIdx].add(rootID)

        if not clust_only:
            cl = Clust.getClust(clustIdx)
            cl.onPartSetClust(self)

        return None

    def setParent(self, parPart, parArgIdx):
        '''
        Unset previous parent if it exists
        '''
        if self.getParPart() is not None:
            self.unsetParent()

        self._parPart = parPart
        self._parArgIdx = parArgIdx
        clustIdx = self.getClustIdx()
        parClustID = parPart.getClustIdx()

        assert (parClustID >= 0) & (clustIdx >= 0)

        pcci = (parClustID, clustIdx)

        # if parClustID not in Part.clustIdx_pairClustIdxs:
        #     Part.clustIdx_pairClustIdxs[parClustID] = set()

        # Part.clustIdx_pairClustIdxs[parClustID].add(pcci)

        # if clustIdx not in Part.clustIdx_pairClustIdxs:
        #     Part.clustIdx_pairClustIdxs[clustIdx] = set()

        # Part.clustIdx_pairClustIdxs[clustIdx].add(pcci)
        pids = (parPart.getRelTreeRoot().getId(), self.getRelTreeRoot().getId())

        if pcci not in Part.pairClustIdxs_pairPartRootNodeIds:
            Part.pairClustIdxs_pairPartRootNodeIds[pcci] = set()

        Part.pairClustIdxs_pairPartRootNodeIds[pcci].add(pids)

        if parPart is not None:
            arg = parPart.getArgument(parArgIdx)
            dep = arg._path.getDep()

            if (parClustID != clustIdx) & dep.startswith('conj_'):
                if parClustID < clustIdx:
                    pci = pcci
                else:
                    pci = (pcci[1], pcci[0])

                if pci not in Clust.pairClustIdx_conjCnt:
                    Clust.pairClustIdx_conjCnt[pci] = 1
                else:
                    Clust.pairClustIdx_conjCnt[pci] += 1

        return None

    def setRelTypeIdx(self, newRelTypeIdx):
        self._relTypeIdx = newRelTypeIdx
        cl = Clust.getClust(self._clustIdx)
        cl.onPartSetRelTypeIdx(newRelTypeIdx)

        return None

    def unsetArgClust(self, argIdx, clust_only=False):
        oldArgClustIdx = self._argIdx_argClustIdx.pop(argIdx)
        arg = self.getArgument(argIdx)
        self._argClustIdx_argIdxs[oldArgClustIdx].remove(argIdx)

        if len(self._argClustIdx_argIdxs[oldArgClustIdx]) == 0:
            del self._argClustIdx_argIdxs[oldArgClustIdx]

        if not clust_only:
            cl = Clust.getClust(self.getClustIdx())
            cl.onPartUnsetArg(self, arg, oldArgClustIdx)

        return None

    def unsetParent(self):
        '''
        Remove parent-child cluster index information
        Remove parent-child relationship index information
        '''
        parent = self.getParPart()
        clustIdx = self.getClustIdx()

        if parent is not None:
            parClustID = parent.getClustIdx()

            par_child_clust_pair = (parClustID, clustIdx)
            # Part.clustIdx_pairClustIdxs[parClustID].discard(par_child_clust_pair)
            # Part.clustIdx_pairClustIdxs[clustIdx].discard(par_child_clust_pair)

            part_pair = (parent.getRelTreeRoot().getId(),
                    self.getRelTreeRoot().getId())
            if par_child_clust_pair in Part.pairClustIdxs_pairPartRootNodeIds:
                Part.pairClustIdxs_pairPartRootNodeIds[par_child_clust_pair].discard(part_pair)

                if len(Part.pairClustIdxs_pairPartRootNodeIds[par_child_clust_pair]) == 0:
                    del Part.pairClustIdxs_pairPartRootNodeIds[par_child_clust_pair]

            arg = parent.getArgument(self.getParArgIdx())
            dep = arg._path.getDep()

            if (parClustID != clustIdx) & dep.startswith('conj_'):
                if parClustID < clustIdx:
                    conj_pair = par_child_clust_pair
                else:
                    conj_pair = (par_child_clust_pair[1], par_child_clust_pair[0])

                if conj_pair in Clust.pairClustIdx_conjCnt:
                    Clust.pairClustIdx_conjCnt[conj_pair] -= 1
                    if Clust.pairClustIdx_conjCnt[conj_pair] == 0:
                        del Clust.pairClustIdx_conjCnt[conj_pair]

        self._parPart = None
        self._parArgIdx = -1

        return None

    # def check_parents():
    #     mistakes = {}

    #     for nid, part in Part.rootNodeId_part.items():
    #         parent = part.getParPart()
    #         clustIdx = part.getClustIdx()

    #         if parent is not None:
    #             parClustID = parent.getClustIdx()
    #             pcci = (parClustID, clustIdx)

    #             if pcci not in Part.clustIdx_pairClustIdxs[parClustID]:
    #                 if nid not in mistakes:
    #                     mistakes[nid] = 1
    #                 else:
    #                     mistakes[nid] += 1

    #             if pcci not in Part.clustIdx_pairClustIdxs[clustIdx]:
    #                 if nid not in mistakes:
    #                     mistakes[nid] = 2
    #                 else:
    #                     mistakes[nid] += 2

    #     return mistakes


    def unsetRelTypeIdx(self):
        old_type = self._relTypeIdx
        cl = Clust.getClust(self._clustIdx)
        cl.onPartUnsetRelTypeIdx(old_type)

        return None



