
from syntax.Relations import RelType


class MLN(object):
	def __init__(self):
		return None


class Argument(object):
	def __init__(self, argNode, path, argPart):
		self._argNode = argNode
		self._path = path
		self._argPart = argPart

		return None

	def getPath(self):
		return self._path

	def getPart(self):
		return self._argPart

	def getNode(self):
		return self._argNode


#
# Part class
# 

class Part(object):
    # dictionary mapping {str: Part}
    rootNodeId_part = {}
    # dictionary mapping {int: set(str)}
    clustIdx_partRootNodeIds = {}
    # dictionary mapping {(int, int): set((str, str))}
    pairClustIdxs_pairPartRootNodeIds = {}
    # dictionary mapping {int: set((int, int))}
    clustIdx_pairClustIdxs = {}

    def __init__(self, relTreeRoot):
        self._isDebug = False

        self._relTreeRoot = relTreeRoot
        self._relTypeIdx = RelType.getRelType(relTreeRoot)
        self._clustIdx = -1
        self._nxtArgIdx = 0 # Remember next index because _args should be ordered Dict

        self._parPart = None
        self._parArgIdx = -1

        # Dictionary mapping {int: Argument}
        self._args = {}
        # Dictionary mapping {int: int}
        self._argIdx_argClustIdx = {}
        # Dictionary mapping {int: set(int)}
        self._argClustIdx_argIdxs = {}

        return None

    def addArgument(self, arg):
        argIdx = self._nxtArgIdx + 1
        self._args[argIdx] = arg

        return argIdx

    def changeClust(self, newClustIdx, newRelTypeIdx, clust_only=False):
        oldClustIdx = self.getClustIdx()
        rootID = self.getRelTreeRoot().getId()
        Part.clustIdx_partRootNodeIds[oldClustIdx].remove(rootID)

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

            Part.pairClustIdxs_pairPartRootNodeIds[opci].remove(ptnid)

            if len(Part.pairClustIdxs_pairPartRootNodeIds[opci]) == 0:
                Part.clustIdx_pairClustIdxs[oldClustIdx].remove(opci)
                Part.clustIdx_pairClustIdxs[parent_clust_id].remove(opci)

            if npci in Part.pairClustIdxs_pairPartRootNodeIds:
                Part.pairClustIdxs_pairPartRootNodeIds[npci].add(ptnid)
            else:
                Part.pairClustIdxs_pairPartRootNodeIds[npci] = set(ptnid)

            Part.clustIdx_pairClustIdxs[parent_clust_id] = npci
            if newClustIdx in Part.clustIdx_pairClustIdxs:
                Part.clustIdx_pairClustIdxs[newClustIdx].add(npci)
            else:
                Part.clustIdx_pairClustIdxs[newClustIdx] = set(npci)

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
                ocl.onPartUnsetArg(this, arg, oaci)

        for ai in self._args:
            aci = argIdx_newArgClustIdx[ai]
            self.setArgClust(ai, aci, clust_only=clust_only)

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

    def getClustPartRootNodeIds():
        return Part.clustIdx_partRootNodeIds

    def getPairPartRootNodeIds(parClustIdx=None, chdClustIdx=None):
        if parClustIdx is None or chdClustIdx is None:
            return Part.pairClustIdxs_pairPartRootNodeIds
        else:
            return Part.pairClustIdxs_pairPartRootNodeIds[(parClustIdx,
                                                            chdClustIdx)]

    def getParPart(self):
        return self._parPart

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

    def getRelTreeRoot(self):
        return self._relTreeRoot

    def getRelTypeIdx(self):
        return self._relTypeIdx

    def removeArgument(self, argIdx, clust_only=False):
        arg = self.getArgument(argIdx)

        oldArgClustIdx = self._argIdx_argClustIdx.pop(argIdx)
        self._argClustIdx_argIdxs[oldArgClustIdx].remove(argIdx)

        if len(self._argClustIdx_argIdxs[oldArgClustIdx]) == 0:
            self._argClustIdx_argIdxs.remove(oldArgClustIdx)

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

            if argClustIdx in self._argClustIdx_argIdxs:
                self._argClustIdx_argIdxs[argClustIdx].add(argIdx)
            else:
                self._argClustIdx_argIdxs[argClustIdx] = set(argIdx)

            arg = self.getArgument(argIdx)

            if not clust_only:
                cl = Clust.getClust(self.getClustIdx())

            if oldArgClustIdx < 0:
                if not clust_only:
                    cl.onPartSetArg(self, arg, argClustIdx)
            else:
                self._argClustIdx_argIdxs[oldArgClustIdx].remove(argIdx)

                if len(self._argClustIdx_argIdxs[oldArgClustIdx]) == 0:
                    self._argClustIdx_argIdxs.remove(oldArgClustIdx)

                if not clust_only:
                    cl.onPartSetArg(self, arg, argClustIdx, oldArgClustIdx)

        return None

    def setClust(self, clustIdx, clust_only=False):
        self._clustIdx = clustIdx
        rootID = self.getRelTreeRoot().getId()

        if clustIdx in Part.clustIdx_partRootNodeIds:
            Part.clustIdx_partRootNodeIds[clustIdx].add(rootID)
        else:
            Part.clustIdx_partRootNodeIds[clustIdx] = set(rootID)

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

        if parClustID in Part.clustIdx_pairClustIdxs:
            Part.clustIdx_pairClustIdxs[parClustID].add(pcci)
        else:
            Part.clustIdx_pairClustIdxs[parClustID] = set(pcci)

        pids = (parPart.getRelTreeRoot().getId(), self.getRelTreeRoot().getId())

        if pcci in Part.pairClustIdxs_pairPartRootNodeIds:
            Part.pairClustIdxs_pairPartRootNodeIds[pcci].add(pids)
        else:
            Part.pairClustIdxs_pairPartRootNodeIds[pcci] = set(pids)

        if parPart is not None:
            arg = parPart.getArgument(parArgIdx)
            dep = arg._path.getDep()

            if (parClustID != clustIdx) & dep.startswith('conj_'):
                if parClustID < clustIdx:
                    pci = pcci
                else:
                    pci = (pcci[1], pcci[0])

                if pci not in Clust._pairClustIdxs_conjCnt:
                    Clust.pairClustIdxs_conjCnt[pci] = 1
                else:
                    Clust.pairClustIdxs_conjCnt[pci] += 1

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
            self._argClustIdx_argIdxs.remove(oldArgClustIdx)

        if not clust_only:
            cl = Clust.getClust(self.getClustIdx())
            cl.onPartUnsetArg(self, arg, oldArgClustIdx)

        return None

    def unsetParent(self):
        '''
        Remove parent-child cluster index information
        Remove parent-child relationship index information
        NEEDS ADDITIONAL FACTORING - where does Cluster come from?
        '''
        parent = self.getParPart()
        clustIdx = self.getClustIdx()

        if parent is not None:
            parClustID = parent.getClustIdx()

            pcci = (parClustID, clustIdx)
            Part.clustIdx_pairClustIdxs[parClustID].remove(pcci)

            pids = (parent.getRelTreeRoot().getId(),
                    self.getRelTreeRoot().getId())
            Part.pairClustIdxs_pairPartRootNodeIds[pcci].remove(pids)

            arg = parent.getArgument(self.getParArgIdx())
            dep = arg._path.getDep()

            if (parClustID != clustIdx) & dep.startswith('conj_'):
                if parClustID < clustIdx:
                    pci = pcci
                else:
                    pci = (pcci[1], pcci[0])

                if pci in Clust._pairClustIdxs_conjCnt:
                    Clust.pairClustIdxs_conjCnt[pci] -= 1
                    if Clust.pairClustIdxs_conjCnt[pci] == 0:
                        del Clust.pairClustIdxs_conjCnt[pci]

        return None

    def unsetRelTypeIdx(self):
        old_type = self._relTypeIdx
        cl = Clust.getClust(self._clustIdx)
        cl.onPartUnsetRelTypeIdx(old_type)

        return None


#
# Clust
# 

class Clust(object):
    whereasClustIdx = -1
    nxtClustIdx = 1
    ttlRootCnt = 0

    # Dictionary mapping 
    pairClustIdx_conjCnt = {}
    # Dictionary mapping {int: {(int, int): int}}
    clustIdx_parArgs = {}
    # Dictionary mapping {int: int}
    clustIdx_rootCnt = {}
    # Dictionary mapping {str: int}
    argComb_cnt = {}
    # Dictionary mapping {int: set(str)}
    clustIdx_argCombs = {}
    # Dictionary mapping {int: Clust}
    clusts = {}
    # Dictionary mapping {int: set(int)}
    relTypeIdx_clustIdx = {}

    def __init__(self):
        self._isStop = False
        self._clustIdx = -1
        self._ttlCnt = 0
        self._nxtArgClustIdx = 0
        self._type = ''

        # Dictionary mapping {int: int}
        self._relTypeIdx_cnt = {}
        # Dictionary mapping {int: set(int)}
        self._argTypeIdx_argClustIdxs = {}
        # Dictionary mapping {int: ArgClust}
        self._argClusts = {}

    def incRootCnt(self):
        Clust.ttlRootCnt += 1

        if self.getId() not in Clust.clustIdx_rootCnt:
            Clust.clustIdx_rootCnt[self.getId()] = 1
        else:
            Clust.clustIdx_rootCnt[self.getId()] += 1

        return None

    def decRootCnt(self):
        Clust.ttlRootCnt -= 1

        Clust.clustIdx_rootCnt[self.getId()] -= 1

        if Clust.clustIdx_rootCnt[self.getId()] == 0:
            del Clust.clustIdx_rootCnt[self.getId()]

        return None

    def onPartUnsetRelTypeIdx(self, oldRelTypeIdx):
        self._relTypeIdx_cnt[oldRelTypeIdx] -= 1
        return None

    def onPartSetRelTypeIdx(self, newRelTypeIdx):
        if newRelTypeIdx not in self._relTypeIdx_cnt:
            self._relTypeIdx_cnt[newRelTypeIdx] = 1
        else:
            self._relTypeIdx_cnt[newRelTypeIdx] += 1

        return None

    def onPartSetClust(self, part):
        self._ttlCnt += 1
        ridx = part.getRelTypeIdx()
        self.onPartSetRelTypeIdx(ridx)

        return None

    def onPartUnsetClust(self, part):
        self._ttlCnt -= 1
        ridx = part.getRelTypeIdx()
        self.onPartUnsetRelTypeIdx(ridx)

        return None

    def createArgClust(self, argTypeIdx):
        assert argTypeIdx not in self._argTypeIdx_argClustIdxs
        argClustIdx = self._nxtArgClustIdx
        self._nxtArgClustIdx += 1
        ac = ArgClust()
        self._argClusts[argClustIdx] = ac
        acs = set()
        acs.add(argClustIdx)
        self._argTypeIdx_argClustIdxs[argTypeIdx] = acs

        return argClustIdx

    def getType(self):
        return self._type

    def isStop(self):
        return self._isStop


    def getClustsWithRelType(relTypeIdx):
        if relTypeIdx in Clust.relTypeIdx_clustIdx:
            return Clust.relTypeIdx_clustIdx[relTypeIdx]
        else:
            return None

    def createClust(relTypeIdx):
        cl = Clust()
        cl._clustIdx = Clust.nxtClustIdx
        Clust.nxtClustIdx += 1

        rt = RelType.getRelType(relTypeIdx)
        cl._type = rt.getType()
        rts = rt.toString()

        if rts in ['(V:be)', '(N:%)', '(V:say)', '($:$)']:
            cl._isStop = True

        if Clust.whereasClustIdx == -1 and rts == '(IN:whereas)':
            Clust.whereasClustIdx = cl._clustIdx

        Clust.clusts[cl._clustIdx] = cl
        if relTypeIdx in Clust.relTypeIdx_clustIdx:
            Clust.relTypeIdx_clustIdx[relTypeIdx].add(cl._clustIdx)
        else:
            Clust.relTypeIdx_clustIdx[relTypeIdx] = set(cl._clustIdx)

        return cl._clustIdx

    def removeClust(clust):
        del Clust.clusts[clust._clustIdx]
        return None

    def getClust(idx):
        return Clust.clusts[idx]

    def incRootCnt(self):
        Clust.ttlRootCnt += 1
        if self.getId() in Clust.clustIdx_rootCnt:
            Clust.clustIdx_rootCnt[self.getId()] += 1
        else:
            Clust.clustIdx_rootCnt[self.getId()] = 1

    def onPartSetClust(self, part):
        self._ttlCnt += 1
        ridx = part.getRelTypeIdx()
        if ridx in self._relTypeIdx_cnt:
            self._relTypeIdx_cnt[ridx] += 1
        else:
            self._relTypeIdx_cnt[ridx] = 1

        return None

    def onPartSetRelTypeIdx(self, newRelTypeIdx):
        if newRelTypeIdx in self._relTypeIdx_cnt:
            self._relTypeIdx_cnt[newRelTypeIdx] += 1
        else:
            self._relTypeIdx_cnt[newRelTypeIdx] = 1

        return None

    def removeArgClust(self, argClustIdx):
        del self._argClusts[argClustIdx]
        toDel = set()

        for ati in self._argTypeIdx_argClustIdxs:
            self._argTypeIdx_argClustIdxs[ati].remove(argClustIdx)

            if len(self._argTypeIdx_argClustIdxs[ati]) == 0:
                del self._argTypeIdx_argClustIdxs[ati]

        return None

    def addArgComb(clustIdx, chdClustIdxs, chdClustIdx2=None):
        if chdClustIdx2 is not None:
            chdClustIdxs = [chdClustIdxs, chdClustIdx2]

        ac = Clust.genArgCombStr(clustIdx, chdClustIdxs)

        if clustIdx not in Clust.clustIdx_argCombs:
            Clust.clustIdx_argCombs[clustIdx] = set()

        Clust.clustIdx_argCombs[clustIdx].add(ac)

        for idx in chdClustIdxs:
            if idx not in Clust.clustIdx_argCombs:
                Clust.clustIdx_argCombs[idx] = set()

            Clust.clustIdx_argCombs[idx].add(ac)

        if ac in Clust.argComb_cnt:
             Clust.argComb_cnt[ac] += 1
        else:
             Clust.argComb_cnt[ac]  = 1

        return None

    def genArgCombStr(clustIdx, clustIdxs):
        s = ':'.join([str(x) for x in [clustIdx] + clustIdxs])

        return s

    def getArgClustIdxs(self, argTypeIdx):
        if argTypeIdx in self._argTypeIdx_argClustIdxs:
            return self._argTypeIdx_argClustIdxs[argTypeIdx]
        else:
            return None

    def onPartSetArg(self, part, arg, argClustIdx, oldArgClustIdx=-1):
        argTypeIdx = arg._path.getArgType()
        chdClustIdx = arg._artPart.getClusterIdx()
        ac = self._argClusts[argClustIdx]

        if argTypeIdx in ac._argTypeIdx_cnt:
            ac._argTypeIdx_cnt[argTypeIdx] += 1
        else:
            ac._argTypeIdx_cnt[argTypeIdx]  = 1

        if chdClustIdx in ac._argTypeIdx_cnt:
            ac._argTypeIdx_cnt[chdClustIdx] += 1
        else:
            ac._argTypeIdx_cnt[chdClustIdx]  = 1

        ac._ttlArgCnt += 1

        if chdClustIdx not in Clust.clustIdx_parArgs:
            Clust.clustIdx_parArgs[chdClustIdx] = {}

        cl_ac = (self.getId(), argClustIdx)

        if cl_ac in Clust.clustIdx_parArgs[chdClustIdx]:
            Clust.clustIdx_parArgs[chdClustIdx][cl_ac] += 1
        else:
            Clust.clustIdx_parArgs[chdClustIdx][cl_ac]  = 1

        newArgNum = len(part._argClustIdx_argIdxs[argClustIdx])

        if newArgNum in ac._argNum_cnt:
            ac._argNum_cnt[newArgNum] += 1
        else:
            ac._argNum_cnt[newArgNum]  = 1

        if newArgNum > 1:
            if ac._argNum_cnt[newArgNum-1] == 1:
                del ac._argNum_cnt[newArgNum-1]
            else:
                ac._argNum_cnt[newArgNum-1] -= 1

        ac._partRootTreeNodeIds.add(part.getRelTreeRoot().getId())

        if oldArgClustIdx >= 0:
            self.onPartUnsetArg(part, arg, oldArgClustIdx)

        return None

    def getId(self):
        return self._clustIdx

    def onPartUnsetArg(self, part, arg, argClustIdx):
        argTypeIdx = arg.getPath().getArgType()
        chdClustIdx = arg.getPart().getClustIdx()
        ac = self._argClusts[argClustIdx]

        if ac._argTypeIdx_cnt[argTypeIdx] == 1:
            del ac._argTypeIdx_cnt[argTypeIdx]
        else:
            ac._argTypeIdx_cnt[argTypeIdx] -= 1

        if ac._chdClustIdx_cnt[chdClustIdx] == 1:
            del ac._chdClustIdx_cnt[chdClustIdx]
        else:
            ac._chdClustIdx_cnt[chdClustIdx] -= 1

        ac._ttlCnt -= 1
        cl_ac = (self.getId(), argClustIdx)

        if Clust.clustIdx_parArgs[chdClustIdx][cl_ac] == 1:
            del Clust.clustIdx_parArgs[chdClustIdx][cl_ac]
        else:
            Clust.clustIdx_parArgs[chdClustIdx][cl_ac] -= 1

        if len(Clust.clustIdx_parArgs[chdClustIdx]) == 0:
            del Clust.clustIdx_parArgs[chdClustIdx]

        ac._partRootTreeNodeIds.remove(part.getRelTreeRoot().getId())

        if ac._ttlArgCnt == 0:
            self.removeArgClust(argClustIdx)
            assert argClustIdx not in part._argClustIdx_argIdxs
        else:
            oldArgNum = 0

            if argClustIdx in part._argClustIdx_argIdxs:
                oldArgNum = part._argClustIdx_argIdxs[argClustIdx]

            if oldArgNum > 0:
                if oldArgNum in ac._argNum_cnt:
                    ac._argNum_cnt[oldArgNum] += 1
                else:
                    ac._argNum_cnt[oldArgNum] = 1

            if ac._argNum_cnt[oldArgNum+1] == 1:
                del ac._argNum_cnt[oldArgNum+1]
            else:
                ac._argNum_cnt[oldArgNum+1] -= 1

    def removePartAndUpdateStat(nid_part):
        for nid, p in nid_part.items():
            cl = Clust.getClust(p.getClustIdx())

            if p.getParPart() is None:
                cl.decRootCnt()

        for nid, p in nid_part.items():
            for ai, a in p._args.items():
                p.removeArgument(ai)
                cp = a._argPart
                cp.unsetParent()

            p.unsetRelType()

        for nid, p in nid_part.items():
            pclust = getClustIdx()
            Part.clustIdx_partRootNodeIds[pclust].remove(p.getRelTreeRoot().getId())

            if len(Part.clustIdx_partRootNodeIds[pclust]) == 0:
                del Part.clustIdx_partRootNodeIds[pclust]

        return None

    def updatePartStat(nid_part):
        for nid, p in nid_part.items():
            cl = Clust.getClust(p.getClustIdx())
            cl.onPartSetClust(p)

            if p.getParPart() is None:
                cl.incRootCnt()

            for ai, arg in p._args:
                aci = p._argTypeIdx_argClustIdxs[ai]
                cl.onPartSetArg(p, arg, aci)

        return None

    def toString(self):
        rts = ['{}:{}'.format(RelType.getRelType(rti).toString(), cnt) 
                for x, y in self._relTypeIdx_cnt.items()]
        s = ',\t'.join(rts)
        s = '[' + s + ']'

        return s


'''
    End Clust class definitions
'''

class ArgClust(object):
    def __init__(self):
        # Dictionary mapping {int: int}
        self._argTypeIdx_cnt = {}
        # Dictionary mapping {int: int}
        self._chdClustIdx_cnt = {}
        # Dictionary mapping {int: int}
        self._argNum_cnt = {}
        self._ttlArgCnt = 0
        self._partRootTreeNodeIds = set()

    def toString(self):
        s = ''
        for k, v in self._argTypeIdx_cnt.items():
            if len(s) > 0:
                s += ' '
            s += '{}:{}'.format(ArgType.getArgType(k), c)

        return s







