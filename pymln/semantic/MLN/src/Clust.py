

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
        self._isDebug = False
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







