
from collections import OrderedDict

from semantic import Part, Clust, SearchOp, ParseParams

class Agenda(object):
    def __init__(self, parse):
        self._parse = parse
        self._skipMC = False
        self._skipCompose = False
        self._mc_neighs = dict()
        self._compose_cnt = dict()
        self._agendaToScore = set()
        self._clustIdx_agenda = dict()
        self._inactiveAgenda_score = dict()
        self._activeAgenda_score = dict()
        self._scoreActiveAgenda = OrderedDict()
        self._minAbsCntObserved =  ParseParams.minAbsCnt \
                                 * (ParseParams.minAbsCnt-1)/2

    def createAgenda(self):
        for clust_id in Part.getClustPartRootNodeIds():
            clust = Clust.getClust(clust_id)
            if clust.getType() == 'C' and not clust.isStop():
                self.addAgendaForNewClust(clust_id)
                
        return None

    def procAgenda(self):
        ttlAgendaScored, ttlExecMC, ttlExecAbs = (0, 0, 0)

        while True:
            As = 0

            for op in self._agendaToScore:
                score = self._parse._scorer.scoreOp(op)
                As += 1

                if score <= 200:
                    continue

                self.addAgenda(op, score)

            self._agendaToScore.clear()
            ttlAgendaScored = As + 1

            if len(self._scoreActiveAgenda) == 0:
                break

            score, op = next(reversed(self._scoreActiveAgenda))
            newClustIdx = self._parse._executor.executeOp(op)
            self.updateAgendaAfterExec(op, newClustIdx)

            if op._op == SearchOp.OP_COMPOSE:
                ttlExecAbs += 1
            elif op._op == SearchOp.OP_MERGE_CLUST:
                ttlExecMC += 1

        return None

    def addAgendaAfterMergeClust(self, part_1, part_2):
        assert part_1._clustIdx == part_2._clustIdx

        clustIdx = part_1._clustIdx

        if part_1.getParPart() is not None and part_2.getParPart() is not None:
            clustIdx1 = part_1.getParPart()._clustIdx
            clustIdx2 = part_2.getParPart()._clustIdx

            if clustIdx1 != clustIdx2:
                self.addAgendaMC(clustIdx1, clustIdx2, 2*clustIdx+1)
            else:
                self.addAgendaAbs(clustIdx1, clustIdx)

        chs1, chs2 = [x.getArguments() for x in (part_1, part_2)]

        for c1 in chs1.values():
            clustIdx1 = c1._argPart._clustIdx

            for c2 in chs2.values():
                clustIdx2 = c2._argPart._clustIdx

                if clustIdx1 != clustIdx2:
                    self.addAgendaMC(clustIdx1, clustIdx2, 2*clustIdx+1)
                else:
                    self.addAgendaAbs(clustIdx, clustIdx1)

        return None

    def moveAgendaToScore(self, op):
        assert op in self._activeAgenda_score or op in self._inactiveAgenda_score

        if op in self._agendaToScore:
            return True

        if op in self._activeAgenda_score:
            score = self._activeAgenda_score[op]
            del self._scoreActiveAgenda[op]
            del self._activeAgenda_score[op]
            self._agendaToScore.add(op)

            return True
        else:
            del self._inactiveAgenda_score[op]
            self._agendaToScore.add(op)

            return True

        return False

    def addAgendaMC(self, clustIdx1, clustIdx2, neighType):
        if not (self._skipMC or clustIdx1 == clustIdx2):
            type1 = Clust.getClust(clustIdx1).getType()
            type2 = Clust.getClust(clustIdx2).getType()

            if type1 == type2 and type1 == 'C':
                op = SearchOp()
                op._clustIdx1 = min((c1, c2))
                op._clustIdx2 = max((c1, c2))

                if not self.moveAgendaToScore(op):
                    if op not in self._mc_neighs:
                        self._mc_neighs[op] = set()

                    if len(self._mc_neighs[op])+1 >= ParseParams.minMCCnt:
                        self._agendaToScore.add(op)
                        del self._mc_neighs[op]
                    else:
                        self._mc_neighs[op].add(neighType)

        return None

    def addAgendaAbs(self, parClustIdx, chdClustIdx):
        if not self._skipCompose:
            op = SearchOp()
            op._op = SearchOp.OP_COMPOSE
            op._parClustIdx = parClustIdx
            op._chdClustIdx = chdClustIdx

            if not self.moveAgendaToScore(op):
                if op not in self._compose_cnt:
                    self._compose_cnt[op] = 1
                elif self._compose_cnt[op]+1 >= self._minAbsCntObserved:
                    del self._compose_cnt[op]
                    self._agendaToScore.add(op)
                else:
                    self._compose_cnt[op] = self._compose_cnt[op] + 1

        return None

    def updateAgendaAfterExec(self, op, newClustIdx):
        self.removeAgenda(op)

        if newClustIdx >= 0:
            if op._op == SearchOp.OP_MERGE_CLUST:
                self.updateAgendaAfterExecMC(op, newClustIdx)
            elif op._op == SearchOp.OP_COMPOSE:
                self.updateAgendaAfterExecAbs(op, newClustIdx)

        return None

    def addAgendaToScore(self, op):
        self._agendaToScore.add(op)
        return None

    def updateAgendaAfterExecMC(self, op, newClustIdx):
        assert op._op == SearchOp.OP_MERGE_CLUST

        oldClustIdx = op._clustIdx2

        if oldClustIdx == newClustIdx:
            oldClustIdx = op._clustIdx1

        while len(self._clustIdx_agenda[oldClustIdx]) > 0:
            oop = self._clustIdx_agenda[oldClustIdx].next()
            self.removeAgenda(oop)

            if oop._op == SearchOp.OP_MERGE_CLUST:
                ci1 = oop._clustIdx1
                ci2 = oop._clustIdx2

                if ci1 == oldClustIdx:
                    ci1 = newClustIdx

                if ci2 == oldClustIdx:
                    ci2 = newClustIdx

                if ci1 != ci2:
                    nop = oop
                    nop._clustIdx1 = min((c1, c2))
                    nop._clustIdx2 = max((c1, c2))
                    nop.genString()
                    self.addAgendaToScore(nop)
            elif oop._op == SearchOp.OP_COMPOSE:
                ci1 = oop._clustIdx1
                ci2 = oop._clustIdx2

                if ci1 == oldClustIdx:
                    ci1 = newClustIdx

                if ci2 == oldClustIdx:
                    ci2 = newClustIdx

                nop = oop
                nop._parClustIdx = ci1
                nop._chdClustIdx = ci2
                nop.genString()
                self.addAgendaToScore(nop)

        del self._clustIdx_agenda[oldClustIdx]

        for prnid in Part.getClustPartRootNodeIds()[newClustIdx]:
            p = Part.getPartByRootNodeId(prnid)

            for prnid2 in Part.getClustPartRootNodeIds()[oldClustIdx]:
                p2 = Part.getPartByRootNodeId(prnid2)
                self.addAgendaAfterMergeClust(p, p2)

        return None

    def updateAgendaAfterExecAbs(self, op, newClustIdx, oop=None):
        if op._op == SearchOp.OP_COMPOSE:
            parClustIdx = op._parClustIdx
            chdClustIdx = op._chdClustIdx

            self._clustIdx_agenda[parClustIdx].remove(op)
            self._clustIdx_agenda[chdClustIdx].remove(op)

            while len(self._clustIdx_agenda[parClustIdx]) > 0:
                oop = self._clustIdx_agenda[parClustIdx].next()
                self.removeAgenda(oop)
                oop.genString()
                self.addAgendaToScore(oop)

            while len(self._clustIdx_agenda[chdClustIdx]) > 0:
                oop = self._clustIdx_agenda[chdClustIdx].next()
                self.removeAgenda(oop)
                oop.genString()
                self.addAgendaToScore(oop)

            self.addAgendaForNewClust(newClustIdx)
        elif oop is not None:
            ci1, ci2 = (-1, -1)

            if oop._op == SearchOp.OP_MERGE_CLUST:
                ci1 = oop._clustIdx1
                ci2 = oop._clustIdx2
            elif oop._op == SearchOp.OP_COMPOSE:
                ci1 = oop._parClustIdx
                ci2 = oop._chdClustIdx

            if c1 in (op._parClustIdx, op._chdClustIdx):
                c1 = newClustIdx

            if c2 in (op._parClustIdx, op._chdClustIdx):
                c2 = newClustIdx

            if oop._op == SearchOp.OP_MERGE_CLUST:
                if c1 != c2:
                    nop = SearchOp()
                    nop._clustIdx1 = min((c1, c2))
                    nop._clustIdx2 = max((c1, c2))
                    nop._op = oop._op
                    self.addAgendaToScore(nop)
            elif oop._op == SearchOp.OP_COMPOSE:
                nop = SearchOp()
                nop._parClustIdx = c1
                nop._chdClustIdx = c2
                nop._op = oop._op
                self.addAgendaToScore(nop)

        return None

    def addAgendaForNewClust(self, newClustIdx):
        part_node_ids = Part.getClustPartRootNodeIds()[newClustIdx]

        if len(part_node_ids) > 1:
            for node_id in part_node_ids:
                part_1 = Part.getPartByRootNodeId(node_id)

                for node_id2 in part_node_ids:
                    if Utils.compareStr(node_id, node_id2) <= 0:
                        break
                    part_2 = Part.getPartByRootNodeId(node_id2)
                    self.addAgendaAfterMergeClust(part_1, part_2)

        return None

    def removeAgenda(self, op):
        assert (op in self._activeAgenda_score or op in self._inactiveAgenda_score)

        if op in self._activeAgenda_score:
            score = self._activeAgenda_score[op]
            del self._scoreActiveAgenda[(score, op)]
            del self._activeAgenda_score[op]
        elif op in self._inactiveAgenda_score:
            del self._inactiveAgenda_score[op]

        if op._op == SearchOp.OP_MERGE_CLUST:
            self._clustIdx_agenda[op._clustIdx1].remove(op)
            self._clustIdx_agenda[op._clustIdx2].remove(op)
        elif op._op == SearchOp.OP_COMPOSE:
            self._clustIdx_agenda[op._parClustIdx].remove(op)
            self._clustIdx_agenda[op._chdClustIdx].remove(op)           

        return None

    def addAgenda(self, op, score):
        ci1, ci2 = (-1, -1)

        if op._op == SearchOp.OP_MERGE_CLUST:
            ci1 = op._clustIdx1
            ci2 = op._clustIdx2
        elif op._op == SearchOp.OP_COMPOSE:
            ci1 = op._parClustIdx
            ci2 = op._chdClustIdx

        if ci1 not in self._clustIdx_agenda:
            self._clustIdx_agenda[ci1] = set()

        self._clustIdx_agenda[ci1].add(op)

        if ci2 not in self._clustIdx_agenda:
            self._clustIdx_agenda[ci2] = set()

        self._clustIdx_agenda[ci2].add(op)

        if score < ParseParams.priorCutOff:
            self._inactiveAgenda_score[op] = score
        else:
            self._activeAgenda_score[op] = score
            self._scoreActiveAgenda[score] = op

        return None




















