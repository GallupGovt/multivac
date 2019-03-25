
#from collections import OrderedDict
from sortedcontainers import SortedSet
from datetime import datetime
import math
import pickle
from semantic import Part, Clust, SearchOp, ParseParams
from utils import Utils

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
        self._scoreActiveAgenda = SortedSet() # (float, SearchOp)
        self._minAbsCntObserved =  ParseParams.minAbsCnt \
                                 * (ParseParams.minAbsCnt-1)/2
        # self.logc = open("/Users/ben_ryan/Documents/DARPA ASKE/usp-code/genia_full/create_agenda.log", "a+")
        # self.logp = open("/Users/ben_ryan/Documents/DARPA ASKE/usp-code/genia_full/proc_agenda.log", "a+")

    def save_agenda(self, path):
        '''
            Save all objects necessary to recreate the current state of Agenda
        '''
        with open(path, 'wb') as f:
            pickle.dump({'saved_agenda': self}, f)

        return None

    def load_agenda(path, prs):
        '''
            Given a Parse object, load the saved state of an Agenda and 
            attach it, returning the updated Parse object.
        '''
        with open(path, 'rb') as f:
            sav = pickle.load(f)

        prs.agenda = sav['saved_agenda']
        prs.agenda._parse = prs

        return prs

    def createAgenda(self, verbose=False):
        if verbose:
            clust_cnt = len(Part.getClustPartRootNodeIds())
            milestones = set([x for x in range(1, 10, 1)])
            i = 0

        for clust_id in Part.getClustPartRootNodeIds():
            clust = Clust.getClust(clust_id)

            if clust.getType() != 'C':
                continue
            elif clust.isStop():
                continue

            # # self.logc.write("Adding to agenda for cluster {}\n".format(clust_id))
            self.addAgendaForNewClust(clust_id, verbose)

            if verbose:
                i += 1
                done = math.floor(i*10/clust_cnt)
                
                if done in milestones:
                    milestones.remove(done)
                    print("{}% complete.".format(done*10))

        # self.logc.close()
                
        return None

    def addAgendaForNewClust(self, newClustIdx, verbose=False):
        part_node_ids = Part.getClustPartRootNodeIds()[newClustIdx]
        num_parts = len(part_node_ids)

        # if verbose:
        #     print("Updating agenda: {} possible operations.".format(num_parts*(num_parts-1)))

        if len(part_node_ids) > 1:
            for node_id in part_node_ids:
                part_1 = Part.getPartByRootNodeId(node_id)

                for node_id2 in part_node_ids:
                    if node_id <= node_id2:
                        break
                    part_2 = Part.getPartByRootNodeId(node_id2)

                    # self.logc.write("\tAdding parts {} and {} to agenda for cluster {}\n".format(node_id, node_id2, newClustIdx))
                    self.addAgendaAfterMergeClust(part_1, part_2)

        return None

    def addAgendaAfterMergeClust(self, part_1, part_2):
        # First, check that these parts belong to the same cluster
        assert part_1._clustIdx == part_2._clustIdx

        clustIdx = part_1._clustIdx

        # If they have parents, check whether the parents are in the same cluster
        # If not, look at merging their clusters, and if so, look at composing
        # the clusters for part_1 and its parent.

        if part_1.getParPart() is not None and part_2.getParPart() is not None:
            clustIdx1 = part_1.getParPart()._clustIdx
            clustIdx2 = part_2.getParPart()._clustIdx

            if clustIdx1 != clustIdx2:
                self.addAgendaMC(clustIdx1, clustIdx2, 2*clustIdx+1)
            else:
                self.addAgendaAbs(clustIdx1, clustIdx)

        # Next, get the arguments (children) of each part
        # Compare each argument in A) with each argument in B) - if they have
        # different clusters, look at merging them, and if they have the same
        # look at composing the clusters for part_1 and its argument(s).

        kids_1 = part_1.getArguments()
        kids_2 = part_2.getArguments()
        # # self.logc.write("\tAdding to agenda for kids of {} and {} in {}\n".format(part_1.getRelTreeRoot().getId(), 
        #                                                                           part_2.getRelTreeRoot().getId(), 
        #                                                                           clustIdx))

        for kid1 in kids_1.values():
            clustIdx1 = kid1._argPart._clustIdx

            for kid2 in kids_2.values():
                clustIdx2 = kid2._argPart._clustIdx

                if clustIdx1 != clustIdx2:
                    #print("Add agenda - Merge Clusters {} and {}".format(clustIdx1, clustIdx2))
                    self.addAgendaMC(clustIdx1, clustIdx2, 2*clustIdx+1)
                else:
                    #print("Add agenda - Compose Clusters {} and {}".format(clustIdx, clustIdx1))
                    self.addAgendaAbs(clustIdx, clustIdx1)

        return None

    def addAgendaMC(self, clustIdx1, clustIdx2, neighType):
        if not (self._skipMC or clustIdx1 == clustIdx2):
            type1 = Clust.getClust(clustIdx1).getType()
            type2 = Clust.getClust(clustIdx2).getType()

            if type2 == 'C' and type1 == 'C':
                op = SearchOp()
                op._op = SearchOp.OP_MERGE_CLUST
                op._clustIdx1 = min((clustIdx1, clustIdx2))
                op._clustIdx2 = max((clustIdx1, clustIdx2))

                if not self.moveAgendaToScore(op):
                    if op not in self._mc_neighs:
                        self._mc_neighs[op] = set()

                    if len(self._mc_neighs[op])+1 >= ParseParams.minMCCnt:
                        self._agendaToScore.add(op)
                        del self._mc_neighs[op]
                    else:
                        self._mc_neighs[op].add(neighType)

                    ## self.logc.write("\t\tMerge Op: {}; mc_neighs: {}, agendaToScore: {}\n".format(op, len(self._mc_neighs), len(self._agendaToScore)))

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

                if self._compose_cnt[op]+1 >= self._minAbsCntObserved:
                    self._agendaToScore.add(op)
                    del self._compose_cnt[op]
                else:
                    self._compose_cnt[op] += 1
                
                ## self.logc.write("\t\tCompose Op: {}; compose_cnt: {}, agendaToScore: {}\n".format(op, len(self._compose_cnt), len(self._agendaToScore)))

        return None

    def moveAgendaToScore(self, op):
        #assert op in self._activeAgenda_score or op in self._inactiveAgenda_score

        if op in self._agendaToScore:
            return True

        if op in self._activeAgenda_score:
            score = self._activeAgenda_score[op]
            self._scoreActiveAgenda.discard((score, op))
            del self._activeAgenda_score[op]
            self._agendaToScore.add(op)

            return True
        elif op in self._inactiveAgenda_score:
            del self._inactiveAgenda_score[op]
            self._agendaToScore.add(op)

            return True

        return False

    def procAgenda(self, verbose=False):
        if verbose:
            print("Processing agenda with {} operations in queue.".format(len(self._agendaToScore)))
        ttlAgendaScored, ttlExecMC, ttlExecAbs = (0, 0, 0)
        i = 1

        while True:
            As = 0

            for op in self._agendaToScore:
                score = self._parse.scorer.scoreOp(op)
                if verbose:
                    print("<SCORE> {} score={}".format(op, score))
                As += 1

                if score < -200:
                    continue

                if verbose:
                    print("<Add Agenda> {} score={}".format(op, score))
                self.addAgenda(op, score)

            self._agendaToScore.clear()
            ttlAgendaScored = As + 1

            if len(self._scoreActiveAgenda) == 0:
                break

            score, op = next(reversed(self._scoreActiveAgenda))
            if verbose:
                print("Executing: {}, score={}".format(op, score))
            newClustIdx = self._parse.executor.executeOp(op)
            self.updateAgendaAfterExec(op, newClustIdx, verbose)

            if op._op == SearchOp.OP_COMPOSE:
                ttlExecAbs += 1
            elif op._op == SearchOp.OP_MERGE_CLUST:
                ttlExecMC += 1

            if verbose:
                print("Total op_compose: {}, Total op_merge_clust: {}".format(ttlExecAbs, ttlExecMC))
            i += 1

            if verbose and i%10==0:
                print("{} Processing agenda: {} loops".format(datetime.now(), i))

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
            self._scoreActiveAgenda.add((score, op))

        return None

    def updateAgendaAfterExec(self, op, newClustIdx, verbose=False):
        self.removeAgenda(op)

        if newClustIdx >= 0:
            if op._op == SearchOp.OP_MERGE_CLUST:
                self.updateAgendaAfterExecMC(op, newClustIdx, verbose)
            elif op._op == SearchOp.OP_COMPOSE:
                self.updateAgendaAfterExecAbs(op, newClustIdx, verbose=verbose)

        return None

    def addAgendaToScore(self, op):
        self._agendaToScore.add(op)
        return None

    def updateAgendaAfterExecMC(self, op, newClustIdx, verbose=False):
        assert op._op == SearchOp.OP_MERGE_CLUST

        oldClustIdx = op._clustIdx2

        if oldClustIdx == newClustIdx:
            oldClustIdx = op._clustIdx1

        while len(self._clustIdx_agenda[oldClustIdx]) > 0:
            oop = next(iter(self._clustIdx_agenda[oldClustIdx]))
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
                    nop._clustIdx1 = min((ci1, ci2))
                    nop._clustIdx2 = max((ci1, ci2))
                    nop.genString()
                    self.addAgendaToScore(nop)
            elif oop._op == SearchOp.OP_COMPOSE:
                ci1 = oop._parClustIdx
                ci2 = oop._chdClustIdx

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

        num_parts_old = len(Part.getClustPartRootNodeIds()[oldClustIdx])
        num_parts_new = len(Part.getClustPartRootNodeIds()[newClustIdx])

        if verbose:
            print("Updating agenda: {} possible operations.".format(num_parts_new*(num_parts_old)))

        for prnid in Part.getClustPartRootNodeIds()[newClustIdx]:
            p = Part.getPartByRootNodeId(prnid)

            for prnid2 in Part.getClustPartRootNodeIds()[oldClustIdx]:
                p2 = Part.getPartByRootNodeId(prnid2)
                self.addAgendaAfterMergeClust(p, p2)

        return None

    def updateAgendaAfterExecAbs(self, op, newClustIdx, oop=None, verbose=False):
        if op._op == SearchOp.OP_COMPOSE:
            parClustIdx = op._parClustIdx
            chdClustIdx = op._chdClustIdx

            while len(self._clustIdx_agenda[parClustIdx]) > 0:
                oop = next(iter(self._clustIdx_agenda[parClustIdx]))
                self.removeAgenda(oop)
                # oop.genString()
                self.addAgendaToScore(oop)

            while len(self._clustIdx_agenda[chdClustIdx]) > 0:
                oop = next(iter(self._clustIdx_agenda[chdClustIdx]))
                self.removeAgenda(oop)
                # oop.genString()
                self.addAgendaToScore(oop)

            self.addAgendaForNewClust(newClustIdx, verbose)
        # elif oop is not None:
        #     ci1, ci2 = (-1, -1)

        #     if oop._op == SearchOp.OP_MERGE_CLUST:
        #         ci1 = oop._clustIdx1
        #         ci2 = oop._clustIdx2
        #     elif oop._op == SearchOp.OP_COMPOSE:
        #         ci1 = oop._parClustIdx
        #         ci2 = oop._chdClustIdx

        #     if ci1 in (op._parClustIdx, op._chdClustIdx):
        #         ci1 = newClustIdx

        #     if ci2 in (op._parClustIdx, op._chdClustIdx):
        #         ci2 = newClustIdx

        #     if oop._op == SearchOp.OP_MERGE_CLUST:
        #         if ci1 != ci2:
        #             nop = SearchOp()
        #             nop._clustIdx1 = min((ci1, ci2))
        #             nop._clustIdx2 = max((ci1, ci2))
        #             nop._op = oop._op
        #             self.addAgendaToScore(nop)
        #     elif oop._op == SearchOp.OP_COMPOSE:
        #         nop = SearchOp()
        #         nop._parClustIdx = ci1
        #         nop._chdClustIdx = ci2
        #         nop._op = oop._op
        #         self.addAgendaToScore(nop)

        return None

    def removeAgenda(self, op):
        # assert (op in self._activeAgenda_score or op in self._inactiveAgenda_score)

        if op in self._activeAgenda_score:
            score = self._activeAgenda_score[op]
            self._scoreActiveAgenda.discard((score, op))
            del self._activeAgenda_score[op]
        elif op in self._inactiveAgenda_score:
            del self._inactiveAgenda_score[op]

        if op._op == SearchOp.OP_MERGE_CLUST:
            self._clustIdx_agenda[op._clustIdx1].discard(op)
            self._clustIdx_agenda[op._clustIdx2].discard(op)
        elif op._op == SearchOp.OP_COMPOSE:
            self._clustIdx_agenda[op._parClustIdx].discard(op)
            self._clustIdx_agenda[op._chdClustIdx].discard(op)           

        return None
















