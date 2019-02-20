
from semantic import SearchOp, Clust, ParseParams
from syntax import RelType
from utils.Utils import inc_key, dec_key, xlogx, java_iter
from math import log

class Scorer(object):
    def __init__(self):
        return None

    def scoreOp(self, op):
        if op._op == SearchOp.OP_MERGE_CLUST:
            return self.scoreOpMC(op)
        elif op._op == SearchOp.OP_COMPOSE:
            return self.scoreOpCompose(op._parClustIdx,op._chdClustIdx)
        else:
            return -100

    def scoreOpMC(self, op):
        cidx1, cidx2 = op._clustIdx1, op._clustIdx2
        assert cidx1<cidx2

        score = 0 - ParseParams.priorMerge

        if (cidx1, cidx2) in Clust.pairClustIdxs_conjCnt:
            score -= ParseParams.priorNumConj * Clust.pairClustIdxs_conjCnt[(cidx1, cidx2)]

        cl1 = Clust.getClust(cidx1)
        cl2 = Clust.getClust(cidx2)

        score -= Scorer.updateScore(cl1._ttlCnt, cl2._ttlCnt)

        for ri, cnt in cl1._relTypeIdx_cnt.items():
            if ri in cl2._relTypeIdx_cnt:
                cnt2 = cl2._relTypeIdx_cnt[ri]
                score += Scorer.updateScore(cnt, cnt2)
                score += ParseParams.priorNumParam

        if cidx1 in Clust.clustIdx_rootCnt and cidx2 in Clust.clustIdx_rootCnt:
            rc1 = Clust.clustIdx_rootCnt[cidx1]
            rc2 = Clust.clustIdx_rootCnt[cidx2]
            score = Scorer.updateScore(rc1, rc2)
            score += ParseParams.priorNumParam

        score += self.scoreMCForParent(cidx1, cidx2)

        if len(cl2._argClusts) > len(cl1._argClusts):
            clx1 = cl2
            clx2 = cl1
        else:
            clx1 = cl1
            clx2 = cl2

        score += self.scoreMCForAlign(clx1, clx2, dict())

        return score


    def scoreMCForParent(self, clustIdx1, clustIdx2):
        score = 0

        if clustIdx1 in Clust.clustIdx_parArgs and clustIdx2 in Clust.clustIdx_parArgs:
            parents1 = Clust.clustIdx_parArgs[clustIdx1]
            parents2 = Clust.clustIdx_parArgs[clustIdx2]

            for par_arg in parents1:
                if par_arg in parents2:
                    par_clust_id, arg_clust_id = par_arg
                    pcl = Clust.getClust(par_clust_id)
                    ac = pcl._argClusts[arg_clust_id]
                    c1 = ac._chdClustIdx_cnt[clustIdx1]
                    c2 = ac._chdClustIdx_cnt[clustIdx2]
                    score += ParseParams.priorNumParam
                    score += Scorer.updateScore(c1, c2)

        return score

    def scoreOpCompose(self, rcidx, acidx):

        def update_score_from_dict(score, d, orig_d):
            for key, cnt in d.items():
                origcnt = orig_d[key]
                assert origcnt >= cnt
                score -= xlogx(origcnt)

                if cnt > 0:
                    score += xlogx(cnt)
                else:
                    score += ParseParams.priorNumParam

            return score

        parChdNids = Part.getPairPartRootNodeIds(rcidx, acidx)

        if parChdNids is None:
            return -10000

        score = 0
        rcl = Clust.getClust(rcidx)
        acl = Clust.getClust(acidx)

        rtc_new, atc_new = [c._ttlCnt for c in [rcl, acl]]
        ratc_new = 0
        raRootCnt = 0

        parArg_cnt = dict()

        rRelTypeIdx_newcnt = dict()
        aRelTypeIdx_newcnt = dict()
        raRelTypeIdx_newcnt = dict()

        rArgClustIdx_argNum_cnt = dict()
        aArgClustIdx_argNum_cnt = dict()
        rNewArgClustIdx_argNum_cnt = dict()
        aNewArgClustIdx_argNum_cnt = dict()

        rArgClustIdx_argTypeIdx_cnt = dict()
        aArgClustIdx_argTypeIdx_cnt = dict()
        rNewArgClustIdx_argTypeIdx_cnt = dict()
        aNewArgClustIdx_argTypeIdx_cnt = dict()

        rArgClustIdx_chdClustIdx_cnt = dict()
        aArgClustIdx_chdClustIdx_cnt = dict()
        rNewArgClustIdx_chdClustIdx_cnt = dict()
        aNewArgClustIdx_chdClustIdx_cnt = dict()

        rArgClustIdx_partCnt = dict()
        aArgClustIdx_partCnt = dict()
        rNewArgClustIdx_partCnt = dict()
        aNewArgClustIdx_partCnt = dict()

        rArgClustIdx_argCnt = dict()
        aArgClustIdx_argCnt = dict()
        rNewArgClustIdx_argCnt = dict()
        aNewArgClustIdx_argCnt = dict()

        for pcnid in parChdNids:
            pp, cp = [Part.getPartByRootNodeId(i) for i in pcnid]

            rtc_new -= 1
            atc_new -= 1
            ratc_new += 1

            rrt = pp.getRelTypeIdx()
            art = cp.getRelTypeIdx()
            raArgClustidx = pp.getArgClust(cp._parArgIdx)

            rRelTypeIdx_newcnt = dec_key(rRelTypeIdx_newcnt,
                                         rrt, 
                                         base=rcl._relTypeIdx_cnt[rrt])

            rRelTypeIdx_newcnt = dec_key(rRelTypeIdx_newcnt,
                                         art, 
                                         base=acl._relTypeIdx_cnt[art])

            raRelTypeIdx_newcnt = inc_key(raRelTypeIdx_newcnt, (rrt, art))

            pp_par = pp.getParPart()

            if pp_par is not None:
                ai = pp.getParArgIdx()
                ppi = pp_par.getClustIdx()
                aci = pp_par.getArgClust(ai)

                parArg_cnt = inc_key(parArg_cnt, (ppi, aci))
            else:
                raRootCnt += 1

            for arg_ci in pp._argClustIdx_argIdxs:
                an = len(pp._argClustIdx_argIdxs[arg_ci])
                ac = rcl._argClusts[arg_ci]

                rArgClustIdx_partCnt = dec_key(rArgClustIdx_partCnt,
                                               arg_ci, 
                                               base=len(ac.partRootTreeNodeIds))
                
                rArgClustIdx_argNum_cnt[arg_ci] = \
                    dec_key(rArgClustIdx_argNum_cnt[arg_ci],
                            an, 
                            base=ac._argNum_cnt[an])

                newArgNum = an + 0

                if arg_ci == raArgClustidx:
                    newArgNum -= 1

                if newArgNum == 0:
                    continue

                rNewArgClustIdx_argNum_cnt[arg_ci] = \
                    inc_key(rNewArgClustIdx_argNum_cnt[arg_ci], newArgNum)

                rNewArgClustIdx_partCnt = inc_key(rNewArgClustIdx_partCnt,
                                                   arg_ci)

            for arg_ci in cp._argClustIdx_argIdxs:
                an = len(cp._argClustIdx_argIdxs[arg_ci])
                ac = acl._argClusts[arg_ci]

                aArgClustIdx_partCnt = dec_key(aArgClustIdx_partCnt,
                                               arg_ci, 
                                               base=len(ac.partRootTreeNodeIds))

                aArgClustIdx_argNum_cnt[arg_ci] = \
                    dec_key(aArgClustIdx_argNum_cnt[arg_ci],
                            an, 
                            base=ac._argNum_cnt[an])

                aNewArgClustIdx_argNum_cnt[arg_ci] = \
                    inc_key(aNewArgClustIdx_argNum_cnt[arg_ci], an)

                aNewArgClustIdx_partCnt = inc_key(aNewArgClustIdx_partCnt, arg_ci)

            args = pp.getArguments()

            for ai, arg in args.items():
                ap = arg._argPart
                cci = ap._clustIdx
                aci = pp.getArgClust(ai)
                ac = rcl._argClusts[aci]
                ati = arg._path.getArgType()

                # Drop the old arguments

                rArgClustIdx_argCnt = dec_key(rArgClustIdx_argCnt, 
                                              aci, 
                                              base=ac._ttlArgCnt)

                rArgClustIdx_argTypeIdx_cnt[aci] = \
                    dec_key(rArgClustIdx_argTypeIdx_cnt[aci], 
                            ati, 
                            base=ac._argTypeIdx_cnt[ati])

                rArgClustIdx_chdClustIdx_cnt[aci] = \
                    dec_key(rArgClustIdx_chdClustIdx_cnt[aci], 
                            cci, 
                            base=ac.chdClustIdx_cnt[cci])

                # Add the new arguments

                if ap.getRelTreeRoot().getId() != cp.getRelTreeRoot().getId():
                    rNewArgClustIdx_argCnt = inc_key(rNewArgClustIdx_argCnt, aci)

                    rNewArgClustIdx_argTypeIdx_cnt[aci] = \
                        inc_key(rNewArgClustIdx_argTypeIdx_cnt, ati)

                    rNewArgClustIdx_chdClustIdx_cnt[aci] = \
                        inc_key(rNewArgClustIdx_chdClustIdx_cnt, cci)

            args = cp.getArguments()

            for ai, arg in args.items():
                ap = arg._argPart
                cci = ap._clustIdx
                aci = pp.getArgClust(ai)
                ac = rcl._argClusts[aci]
                ati = arg._path.getArgType()

                # Drop the old arguments

                aArgClustIdx_argCnt = dec_key(aArgClustIdx_argCnt, 
                                              aci, 
                                              base=ac._ttlArgCnt)

                aArgClustIdx_argTypeIdx_cnt[aci] = \
                    dec_key(aArgClustIdx_argTypeIdx_cnt[aci], 
                            ati, 
                            base=ac._argTypeIdx_cnt[ati])

                if aci not in aArgClustIdx_chdClustIdx_cnt:
                    aArgClustIdx_chdClustIdx_cnt[aci] = dict()

                aArgClustIdx_chdClustIdx_cnt[aci] = \
                    dec_key(aArgClustIdx_chdClustIdx_cnt[aci], 
                            cci, 
                            base=ac.chdClustIdx_cnt[cci])

                # Add the new arguments

                aNewArgClustIdx_argCnt = inc_key(aNewArgClustIdx_argCnt, aci)

                aNewArgClustIdx_argTypeIdx_cnt[aci] = \
                    inc_key(aNewArgClustIdx_argTypeIdx_cnt, ati)

                aNewArgClustIdx_chdClustIdx_cnt[aci] = \
                    inc_key(aNewArgClustIdx_chdClustIdx_cnt, cci)

        if raRootCnt > 0:
            origRootCnt = Clust.clustIdx_rootCnt[rcidx]

            if origRootCnt > raRootCnt:
                score =   xlogx(raRootCnt) \
                        + xlogx(origRootCnt - raRootCnt) \
                        - xlogx(origRootCnt)
                score -= ParseParams.priorNumParam

        denomor = xlogx(rcl._ttlCnt)
        denomnr = xlogx(rtc_new)

        score = update_score_from_dict(score, 
                                       rRelTypeIdx_newcnt, 
                                       rcl._relTypeIdx_cnt)

        socre += denomor
        score -= denomnr

        denomoa = xlogx(acl._ttlCnt)
        denomoa = xlogx(atc_new)

        score = update_score_from_dict(score, 
                                       aRelTypeIdx_newcnt, 
                                       acl._relTypeIdx_cnt)

        socre += denomoa
        score -= denomna

        for cnt in raRelTypeIdx_newcnt.values():
            score -= ParseParams.priorNumParam
            score += xlogx(cnt)

        score -= xlogx(ratc_new)

        for pi, cnt in parArg_cnt:
            pc = Clust.getClust(pi[0])
            ac = pc._argClusts[pi[1]]
            origcnt = ac._chdClustIdx_cnt[rcidx]

            if cnt == origcnt:
                continue

            score -= ParseParams.priorNumParam
            score += xlogx(cnt) + xlogx(origcnt-cnt) - xlogx(origcnt)

        for aci, ac in rcl._argClusts:
            origPartCnt = len(ac._partRootTreeNodeIds)
            score -= xlogx(rcl._ttlCnt - origPartCnt) - denomor

            if aci not in rArgClustIdx_partCnt:
                score += xlogx(rtc_new - origPartCnt) - denomnr
                continue
            elif rArgClustIdx_partCnt[aci] > 0:
                score += xlogx(rtc_new - rArgClustIdx_partCnt[aci]) - denomnr

            score = update_score_from_dict(score, 
                                           rArgClustIdx_argNum_cnt[aci], 
                                           ac._argNum_cnt)

            score -= 2 * (xlogx(rArgClustIdx_argCnt[aci]) - xlogx(ac._ttlArgCnt))

            score = update_score_from_dict(score, 
                                           rArgClustIdx_argTypeIdx_cnt[aci], 
                                           ac._argTypeIdx_cnt)

            score = update_score_from_dict(score, 
                                           rArgClustIdx_chdClustIdx_cnt[aci], 
                                           ac._chdClustIdx_cnt)

        # line 570 in Scorer.java

        for aci, ac in acl._argClusts.items():
            origPartCnt = len(ac._partRootTreeNodeIds)
            score -= xlogx(acl._ttlArgCnt - origPartCnt) + denomoa

            if aci not in aArgClustIdx_partCnt:
                score += xlogx(atc_new - origPartCnt) + denomoa
                continue
            elif aArgClustIdx_partCnt[aci] > 0:
                score += xlogx(atc_new - aArgClustIdx_partCnt[aci]) + denomoa

            score = update_score_from_dict(score, 
                                           aArgClustIdx_argNum_cnt[aci],
                                           ac._argNum_cnt)
            
            score -= 2 * (xlogx(aArgClustIdx_argCnt[aci]) - xlogx(ac._ttlArgCnt))

            score = update_score_from_dict(score, 
                                           aArgClustIdx_argTypeIdx_cnt[aci],
                                           ac._argTypeIdx_cnt)
            score = update_score_from_dict(score, 
                                           aArgClustIdx_chdClustIdx_cnt[aci],
                                           ac._chdClustIdx_cnt)

        for ds in [(rNewArgClustIdx_partCnt, rNewArgClustIdx_argNum_cnt), 
                   (aNewArgClustIdx_partCnt, rNewArgClustIdx_argNum_cnt)]:
            for aci, partCnt in ds[0].items():
                score += xlogx(ratc_new-PartCnt) - denomra

                for idx, cnt in ds[1].items():
                    score += xlogx(cnt)
                    score -= ParseParams.priorNumParam

        for ds in [(rNewArgClustIdx_argCnt, rNewArgClustIdx_argTypeIdx_cnt, rNewArgClustIdx_chdClustIdx_cnt), 
                   (aNewArgClustIdx_argCnt, aNewArgClustIdx_argTypeIdx_cnt, aNewArgClustIdx_chdClustIdx_cnt)]:
            for aci, argCnt in ds[0].items():
                score -= 2 * xlogx(argCnt)

                for idx, cnt in ds[1].items():
                    score += xlogx(cnt)
                    score -= ParseParams.priorNumParam

                for idx, cnt in ds[2].items():
                    score += xlogx(cnt)
                    score -= ParseParams.priorNumParam

        return score

    def scoreOpComposePart(self, pp, cp):
        score = 0
        rcl, acl = [Clust.getClust(x._clustIdx) for x in (pp, cp)]

        ptn, ctn = [x._relTreeRoot for x in (pp, cp)]
        ptn.addChild(dep, ctn)
        nrti = RelType.getRelType(ptn)

        if Clust.getClustsWithRelType(nrti) is None:
            return score

        pai = cp._parArgIdx
        pcarg = pp.getArguments()[pai]
        dep = pcarg._path.getDep()
        orti = pp._relTypeIdx

        ncl = Clust.getClust(Clust.getClustsWithRelType(nrti).next())
        nci = ncl._clustIdx

        if pp.getParPart() is not None:
            ppp = pp.getParPart()
            ppcl = Clust.getClust(ppp.getClustIdx())
            ac = ppcl._argClusts[ppp.getArgClust(pp.getParArgIdx())]
            oc = ac._chdClustIdx_cnt[rcl._clustIdx]
            nc = ac._chdClustIdx_cnt[nci]
        else:
            oc = Clust.clustIdx_rootCnt[rcl]
            nc = Clust.clustIdx_rootCnt[ncl]

        score += log(nc) - log(oc)

        for aci, ais in pp._argClustIdx_argIdxs.items():
            ac = rcl._argClusts[aci]
            score -= (log(ac._argNum_cnt[len(ais)])-log(ac._ttlArgCnt))

            for ai in ais:
                arg = pp.getArgument(ai)
                score -= (log(ac._chdClustIdx_cnt[arg._argPart._clustIdx]) \
                        - log(ac._ttlArgCnt))
                score -= (log(ac._argTypeIdx_cnt[arg._path.getArgType()]) \
                        - log(ac._ttlArgCnt))

        ai_newaci = dict()

        for ai, arg in pp._args.items():
            if ai == pai:
                pass
            else:
                ati = arg._path.getArgType()
                aci = ncl._argTypeIdx_argClustIdxs[ati].next()
                ai_newaci[ai] = aci

        newArgClustIdx_ais = dict()

        for ai, aci in ai_newaci.items():
            if aci not in newArgClustIdx_ais:
                newArgClustIdx_ais[aci] = set()
            
            newArgClustIdx_ais[aci].add(ais)

        for aci, ais in newArgClustIdx_ais.items():
            ac = ncl._argClusts[aci]
            score += (log(ac._argNum_cnt[len(ais)])-log(ac._ttlArgCnt))

            for ai in ais:
                arg = pp.getArgument(ai)
                score -= (log(ac._chdClustIdx_cnt[arg._argPart._clustIdx]) \
                        - log(ac._ttlArgCnt))
                score -= (log(ac._argTypeIdx_cnt[arg._path.getArgType()]) \
                        - log(ac._ttlArgCnt))

        return score

    def scoreMCForAlign(self, clx1, clx2, aci2_aci1):
        finalScore = 0

        acIdxs1, acIdxs2 = [x._argClusts for x in (clx1, clx2)]
        tc1, tc2 = [x._ttlCnt for x in (clx1, clx2)]

        denom  = xlogx(tc1+tc2)
        denom1 = xlogx(tc1)
        denom2 = xlogx(tc2)

        deltaNoMergeArgClust = 0

        for ac in acIdxs1.values():
            pc = len(ac._partRootTreeNodeIds)
            deltaNoMergeArgClust +=  xlogx(tc1+tc2-pc) \
                                   - denom \
                                   - xlogx(tc1-pc) \
                                   + denom1

        for ac in acIdxs2.values():
            pc = len(ac._partRootTreeNodeIds)
            deltaNoMergeArgClust +=  xlogx(tc1+tc2-pc) \
                                   - denom \
                                   - xlogx(tc2-pc) \
                                   + denom1

        for ai2, ac2 in acIdxs2.items():
            ptids2 = ac2._partRootTreeNodeIds
            pc2 = len(ptids2)
            tac2 = ac2._ttlArgCnt

            newBaseScore = xlogx(tc1+tc2-pc2) - denom - (2 * xlogx(tac2))
            maxScore = newBaseScore
            maxMap = -1

            for j, (ai1, ac1) in enumerate(acIdxs1.items()):
                ptids1 = ac1._partRootTreeNodeIds
                pc1 = len(ptids1)
                tac1 = ac1._ttlArgCnt

                if pc1 == 0:
                    continue
                elif pc2 == 0:
                    aci2_aci1[ai2] = ai1
                    maxScore = 0
                    break

                score = 0
                score -= ParseParams.priorMerge
                score +=  xlogx(tc1+tc2-pc1-pc2) \
                        - xlogx(tc1+tc2-pc1) \
                        + (2 * xlogx(tac1)) \
                        - (2 * xlogx(tac1+tac2))

                argNum_newCnt = dict()

                for an, c in ac1._argNum_cnt.items():
                    argNum_newCnt = inc_key(argNum_newCnt, an, inc=c)

                for an, c in ac2._argNum_cnt.items():
                    argNum_newCnt = inc_key(argNum_newCnt, an, inc=c)

                # There is a while() loop in the original code right here 
                # that is the same as the one in scoreMergeArgs() but here
                # it doesn't seem to do anything except error out if a certain
                # condition is met: Scorer.java line 950

                for d in [argNum_newCnt, ac1._argNum_cnt, ac2._argNum_cnt]:
                    for c in d.values():
                        if c > 0:
                            score -= xlogx(c) + ParseParams.priorNumParam

                atc1, atc2 = [x._argTypeIdx_cnt for x in (ac1, ac2)]
                score = Scorer.update_score_from_ds(score, atc1, atc2)

                ccc1, ccc2 = [x._chdClustIdx_cnt for x in (ac1, ac2)]
                score = Scorer.update_score_from_ds(score, ccc1, ccc2)

                if score > maxScore:
                    maxScore = score
                    maxMap = j
                    aci2_aci1[ai2] = ai1

            finalScore += maxScore - newBaseScore

        finalScore += deltaNoMergeArgClust

        return finalScore

    def scoreMergeArgs(self, clust, ai1, ai2):
        score = -ParseParams.priorMerge
        ac1, ac2 = [clust._argClusts[x] for x in (ai1, ai2)]
        tpc = clust._ttlCnt
        ptids1, ptids2 = [x._partRootTreeNodeIds for x in (ac1, ac2)]
        tpc1, tpc2 = [len(x) for x in (ptids1, ptids2)]
        tac1, tac2 = [x._ttlArgCnt for x in (ac1, ac2)]

        score -= (xlogx(tpc-tpc1) + xlogx(tpc-tpc2))
        score += xlogx(tpc)
        score -= (2 * (xlogx(tac1+tac2)-xlogx(tac1)-xlogx(tac2)))

        argNum_newCnt = dict()

        for d in (ac1._argNum_cnt, ac2._argNum_cnt):
            for an, cnt in d.items():
                if cnt == 0:
                    raise ZeroArgumentException
                else:
                    score -= xlogx(cnt)

                argNum_newCnt = inc_key(argNum_newCnt, an, inc=cnt)

        tpc12 = tpc1 + tpc2
        sit1, sit2 = [java_iter(x) for x in (ptids1, ptids2)]
        pid1, pid2 = [x.next() for x in (sit1, sit2)]

        while True:
            if pid1 == pid2:
                c1 = len(Part.getPartByRootNodeId(pid1)._argClustIdx_argIdxs[ai1])
                c2 = len(Part.getPartByRootNodeId(pid2)._argClustIdx_argIdxs[ai2])
                c0 = c1 + c2
                tpc12 -= 1

                argNum_newCnt = inc_key(argNum_newCnt, c0)
                argNum_newCnt = dec_key(argNum_newCnt, c1, remove=True)
                argNum_newCnt = dec_key(argNum_newCnt, c2, remove=True)

                if not (sit1.hasnext() or sit2.hasnext()):
                    break

                pid1, pid2 = [x.next() for x in (sit1, sit2)]
            elif pid1 < pid2:
                while sit1.hasnext():
                    pid1 = sit1.next()

                    if pid1 >= pid2:
                        break

                if pid1 < pid2:
                    break
            else:
                while sit2.hasnext():
                    pid2 = sit2.next()

                    if pid1 <= pid2:
                        break

                if pid1 > pid2:
                    break

        score += xlogx(tpc - tpc12)

        for c in argNum_newCnt.values():
            score += xlogx(c)

        score += ((len(ac1._argNum_cnt)+len(ac2._argNum_cnt)-len(argNum_newCnt)) \
                 * ParseParams.priorNumParam)

        atc1, atc2 = [x._argTypeIdx_cnt for x in (ac1, ac2)]
        score = Scorer.update_score_from_ds(score, atc1, atc2)

        ccc1, ccc2 = [x._chdClustIdx_cnt for x in (ac1, ac2)]
        score = Scorer.update_score_from_ds(score, ccc1, ccc2)

        return score
        

    def update_score_from_ds(score, cntd1, cntd2):
        if len(cntd1) <= len(cntd2):
            for key in cntd1:
                if key in cntd2:
                    cx1, cx2 = [x[key] for x in (cntd1, cntd2)]
                    score = Scorer.updateScore(cx1, cx2)
                    score += ParseParams.priorNumParam
        else:
            for key in cntd2:
                if key in cntd1:
                    cx1, cx2 = [x[key] for x in (cntd1, cntd2)]
                    score = Scorer.updateScore(cx1, cx2)
                    score += ParseParams.priorNumParam

        return score

    def updateScore(x, y):
        update = xlogx(x+y) - xlogx(x) - xlogx(y)

        return update

