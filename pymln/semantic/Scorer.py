
from semantic import SearchOp, Clust, ParseParams, Part
from syntax import RelType
from utils.Utils import inc_key, dec_key, xlogx
from math import log

class Scorer(object):
    def __init__(self):
        return None

    def scoreOp(self, op):
        if op._op == SearchOp.OP_MERGE_CLUST:
            return self.scoreOpMC(op)
        elif op._op == SearchOp.OP_COMPOSE:
            return self.scoreOpCompose(op._parClustIdx, op._chdClustIdx)
        else:
            return -100

    def scoreOpMC(self, op):
        # Get our two cluster ids, and make sure cluster 1 was defined earlier 
        # than cluster 2.
        clust_id1, clust_id2 = op._clustIdx1, op._clustIdx2
        assert clust_id1 < clust_id2

        # Subtract 0 from score? Weird.

        score = 0 - ParseParams.priorMerge

        # If these clusters appear in conjunction with each other (indicating a
        # dissimilarity, otherwise it's redundant) penalize the score.
        #
        # Clust.pairClustIdx_conjCnt is dictionary of type {(int, int): int}
        #

        if (clust_id1, clust_id2) in Clust.pairClustIdx_conjCnt:
            score -= (ParseParams.priorNumConj \
                    * Clust.pairClustIdx_conjCnt[(clust_id1, clust_id2)])

        # Now get the actual Clust objects

        clust1 = Clust.getClust(clust_id1)
        clust2 = Clust.getClust(clust_id2)

        #
        # We update the score by taking xlogx(x+y) - xlogx(x) - xlogx(y)
        # where xlogx() == x*log(x)
        # Here we calculate on total counts - basically how common these clusters
        # are in our corpus. Here we penalize if they are too common.

        score -= Scorer.updateScore(clust1._ttlCnt, clust2._ttlCnt)

        # 
        # Then we check for shared relTypes, and score up those with lots of 
        # shared relTypes. 

        for reltype, count1 in clust1._relTypeIdx_cnt.items():
            if reltype in clust2._relTypeIdx_cnt:
                count2 = clust2._relTypeIdx_cnt[reltype]
                score += Scorer.updateScore(count1, count2)
                score += ParseParams.priorNumParam

        #
        # Bonus as well if we have lots of sentence roots in these clusters, 
        # indicating they are semantically important.

        if clust_id1 in Clust.clustIdx_rootCnt and clust_id2 in Clust.clustIdx_rootCnt:
            root_count1 = Clust.clustIdx_rootCnt[clust_id1]
            root_count2 = Clust.clustIdx_rootCnt[clust_id2]
            score += Scorer.updateScore(root_count1, root_count2)
            score += ParseParams.priorNumParam

        #
        # Let's compare the parent components as well
        # 

        score += self.scoreMCForParent(clust_id1, clust_id2)

        #
        # Finally, if cluster 2 has more arguments than cluster 1, reverse them
        # before scoring on alignment of their arguments

        if len(clust2._argClusts) > len(clust1._argClusts):
            clx1 = clust2
            clx2 = clust1
        else:
            clx1 = clust1
            clx2 = clust2

        score_add, _ = self.scoreMCForAlign(clx1, clx2, dict())
        score += score_add

        return score


    def scoreMCForParent(self, clustIdx1, clustIdx2):
        scr = 0

        if clustIdx1 in Clust.clustIdx_parArgs and clustIdx2 in Clust.clustIdx_parArgs:
            parents1 = Clust.clustIdx_parArgs[clustIdx1]
            parents2 = Clust.clustIdx_parArgs[clustIdx2]

            for par_arg in parents1:
                if par_arg in parents2:
                    par_clust_id, arg_clust_id = par_arg
                    pcl = Clust.getClust(par_clust_id)

                    if pcl is None:
                        print("ERR: ScoreMC parent cluster is null: {}, {}".format(clustIdx1, clustIdx2))
                        continue

                    ac = pcl._argClusts[arg_clust_id]
                    c1 = ac._chdClustIdx_cnt[clustIdx1]
                    c2 = ac._chdClustIdx_cnt[clustIdx2]
                    scr += ParseParams.priorNumParam
                    scr += Scorer.updateScore(c1, c2)

        return scr

    def scoreOpCompose(self, rcidx, acidx):

        def update_score_from_dict(scr, d, orig_d):
            for key, cnt in d.items():
                origcnt = orig_d[key]
                # assert origcnt >= cnt
                scr -= xlogx(origcnt)

                if cnt > 0:
                    scr += xlogx(cnt)
                else:
                    scr += ParseParams.priorNumParam

            return scr

        # get parent and child root-node id numbers
        parChdNids = Part.getPairPartRootNodeIds(rcidx, acidx)

        if parChdNids is None:
            return -10000

        score = 0
        rcl = Clust.getClust(rcidx)
        acl = Clust.getClust(acidx)

        # Parent count, child count, and count of times they occur
        # together. 
        rtc_new = rcl._ttlCnt
        atc_new = acl._ttlCnt
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

        # For each parent-child pair:
        for pcnid in parChdNids:
            pp, cp = Part.getPartByRootNodeId(pcnid[0]), Part.getPartByRootNodeId(pcnid[1])

            rtc_new -= 1
            atc_new -= 1
            ratc_new += 1

            rrt = pp.getRelTypeIdx()
            art = cp.getRelTypeIdx()
            raArgClustidx = pp.getArgClust(cp._parArgIdx)

            # Decrement individual relType counts and increment the combined
            # relType count for this pair
            rRelTypeIdx_newcnt = dec_key(rRelTypeIdx_newcnt,
                                         rrt, 
                                         base=rcl._relTypeIdx_cnt[rrt])

            aRelTypeIdx_newcnt = dec_key(aRelTypeIdx_newcnt,
                                         art, 
                                         base=acl._relTypeIdx_cnt[art])

            raRelTypeIdx_newcnt = inc_key(raRelTypeIdx_newcnt, (rrt, art))

            pp_par = pp.getParPart()

            # If the parent has a parent, increment the parArg count, otherwise
            # increment the root count.
            if pp_par is not None:
                ai = pp.getParArgIdx()
                ppi = pp_par.getClustIdx()
                aci = pp_par.getArgClust(ai)

                parArg_cnt = inc_key(parArg_cnt, (ppi, aci))
            else:
                raRootCnt += 1

            # For each argClust on the parent part, decrement the old parent 
            # part count and argClust count, and increment the new ones. The 
            # trick is don't copy/increment the counts for argument shared by
            # this pair.
            for arg_ci in pp._argClustIdx_argIdxs:
                an = len(pp._argClustIdx_argIdxs[arg_ci])
                ac = rcl._argClusts[arg_ci]

                rArgClustIdx_partCnt = dec_key(rArgClustIdx_partCnt,
                                               arg_ci, 
                                               base=len(ac._partRootTreeNodeIds))
                
                if arg_ci not in rArgClustIdx_argNum_cnt:
                    rArgClustIdx_argNum_cnt[arg_ci] = {}

                rArgClustIdx_argNum_cnt[arg_ci] = \
                    dec_key(rArgClustIdx_argNum_cnt[arg_ci],
                            an, 
                            base=ac._argNum_cnt[an])

                newArgNum = an

                if arg_ci == raArgClustidx:
                    newArgNum -= 1

                if newArgNum == 0:
                    continue

                if arg_ci not in rNewArgClustIdx_argNum_cnt:
                    rNewArgClustIdx_argNum_cnt[arg_ci] = {}

                rNewArgClustIdx_argNum_cnt[arg_ci] = \
                    inc_key(rNewArgClustIdx_argNum_cnt[arg_ci], newArgNum)

                rNewArgClustIdx_partCnt = inc_key(rNewArgClustIdx_partCnt,
                                                   arg_ci)

            # Same as above, but for child part, and we don't skip anything.
            for arg_ci in cp._argClustIdx_argIdxs:
                an = len(cp._argClustIdx_argIdxs[arg_ci])
                ac = acl._argClusts[arg_ci]

                aArgClustIdx_partCnt = dec_key(aArgClustIdx_partCnt,
                                               arg_ci, 
                                               base=len(ac._partRootTreeNodeIds))

                if arg_ci not in aArgClustIdx_argNum_cnt:
                    aArgClustIdx_argNum_cnt[arg_ci] = {}

                aArgClustIdx_argNum_cnt[arg_ci] = \
                    dec_key(aArgClustIdx_argNum_cnt[arg_ci],
                            an, 
                            base=ac._argNum_cnt[an])

                if arg_ci not in aNewArgClustIdx_argNum_cnt:
                    aNewArgClustIdx_argNum_cnt[arg_ci] = {}

                aNewArgClustIdx_argNum_cnt[arg_ci] = \
                    inc_key(aNewArgClustIdx_argNum_cnt[arg_ci], an)

                aNewArgClustIdx_partCnt = inc_key(aNewArgClustIdx_partCnt, arg_ci)

            args = pp.getArguments()

            # For all the parent's arguments
            for ai, arg in args.items():
                arg_part = arg._argPart
                child_clust_id = arg_part._clustIdx
                aci = pp.getArgClust(ai)
                ac = rcl._argClusts[aci]
                ati = arg._path.getArgType()

                # Drop the old arguments

                rArgClustIdx_argCnt = dec_key(rArgClustIdx_argCnt, 
                                              aci, 
                                              base=ac._ttlArgCnt)

                if aci not in rArgClustIdx_argTypeIdx_cnt:
                    rArgClustIdx_argTypeIdx_cnt[aci] = {}

                rArgClustIdx_argTypeIdx_cnt[aci] = \
                    dec_key(rArgClustIdx_argTypeIdx_cnt[aci], 
                            ati, 
                            base=ac._argTypeIdx_cnt[ati])

                if aci not in rArgClustIdx_chdClustIdx_cnt:
                    rArgClustIdx_chdClustIdx_cnt[aci] = {}

                rArgClustIdx_chdClustIdx_cnt[aci] = \
                    dec_key(rArgClustIdx_chdClustIdx_cnt[aci], 
                            child_clust_id, 
                            base=ac._chdClustIdx_cnt[child_clust_id])

                # Add the new arguments, except for the child part we're possibly
                # absorbing

                if arg_part.getRelTreeRoot().getId() != cp.getRelTreeRoot().getId():
                    rNewArgClustIdx_argCnt = inc_key(rNewArgClustIdx_argCnt, aci)

                    if aci not in rNewArgClustIdx_argTypeIdx_cnt:
                        rNewArgClustIdx_argTypeIdx_cnt[aci] = {}

                    rNewArgClustIdx_argTypeIdx_cnt[aci] = \
                        inc_key(rNewArgClustIdx_argTypeIdx_cnt[aci], ati)

                    if aci not in rNewArgClustIdx_chdClustIdx_cnt:
                        rNewArgClustIdx_chdClustIdx_cnt[aci] = {}
                        
                    rNewArgClustIdx_chdClustIdx_cnt[aci] = \
                        inc_key(rNewArgClustIdx_chdClustIdx_cnt[aci], child_clust_id)

            args = cp.getArguments()

            for ai, arg in args.items():
                ap = arg._argPart
                cci = ap._clustIdx
                aci = cp.getArgClust(ai)
                ac = acl._argClusts[aci]
                ati = arg._path.getArgType()

                # Drop the old arguments

                aArgClustIdx_argCnt = dec_key(aArgClustIdx_argCnt, 
                                              aci, 
                                              base=ac._ttlArgCnt)

                if aci not in aArgClustIdx_argTypeIdx_cnt:
                    aArgClustIdx_argTypeIdx_cnt[aci] = {}

                aArgClustIdx_argTypeIdx_cnt[aci] = \
                    dec_key(aArgClustIdx_argTypeIdx_cnt[aci], 
                            ati, 
                            base=ac._argTypeIdx_cnt[ati])

                if aci not in aArgClustIdx_chdClustIdx_cnt:
                    aArgClustIdx_chdClustIdx_cnt[aci] = dict()

                aArgClustIdx_chdClustIdx_cnt[aci] = \
                    dec_key(aArgClustIdx_chdClustIdx_cnt[aci], 
                            cci, 
                            base=ac._chdClustIdx_cnt[cci])

                # Add the new arguments

                aNewArgClustIdx_argCnt = inc_key(aNewArgClustIdx_argCnt, aci)

                if aci not in aNewArgClustIdx_argTypeIdx_cnt:
                    aNewArgClustIdx_argTypeIdx_cnt[aci] = {}
                    
                aNewArgClustIdx_argTypeIdx_cnt[aci] = \
                    inc_key(aNewArgClustIdx_argTypeIdx_cnt[aci], ati)

                if aci not in aNewArgClustIdx_chdClustIdx_cnt:
                    aNewArgClustIdx_chdClustIdx_cnt[aci] = {}
                    
                aNewArgClustIdx_chdClustIdx_cnt[aci] = \
                    inc_key(aNewArgClustIdx_chdClustIdx_cnt[aci], cci)

        if raRootCnt > 0:
            origRootCnt = Clust.clustIdx_rootCnt[rcidx]

            if origRootCnt > raRootCnt:
                score +=  xlogx(raRootCnt) \
                        + xlogx(origRootCnt - raRootCnt) \
                        - xlogx(origRootCnt)
                score -= ParseParams.priorNumParam

        denomor = xlogx(rcl._ttlCnt)
        denomnr = xlogx(rtc_new)

        score = update_score_from_dict(score, 
                                       rRelTypeIdx_newcnt, 
                                       rcl._relTypeIdx_cnt)

        score += denomor
        score -= denomnr

        denomoa = xlogx(acl._ttlCnt)
        denomna = xlogx(atc_new)

        score = update_score_from_dict(score, 
                                       aRelTypeIdx_newcnt, 
                                       acl._relTypeIdx_cnt)

        score += denomoa
        score -= denomna

        for cnt in raRelTypeIdx_newcnt.values():
            score -= ParseParams.priorNumParam
            score += xlogx(cnt)

        denomra = xlogx(ratc_new)
        score -= denomra

        for pi, cnt in parArg_cnt.items():
            pc = Clust.getClust(pi[0])
            ac = pc._argClusts[pi[1]]
            origcnt = ac._chdClustIdx_cnt[rcidx]

            if cnt == origcnt:
                continue

            score -= ParseParams.priorNumParam
            score += xlogx(cnt) + xlogx(origcnt-cnt) - xlogx(origcnt)

        for aci, ac in rcl._argClusts.items():
            origPartCnt = len(ac._partRootTreeNodeIds)
            score -= (xlogx(rcl._ttlCnt - origPartCnt) - denomor)

            if aci not in rArgClustIdx_partCnt:
                score += (xlogx(rtc_new - origPartCnt) - denomnr)
                continue
            
            if rArgClustIdx_partCnt[aci] > 0:
                score += (xlogx(rtc_new - rArgClustIdx_partCnt[aci]) - denomnr)

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
            score -= (xlogx(acl._ttlCnt - origPartCnt) - denomoa)

            if aci not in aArgClustIdx_partCnt:
                score += (xlogx(atc_new - origPartCnt) - denomna)
                continue
            
            if aArgClustIdx_partCnt[aci] > 0:
                score += (xlogx(atc_new - aArgClustIdx_partCnt[aci]) - denomna)

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
                   (aNewArgClustIdx_partCnt, aNewArgClustIdx_argNum_cnt)]:
            for aci, partCnt in ds[0].items():
                score += xlogx(ratc_new-partCnt) - denomra

                for idx, cnt in ds[1][aci].items():
                    score += xlogx(cnt)
                    score -= ParseParams.priorNumParam

        for ds in [(rNewArgClustIdx_argCnt, 
                    rNewArgClustIdx_argTypeIdx_cnt, 
                    rNewArgClustIdx_chdClustIdx_cnt), 
                   (aNewArgClustIdx_argCnt, 
                    aNewArgClustIdx_argTypeIdx_cnt, 
                    aNewArgClustIdx_chdClustIdx_cnt)]:
            for aci, argCnt in ds[0].items():
                score -= 2 * xlogx(argCnt)

                for idx, cnt in ds[1][aci].items():
                    score += xlogx(cnt)
                    score -= ParseParams.priorNumParam

                for idx, cnt in ds[2][aci].items():
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

    def scoreMCForAlign(self, cluster1, cluster2, aci2_aci1):
        finalScore = 0

        arg_clust_indices1 = cluster1._argClusts
        arg_clust_indices2 = cluster2._argClusts

        total_count1 = cluster1._ttlCnt
        total_count2 = cluster2._ttlCnt

        denom  = xlogx(total_count1+total_count2)
        denom1 = xlogx(total_count1)
        denom2 = xlogx(total_count2)

        deltaNoMergeArgClust = 0

        for arg_clust in arg_clust_indices1.values():
            part_cnt = len(arg_clust._partRootTreeNodeIds)
            deltaNoMergeArgClust += (xlogx(total_count1+total_count2-part_cnt) \
                                   - denom \
                                   - xlogx(total_count1-part_cnt) \
                                   + denom1)

        for arg_clust in arg_clust_indices2.values():
            part_cnt = len(arg_clust._partRootTreeNodeIds)
            deltaNoMergeArgClust += (xlogx(total_count1+total_count2-part_cnt) \
                                   - denom \
                                   - xlogx(total_count2-part_cnt) \
                                   + denom2)

        for arg_clust_id2, arg_clust2 in arg_clust_indices2.items():
            part_count2 = len(arg_clust2._partRootTreeNodeIds)
            total_arg_count2 = arg_clust2._ttlArgCnt

            newBaseScore =  xlogx(total_count1 + total_count2 - part_count2) \
                          - denom
            newBaseScore -= 2 * xlogx(total_arg_count2)
            maxScore = newBaseScore
            maxMap = -1

            for arg_clust_id1, arg_clust1 in arg_clust_indices1.items():
                part_count1 = len(arg_clust1._partRootTreeNodeIds)
                total_arg_count1 = arg_clust1._ttlArgCnt

                if part_count1 == 0:
                    continue
                
                if part_count2 == 0:
                    aci2_aci1[arg_clust_id2] = arg_clust_id1
                    maxScore = 0
                    break

                score = 0
                score -= ParseParams.priorMerge
                score +=  xlogx(total_count1 + total_count2 - part_count1 - part_count2) \
                        - xlogx(total_count1 + total_count2 - part_count1) \
                        + (2 * xlogx(total_arg_count1)) \
                        - (2 * xlogx(total_arg_count1 + total_arg_count2))

                argNum_newCnt = dict()

                for arg_num, count in arg_clust1._argNum_cnt.items():
                    argNum_newCnt = inc_key(argNum_newCnt, arg_num, inc=count)

                for arg_num, count in arg_clust2._argNum_cnt.items():
                    argNum_newCnt = inc_key(argNum_newCnt, arg_num, inc=count)

                # There is a while() loop in the original code right here 
                # that is the same as the one in scoreMergeArgs() but here
                # it doesn't seem to do anything except error out if a certain
                # condition is met: Scorer.java line 950

                for count in argNum_newCnt.values():
                    if count > 0:
                        score += xlogx(count) 
                        score -= ParseParams.priorNumParam

                for dictionary in [arg_clust1._argNum_cnt, arg_clust2._argNum_cnt]:
                    for count in dictionary.values():
                        if count > 0:
                            score -= xlogx(count) 
                            score += ParseParams.priorNumParam

                argtype_count1 = arg_clust1._argTypeIdx_cnt
                argtype_count2 = arg_clust2._argTypeIdx_cnt

                score = Scorer.update_score_from_ds(score, 
                                                    argtype_count1, 
                                                    argtype_count2)

                child_clust_count1 = arg_clust1._chdClustIdx_cnt
                child_clust_count2 = arg_clust2._chdClustIdx_cnt

                score = Scorer.update_score_from_ds(score, 
                                                    child_clust_count1, 
                                                    child_clust_count2)

                if score > maxScore:
                    maxScore = score
                    aci2_aci1[arg_clust_id2] = arg_clust_id1

            finalScore += maxScore - newBaseScore

        finalScore += deltaNoMergeArgClust

        return finalScore, aci2_aci1

# cl, arg1=0, arg2=11
    def scoreMergeArgs(self, clust, arg1, arg2):
        # log = open("/Users/ben_ryan/Documents/DARPA ASKE/usp-code/genia_full/score.log", "a+")
        # log.write("Scoring merge for args {} and {} for cluster {}\n".format(arg1, arg2, clust))
        score = 0
        score -= ParseParams.priorMerge
        # log.write("Score = {}\n".format(score))

        total_part_cnt = clust._ttlCnt

        arg_clust1 = clust._argClusts[arg1]
        arg_clust2 = clust._argClusts[arg2]

        part_ids1 = arg_clust1._partRootTreeNodeIds
        part_ids2 = arg_clust2._partRootTreeNodeIds

        total_part_count1 = len(part_ids1)
        total_part_count2 = len(part_ids2)

        total_arg_count1 = arg_clust1._ttlArgCnt
        total_arg_count2 = arg_clust2._ttlArgCnt

        score -= (xlogx(total_part_cnt - total_part_count1) \
                + xlogx(total_part_cnt - total_part_count2))
        # log.write("score -= (xlogx(total_part_cnt - total_part_count1) + xlogx(total_part_cnt - total_part_count2)) = {}\n".format(score))
        score += xlogx(total_part_cnt)
        # log.write("score += xlogx(total_part_cnt) = {}\n".format(score))
        score -= (2 * Scorer.updateScore(total_arg_count1, total_arg_count2))
        # log.write("score -= (2 * Scorer.updateScore(total_arg_count1, total_arg_count2)) = {}\n".format(score))

        argNum_newCnt = dict()

        for dic in (arg_clust1._argNum_cnt, arg_clust2._argNum_cnt):
            for arg_num, count in dic.items():
                if count == 0:
                    print("Zero arguments of type {}".format(arg_num))
                    raise Exception
                else:
                    score -= xlogx(count)
                    # log.write("score -= xlogx({} argnum {}) = {}\n".format(arg_num, count, score))

                argNum_newCnt = inc_key(argNum_newCnt, arg_num, inc=count)

        comb_part_cnt = total_part_count1 + total_part_count2
        part_iter1 = iter(part_ids1)
        part_iter2 = iter(part_ids2)
        pid1 = next(part_iter1)
        pid2 = next(part_iter2)

        while True:
            # log.write("pid1 = {}, pid2 = {}\n".format(pid1, pid2))
            if pid1 == pid2:
                cnt1 = len(Part.getPartByRootNodeId(pid1)._argClustIdx_argIdxs[arg1])
                cnt2 = len(Part.getPartByRootNodeId(pid2)._argClustIdx_argIdxs[arg2])
                comb_cnts = cnt1 + cnt2
                comb_part_cnt -= 1

                argNum_newCnt = inc_key(argNum_newCnt, comb_cnts)
                argNum_newCnt = dec_key(argNum_newCnt, cnt1, remove=True)
                argNum_newCnt = dec_key(argNum_newCnt, cnt2, remove=True)

                try:
                    pid1 = next(part_iter1)
                    pid2 = next(part_iter2)
                except StopIteration:
                    break
            elif pid1 < pid2:
                while True:
                    try:
                        pid1 = next(part_iter1)
                    except StopIteration:
                        break

                    if pid1 >= pid2:
                        break

                if pid1 < pid2:
                    break
            else:
                while True:
                    try:
                        pid2 = next(part_iter2)
                    except StopIteration:
                        break

                    if pid1 <= pid2:
                        break

                if pid1 > pid2:
                    break

        score += xlogx(total_part_cnt - comb_part_cnt)
        # log.write("score += xlogx(total_part_cnt - comb_part_cnt) = {}\n".format(score))

        for count in argNum_newCnt.values():
            score += xlogx(count)
            # log.write("score += xlogx(argNum_newCnt ({})) = {}\n".format(count, score))

        score += ((len(arg_clust1._argNum_cnt) \
                 + len(arg_clust2._argNum_cnt) \
                 - len(argNum_newCnt)) \
                 * ParseParams.priorNumParam)
        # log.write("score += ((len(arg_clust1._argNum_cnt) + len(arg_clust2._argNum_cnt) - len(argNum_newCnt)) * ParseParams.priorNumParam) = {}\n".format(score))

        argtype_count1 = arg_clust1._argTypeIdx_cnt
        argtype_count2 = arg_clust2._argTypeIdx_cnt

        score = Scorer.update_score_from_ds(score, 
                                            argtype_count1, 
                                            argtype_count2)
        # log.write("score after counting ArgTypes = {}\n".format(score))

        child_clust_count1 = arg_clust1._chdClustIdx_cnt
        child_clust_count2 = arg_clust2._chdClustIdx_cnt

        score = Scorer.update_score_from_ds(score, 
                                            child_clust_count1, 
                                            child_clust_count2)
        # log.write("score after counting child clusters = {}\n\n".format(score))
        # log.close()

        return score
        

    def update_score_from_ds(scr, count_dict1, count_dict2):
        if len(count_dict1) <= len(count_dict2):
            for key in count_dict1:
                if key in count_dict2:
                    cnt_1 = count_dict1[key]
                    cnt_2 = count_dict2[key]
                    scr += Scorer.updateScore(cnt_1, cnt_2)
                    scr += ParseParams.priorNumParam
        else:
            for key in count_dict2:
                if key in count_dict1:
                    cnt_1 = count_dict1[key]
                    cnt_2 = count_dict2[key]
                    scr += Scorer.updateScore(cnt_1, cnt_2)
                    scr += ParseParams.priorNumParam

        return scr

    def updateScore(x, y):
        update = xlogx(x+y) - xlogx(x) - xlogx(y)

        return update

