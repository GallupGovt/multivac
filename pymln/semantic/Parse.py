

from semantic import Argument, Clust, Part, Agenda, Executor, Scorer
from syntax.StanfordParseReader import StanfordParseReader
from syntax.Nodes import TreeNode
from syntax.Relations import Path
from utils import genTreeNodeID

class Parse(object):
    def __init__(self, priorNumParam=None, priorNumConj=None):
        self.priorNumConj = priorNumConj
        self.priorNumParam = priorNumParam
        self.numSents = 0
        self.numTkns = 0

        self.id_article = {}

        self.rootTreeNodeIds = set()
#        self.parseReader = StanfordParseReader()
        self.scorer = Scorer()
        self.agenda = Agenda(self)
        self.executor = Executor(self)

    def createArgs(self, ai, sj, sent, idx):
        nid = genTreeNodeID(ai, sj, idx)
        node = TreeNode.getTreeNode(nid)
        np = Part.getPartByRootNodeId(nid)
        ncl = Clust.getClust(np.getClustIdx())
        chds = sent.get_children(idx)

        if chds is not None:
            for dep, cidx in chds:
                cid = genTreeNodeID(ai, sj, cidx)
                p = Path(dep)
                argTypeIdx = p.getArgType()
                cp = Part.getPartByRootNodeId(cid)

                if cp.getParPart() is None:
                    continue
                
                arg = Argument(node, p, cp)
                argIdx = np.addArgument(arg)
                cp.setParent(np, argIdx)
                argClustIdxs = ncl.getArgClustIdxs(argTypeIdx)
                argClustIdx = -1
                
                if argClustIdxs is None:
                    argClustIdx = ncl.createArgClust(argTypeIdx)
                else:
                    argClustIdx = next(iter(argClustIdxs))

                np.setArgClust(argIdx, argClustIdx)
                self.createArgs(ai, sj, sent, cidx)

        return None

    def chkArgs(self):
        '''
        To Do: for debugging purposes
        '''

        return None

    def initialize(self, arts, verbose=False):
        #
        # Look to vectorize this
        # 
        for art in arts:
            self.id_article[art.uid] = art
            self.numSents += len(art.sentences)

            for i, sent in art.sentences.items():
                self.initializeSent(art.uid, i, sent)

        return None

    def initializeSent(self, ai, sj, sent):
        self.numTkns += len(sent.get_tokens())-1

        if len(sent.get_children()) < 1 or len(sent.get_children(0)) < 1:
            return None

        for k in range(1,len(sent.get_tokens())):
            if Parse.isIgnore(sent, k):
                continue

            part, clustIdx = Parse.part_from_node(ai, sj, sent, k)

            part.setClust(clustIdx)

        roots = sent.get_children(0)
        
        if len(roots) == 1:
            for idx, child in roots:
                sub_node_id = genTreeNodeID(ai, sj, idx)
                self.rootTreeNodeIds.add(sub_node_id)
                node_part = Part.getPartByRootNodeId(sub_node_id)
                if node_part is None:
                    continue
                ncl = Clust.getClust(node_part.getClustIdx())
                ncl.incRootCnt()
                self.createArgs(ai, sj, sent, idx)

        return None

    def part_from_node(ai, sj, sent, k):
            node_id = genTreeNodeID(ai,sj,k)
            tn = TreeNode(node_id, sent.get_tokens(k))
            part = Part(tn)
            relTypeIdx = part.getRelTypeIdx()
            clustIdx = -1
            clustIdxs = Clust.getClustsWithRelType(relTypeIdx)

            if clustIdxs is not None:
                clustIdx = next(iter(clustIdxs))
            else: 
                clustIdx = Clust.createClust(relTypeIdx)

            return part, clustIdx

    def isIgnore(sent, k):
        while True:
            parent = sent.get_parent(k)
            
            if parent is None:
                break
            else:
                k = parent[1]

        return (k>0)

    def mergeArgs(self):
        for clustIdx in Clust.clusts:
            cl = Clust.getClust(clustIdx)
            newArgClusts = {}
            cnt_acis = []

            for argClustIdx in cl._argClusts:
                acl = cl._argClusts[argClustIdx]
                cnt = acl._ttlArgCnt
                cnt_acis.append((cnt,argClustIdx))

            cnt_acis.sort(reverse=True)

            for item in cnt_acis:
                aci = item[1]
                ac = cl._argClusts[aci]

                if len(newArgClusts) == 0:
                    newArgClusts[aci] = ac

                maxScore = 0
                maxMap = -1

                for acix in newArgClusts:
                    score = self.scorer.scoreMergeArgs(cl, acix, aci)
                    acx = cl._argClusts[acix]

                    if score > maxScore:
                        maxScore = score
                        maxMap = acix

                if maxMap >= 0:
                    acx = cl._argClusts[maxMap]
                    self.executor.mergeArg(cl, maxMap, aci)
                else:
                    newArgClusts[aci] = ac

            cl._argClusts = newArgClusts

        return None

    def parse(self, files, DIR, verbose=False):
        articles = []

        for file in files:
            a = StanfordParseReader.readParse(file, DIR)
            articles.append(a)

        self.initialize(articles)

        if verbose:
            print("{} articles parsed, of {} sentences and "
                "{} total tokens.".format(len(articles), 
                                          self.numSents, 
                                          self.numTkns))
            num_arg_clusts = sum([len(x._argClusts) for x in Clust.clusts.values()])
            print("{} initial clusters, with "
                "{} argument clusters.".format(Clust.nxtClustIdx-1, 
                                                 num_arg_clusts))

        self.mergeArgs()
        self.agenda.createAgenda()
        self.agenda.procAgenda()

        return None

    def reparse(self, aid, si):
        a = id_article[aid]
        sent = a.sentences[si]

        children = sent.get_children(0)

        if children is None:
            return None
        elif len(children) == 0:
            return None
        else:
            old_nid_part = {}

            for ni in range(len(sent.get_tokens())):
                if Parse.isIgnore(sent, ni):
                    continue
                nid = genTreeNodeID(aid, si, ni)
                np = Part.getPartByRootNodeId(nid)
                del Part.rootTreeNodeId_part[nid]
                old_nid_part[nid] = np

            nid_part = {}

            for ni in range(len(sent.get_tokens())):
                if Parse.isIgnore(sent, ni):
                    continue
                part, clustIdx = Parse.part_from_node(aid, si, sent, ni)
                nid_part[genTreeNodeID(aid, si, ni)] = part
                part.setClust(clustIdx, clust_only=True)

            roots = sent.get_children(0)
            assert len(roots) == 1

            dep_idx = next(iter(roots))
            idx = dep_idx[1]
            nid = genTreeNodeID(aid, si, idx)
            np = Part.getPartByRootNodeId(nid)

            if np is not None:
                setArgs(aid, si, sent, idx)

            maxImp = 1

            while maxImp > 0:
                rp, ap = None, None
                maxImp = 0

                for prt in nid_part.values():
                    for arg in prt.getArguments().values():
                        score = self.scorer.scoreOpComposePart(prt,arg)

                        if score > maxImp:
                            maxImp = score
                            rp, ap = prt, arg

                if maxImp <= 0:
                    break

                self.executor.execComposePart(rp, ap)
                del nid_part[ap.getRelTreeRoot().getId()]

            Clust.removePartAndUpdateStat(old_nid_part)
            Clust.updatePartStat(nid_part)

        return None

    def setArgs(self, aid, si, sent, idx):
        nid = genTreeNodeID(aid, si, idx)
        node = TreeNode.getTreeNode(nid)
        np = Part.getPartByRootNodeId(nid)
        ncl = Clust.getClust(np.getClustIdx())
        chds = sent.get_children(idx)

        if chds is None:
            return None
        else:
            for dep, cidx in chds:
                cid = genTreeNodeID(aid, si, cidx)
                p = Path(dep)
                argTypeIdx = p.getArgType()
                cp = Part.getPartByRootNodeId(cid)

                if cp.getParPart() is not None:
                    continue

                arg = Argument(node, p, cp)
                argIdx = np.addArgument(arg)
                cp.setParent(np, argIdx)
                argClustIdxs = ncl.getArgClustIdxs(argTypeIdx)
                argClustIdx = -1

                if argClustIdxs is None:
                    argClustIdx = ncl.createArgClust(argTypeIdx)
                else:
                    argClustIdx = next(iter(argClustIdxs))

                np.setArgClust(argIdx, argClustIdx, clust_only=True)

                setArgs(aid, si, sent, cidx)

        return None
