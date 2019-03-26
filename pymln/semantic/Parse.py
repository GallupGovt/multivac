
from datetime import datetime
from sortedcontainers import SortedDict
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

        self.id_article = SortedDict()

        self.rootTreeNodeIds = set()
        self.parseReader = StanfordParseReader()
        self.scorer = Scorer()
        self.executor = Executor(self)
        self.agenda = Agenda(self)

    def createArgs(self, art_id, sent_id, sent, parent_id, 
                   done=set(), verbose=False):
        '''
            For each token, get the TreeNode, Part, Cluster and (based on 
            sentence dependencies) the children tokens.

            For each child token, use the dependency relationship to define 
            a Path and then argument type and Argument defining the parent-
            child relationship. Then add/create an ArgClust before recursing
            on any grand-child tokens. 

            #
            ## CHECK TOKENS SO WE DON'T GET STUCK IN A RECURSIVE LOOP IF 
            ## DEPENDENCIES ARE MALFORMED
            # 
        '''
        parent_node_id = genTreeNodeID(art_id, sent_id, parent_id)
        parent = TreeNode.getTreeNode(parent_node_id)
        parent_part = Part.getPartByRootNodeId(parent_node_id)
        parent_clust = Clust.getClust(parent_part.getClustIdx())
        children = sent.get_children(parent_id)

        if children is not None:
            for relation, child_id in children:
                child_node_id = genTreeNodeID(art_id, sent_id, child_id)
                path = Path(relation)
                arg_type_id = path.getArgType()

                # if child_node_id in done:
                #     continue

                child_part = Part.getPartByRootNodeId(child_node_id)

                if child_part is None:
                    if verbose:
                        print("Child node id {} has no part".format(child_node_id))
                
                if child_part.getParPart() is not None:
                    if verbose:
                        print("Child node id {} already has "
                              "parent {}".format(child_node_id,
                                                 child_part.getParPart().getRelTreeRoot().getId()))
                    continue
                
                arg = Argument(parent, path, child_part)
                arg_id = parent_part.addArgument(arg)
                child_part.setParent(parent_part, arg_id)

                arg_clust_ids = parent_clust.getArgClustIdxs(arg_type_id)
                
                if arg_clust_ids is None:
                    arg_clust_id = parent_clust.createArgClust(arg_type_id)
                else:
                    arg_clust_id = next(iter(arg_clust_ids))

                parent_part.setArgClust(arg_id, arg_clust_id)

                #done.add(child_node_id)
                self.createArgs(art_id, sent_id, sent, child_id)
        
        #done.add(parent_node_id)

        return None

    def chkArgs(self):
        '''
        To Do: for debugging purposes
        '''

        return None

    def initialize(self, arts, verbose=False):
        for art in arts:
            self.id_article[art.uid] = art
            self.numSents += len(art.sentences)

            if verbose:
                    print("Article {} processing...".format(art.uid))

            for i, sent in enumerate(art.sentences):
                self.initializeSent(art.uid, i, sent, verbose)

        return None

    def initializeSent(self, ai, sj, sent, verbose=False):
        '''
            Create TreeNode, Part, and Clust for each token in a sentence,
            also adding/assigning RelTypes.
            Increment the root count for the cluster assigned to the root
            token (tokens with a parent of ROOT). 
            Finally, run CreateArgs() to define the parent-child relation-
            ships. This call is recursive, traversing the whole dependency
            tree for each sentence.
        '''
        self.numTkns += len(sent.get_tokens())-1
        roots = sent.get_children(0)

        if roots is None:
            return None
        elif len(roots) == 0:
            return None

        for k in range(1, len(sent.get_tokens())):
            Parse.part_from_node(ai, sj, sent, k, sent.get_token(k))

        # if len(roots) == 1:
        for _, idx in roots:
            sub_node_id = genTreeNodeID(ai, sj, idx)
            # Is this global set really necessary? I don't think it is...
            self.rootTreeNodeIds.add(sub_node_id)
            node_part = Part.getPartByRootNodeId(sub_node_id)

            if node_part is None:
                continue
            
            ncl = Clust.getClust(node_part.getClustIdx())
            ncl.incRootCnt()
            self.createArgs(ai, sj, sent, idx, verbose)

        return None

    def part_from_node(ai, sj, sent, k, tok):
            if not Parse.isIgnore(sent, k):
                tn = TreeNode(genTreeNodeID(ai,sj,k), tok)
                part = Part(tn)
                relTypeIdx = part.getRelTypeIdx()
                clustIdxs = Clust.getClustsWithRelType(relTypeIdx)

                if clustIdxs is not None:
                    clustIdx = next(iter(clustIdxs))
                else: 
                    clustIdx = Clust.createClust(relTypeIdx)

                part.setClust(clustIdx)

            return None

    def isIgnore(sent, k):
        done = set()

        while True:
            parent = sent.get_parent(k)
            
            if parent is None:
                break
            elif parent in done:
                break
            else:
                done.add(k)
                k = parent[1]

        return (k>0)

    def mergeArgs(self, verbose=False):
        '''
            For each cluster, count up all the arguments for each ArgClust. 
            Iterating from most args to least, for each ArgClust score whether
            merging it makes sense.
        '''
        i = 0
        for clust_id, clust in Clust.clusts.items():
            new_arg_clusts = {}
            counts_per_ArgClust = []

            for arg_clust_id, arg_clust in clust._argClusts.items():
                arg_count = arg_clust._ttlArgCnt
                counts_per_ArgClust.append((arg_count, arg_clust_id))

            counts_per_ArgClust.sort(reverse=True)

            for _, arg_clust_id in counts_per_ArgClust:
                arg_clust = clust._argClusts[arg_clust_id]

                if len(new_arg_clusts) == 0:
                    new_arg_clusts[arg_clust_id] = arg_clust
                    continue

                maxScore = 0
                maxMap = -1

                for aci in new_arg_clusts.keys():
                    # This sorting is not necessary - for debugging only
                    # remove on final version
                    score = self.scorer.scoreMergeArgs(clust, aci, arg_clust_id)

                    if score > maxScore:
                        maxScore = score
                        maxMap = aci

                if maxMap >= 0:
                    self.executor.mergeArg(clust, maxMap, arg_clust_id)
                else:
                    new_arg_clusts[arg_clust_id] = arg_clust

            clust._argClusts = new_arg_clusts

            i += 1

            if verbose:
                if i%100==0:
                    print("{} MergeArgs: {} clusters processed.".format(datetime.now(), 
                                                                        i))

        return None

    def parse(self, files, DIR, verbose=False):
        articles = []

        for file in files:
            a = StanfordParseReader.readParse(file, DIR)
            articles.append(a)

        self.initialize(articles)
        self.mergeArgs()
        self.agenda.createAgenda()
        self.agenda.procAgenda()

        return None

    def reparse(self, aid, si):
        a = id_article[aid]
        sent = a.sentences[si]

        roots = sent.get_children(0)

        if roots is None:
            return None
        elif len(roots) == 0:
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

            if len(roots) == 1:
                _, idx = next(iter(roots))
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

    def setArgs(self, art_id, sent_id, sent, idx):
        this_part = Part.getPartByRootNodeId(genTreeNodeID(art_id, sent_id, idx))
        this_node = this_part.getRelTreeRoot()
        node_clust = Clust.getClust(this_part.getClustIdx())
        children = sent.get_children(idx)

        if children is None:
            return None
        else:
            for dependency, child_index in children:
                child_node_id = genTreeNodeID(art_id, sent_id, child_index)
                path = Path(dependency)
                argTypeIdx = path.getArgType()
                child_part = Part.getPartByRootNodeId(child_node_id)

                if child_part.getParPart() is not None:
                    continue

                arg = Argument(this_node, path, child_part)
                argIdx = this_part.addArgument(arg)
                child_part.setParent(this_part, argIdx)
                argClustIdxs = node_clust.getArgClustIdxs(argTypeIdx)
                argClustIdx = -1

                if argClustIdxs is None:
                    argClustIdx = node_clust.createArgClust(argTypeIdx)
                else:
                    argClustIdx = next(iter(argClustIdxs))

                this_part.setArgClust(argIdx, argClustIdx, clust_only=True)

                setArgs(art_id, sent_id, sent, child_index)

        return None
