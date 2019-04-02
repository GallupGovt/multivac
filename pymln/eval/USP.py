
import argparse
import os
import re

import corenlp

from sortedcontainers import SortedDict, SortedSet

from multivac import settings
from multivac.pymln.utils import Utils
from multivac.pymln.syntax.Nodes import Article, Sentence, Token
from multivac.pymln.semantic import MLN, Part, Clust
from multivac.pymln.syntax.Relations import RelType, ArgType
from multivac.pymln.eval import Answer, Question

class stanford_token():
    def __init__(self, text='', index=None, lemma_='', pos_='',
                 ner='', dep_='', head=None):
        self.i = index
        self.text = text
        self.lemma_ = lemma_
        self.pos_ = pos_
        self.ner = ner
        self.dep_ = dep_
        self.head = head
        self.has_children = False

    def __repr__(self):
        return "{}:{}=>{}:{}".format(self.i,
                                     self.text,
                                     self.dep_,
                                     self.head)

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return self.compareTo(other) == 0

    def __lt__(self, other):
        return self.compareTo(other) < 0

    def compareTo(self, other):
        result = 0

        if self.__repr__() != other.__repr__():
            if self.__repr__() < other.__repr__():
                result -= 1
            else:
                result += 1

        return result


class stanford_parse():
    def __init__(self, sentence, deptype='basicDependencies'):
        self.tokens = []
        self.root = 0

        if isinstance(sentence,str):
            sentence = stanford_parse.get_parse(sentence)

        self.deps = sorted(sentence[deptype], key=lambda k: k['dependent'])

        for i, w in enumerate(sentence['tokens']):
            tok = stanford_token(text=w['originalText'],
                                 index=w['index'],
                                 lemma_=w['lemma'],
                                 pos_=w['pos'],
                                 ner=w['ner'],
                                 dep_=self.deps[i]['dep'].replace(":",""),
                                 head=self.deps[i]['governor']-1)
            self.tokens.append(tok)
            if tok.dep_ == 'ROOT':
                self.root = len(self.tokens)-1

        self._tokens = {t.text: t.i for t in self.tokens}

        for tok in self.tokens:
            self.tokens[tok.head].has_children = True

    def __repr__(self):
        return ' '.join(["{}".format(t) for t in self.tokens])

    def get_root(self):
        return self.tokens[self.root]

    def get_parse(sentence):
        anns = "tokenize ssplit pos lemma ner depparse"
        with corenlp.CoreNLPClient(annotators=anns.split(), output_format='json') as client:
            ann = client.annotate(sentence)

        return ann['sentences'][0]

    def get_deps(sentence, deptype='basicDependencies', ret='asis'):
        if isinstance(sentence,str):
            sentence = stanford_parse.get_parse(sentence, deptype)

        deps = sentence[deptype]

        if ret == 'asis':
            retval = deps
        else:
            retval = {}
            retval['deps'] = {x['dep']: x['dependent'] for x in deps}
            retval['heads'] = {x['dependentGloss']: x['governorGloss'] for x in deps}
            retval['governors'] = {x['dependent']: x['governorGloss'] for x in deps}
            retval['dependents'] = {x['dependent']: x['dependentGloss'] for x in deps}
            retval['text'] = ["{}({}-{}, {}-{})".format(x['dep'],
                                                x['governorGloss'],
                                                x['governor'],
                                                x['dependentGloss'],
                                                x['dependent']) for x in deps]
        return retval

    def get_children(self, tok):
        return set([t for t in self.tokens if t.head+1 == tok.i])

    def toString(self):
        return ' '.join([t.text for t in self.tokens])


class USP(object):
    allowedDeps = set(['nn','nmod','obl','amod','case','num','appos'])
    target_args = set(['nsubj','nsubjpass','dobj','attr','pobj'])
    five_ws_and_h = ['who','what','where','when','why','how']
    evalDir = ''
    resultDir = ''
    query_file = ''

    qas = SortedDict() # {Question: set(Answers)}
    rel_qs = SortedDict() # {str: list(Questions)}

    # Library of tokens
    qForms = set()
    qLemmas = set()
    form_lemma = dict() # {str: set(str)}

    # Clusters, Argument Clusters, Parts
    headDep_clustIdxs = dict() # {(str,str): str}
    lemma_clustIdxs = dict() # {str: set(str)}
    rel_clustIdx = dict() # {str: int}
    clustIdx_depArgClustIdx = dict() # {int: {str: int}}
    arg_cis = dict() # {str: list(list(str))}
    ptId_clustIdxStr = dict() # {str: (int, str)}
    ptId_aciChdIds = dict() # {str: {int: set(str)}}
    ptId_parDep = dict() # {str: str}

    id_sent = dict() # {str: str}
    id_article = dict() # {str: Article}

    def readQuestions(verbose=False):
        filename = os.join.path(USP.evalDir, USP.query_file)

        with open(filename, "r") as f:
            lines = f.readlines()

        questions = [stanford_parse(line) for line in lines]

        for question in questions:
            if len(question.tokens) == 0:
                continue

            if verbose:
                print(' '.join([t.text for t in question.tokens]))

            verbs = [t for t in question.tokens if t.pos_.startswith('V')
                                               and t.has_children
                                               and not t.dep_.startswith('aux')
                                               and not t.dep_ in ['cop','dep']]
            if len(verbs) == 0:
                verbs = [question.get_root()]

            if verbose:
                print("Key relations: {}".format(verbs))

            for t in question.tokens:
                if t.text not in USP.form_lemma:
                    USP.form_lemma[t.text] = set()

                USP.form_lemma[t.text].add(t.lemma_)

            for rel in verbs:
                args = [t for t in question.get_children(rel)
                                if t.pos_.startswith('N')
                                or 'subj' in t.dep_
                                or 'obj' in t.dep_]

                if len(args) == 0:
                    args = [t for t in question.tokens if t.pos_.startswith('N')]

                arg = [t for t in args if t.dep_.startswith('nsubj')]

                if len(arg) == 0:
                    if len(args) == 0:
                        if verbose:
                            print("Skipping question as unparsable: "
                                  "{}".format(question.toString()))
                        continue
                    else:
                        arg = [args[0]]

                    dep = 'nsubj'
                else:
                    if len(args) > 1:
                        dep = [t for t in args if t not in arg[0:1]][0].dep_
                    else:
                        dep = 'dobj'

                # Sub older version of the obl dependency type.
                dep = dep.replace('nmod','obl')

                if verbose:
                    print("Main arguments: {} and {}".format(arg, dep))

                if arg[0].has_children:
                    if verbose:
                        print("Argument has children; building sub-tree.")
                        arg += question.get_children(arg[0])
                    # arg += sorted(USP.build_subtree(question, arg[0],
                    #               verbose=verbose), key=lambda k: k.i)

                if verbose:
                    print("Arg sub-tree: {}".format(arg))

                qu = Question(rel.text, ' '.join([t.text for t in arg]), dep)

                if rel.text not in USP.rel_qs:
                    USP.rel_qs[rel.text] = list()

                USP.rel_qs[rel.text].append(qu)
                USP.qForms.update(arg + [rel.text])

                del arg
                del rel
                del dep

        return None

    def build_subtree(q, parent, children=set()):
        if parent.has_children:
            for child in q.get_children(parent):
                children.add(child)
                children = children.union(USP.build_subtree(q, child, children))

        return children

    def readSents(aid=None, filename=None):
        '''
            Read in the actual text of the sentences in each article.
            Ensure that the old Parse.id_article dictionary is loaded/
            recreated. However, this time don't ignore any dependencies?
        '''
        idx = 0
        with open(filename, "r") as f:
            line = f.readline()

            while line:
                line = line.trim()

                if len(line) == 0:
                    continue
                else:
                    idx += 1
                    sid = aid + ":" + str(idx)
                    USP.id_sent[sid] = line

        return None

    def readPart():
        Part.clustIdx_partRootNodeIds = Part.clustIdx_partRootNodeIds
        USP.ptId_clustIdxStr = {k: (p.getClustIdx(),
                                    p.getRelTreeRoot().getTreeStr())
                                for k, p in Part.rootNodeId_part.items()}

        for pid, part in Part.rootNodeId_part.items():
            USP.ptId_clustIdxStr[pid] = (part.getClustIdx(),
                                         part.getRelTreeRoot().getTreeStr())

            if part._parPart is not None:
                par = part._parPart._relTreeRoot.getId()
                arg = part._parPart.getArgument(part._parArgIdx)
                aci = part._parPart.getArgClust(part._parArgIdx)
                dep = ArgType.getArgType(arg._path.getArgType()).toString()[1:-1]
                USP.ptId_parDep[pid] = dep

                if par not in USP.ptId_aciChdIds:
                    USP.ptId_aciChdIds[par] = {aci: set()}
                elif aci not in USP.ptId_aciChdIds[par]:
                    USP.ptId_aciChdIds[par][aci] = set()

                cids = USP.ptId_aciChdIds[par][aci]
                cids.add(pid)
                USP.ptId_aciChdIds[par][aci] = cids

        return None

    def procRelType(clustIdx, POS, rel):
        if rel in USP.rel_qs and POS.startswith('V'):
            USP.rel_clustIdx[rel] = clustIdx

        if ' (' not in rel:
            if rel not in USP.lemma_clustIdxs:
                USP.lemma_clustIdxs[rel] = set()

            USP.lemma_clustIdxs[rel].add(clustIdx)
        else:
            headdep = rel.split(' ', 1)
            head = headdep[0]
            dep = re.search(r'\(\w+:\S+\)', headdep[1]).group()

            if len(dep) > 0:
                dep = dep[dep.index(":")+1:-1]

            if (head and dep) in USP.qLemmas:
                USP.headDep_clustIdxs[(head, dep)] = str(clustIdx)

        return None

    def readClust():
        USP.clustIdx_depArgClustIdx = {} # ci int: {dep str: aci int}

        for cid, clust in Clust.clusts.items():
            USP.clustIdx_depArgClustIdx[cid] = {}

            for arg_clust_id, arg_clust in clust._argClusts.items():
                for ati in arg_clust._argTypeIdx_cnt:
                    arg_type_str = ArgType.getArgType(ati).toString()[1:-1]
                    USP.clustIdx_depArgClustIdx[cid][arg_type_str] = arg_clust_id

            for relType in clust._relTypeIdx_cnt:
                rel_str = RelType.getRelType(relType).toString()
                POS = rel_str[rel_str.index('(')+1:rel_str.index(':')]
                rel = rel_str[rel_str.index(':')+1:rel_str.rfind(")")]
                USP.procRelType(cid, POS, rel)

        return None

    def printAns(answer_file='Answers.txt'):
        with open('{}/Answers.txt'.format(USP.evalDir), 'w') as f:
            for q, a_s in USP.qas.items():
                for ans in a_s:
                    sid = ans.getSentId()
                    sent = USP.id_sent[sid]
                    out  = "<question str=\"{}\">\n".format(q)
                    out += "<answer>{}</answer>\n".format(ans.getRst())
                    out += "<sentence id=\"{}\">{}</sentence>\n".format(sid,
                                                                      sent)
                    out += "</question>\n\n"

                    f.write(out)

        return None

    # def getTreeCis(ptId):
    #     cis = SortedSet()
    #     cis.add(str(USP.ptId_clustIdxStr[ptId][0]))

    #     if ptId in USP.ptId_aciChdIds:
    #         for cids in USP.ptId_aciChdIds[ptId].values():
    #             for cid in cids:
    #                 if USP.ptId_parDep[cid] not in USP.allowedDeps:
    #                     continue

    #                 cis = cis.update(USP.getTreeCis(cid))

    #     return cis

    # def isMatchFromHead(chdPtId, cis):
    #     hci = USP.ptId_clustIdxStr[chdPtId][0]

    #     if hci not in cis:
    #         return False

    #     tcis = USP.getTreeCis(chdPtId)

    #     for x in cis:
    #         ts = x.split()
    #         if tcis.isdisjoint(ts):
    #             return False

    #     return True

    # def isMatch(chdPtId, arg):
    #     allcis = USP.arg_cis[arg]

    #     for cis in allcis:
    #         if USP.isMatchFromHead(chdPtId, cis):
    #             return True

    #     if chdPtId in USP.ptId_aciChdIds:
    #         for cids in USP.ptId_aciChdIds[chdPtId].values():
    #             for cid in cids:
    #                 dep = USP.ptId_parDep[cid]

    #                 if (dep.startswith('conj') and not dep=='conj_negcc') or \
    #                         dep == 'appos':
    #                     for cis in allcis:
    #                         if USP.isMatchFromHead(cid, cis):
    #                             return True

    #     return False

    def getTreeCis(ptId):
        cis = SortedSet()
        cis.add(USP.ptId_clustIdxStr[ptId][0])

        if ptId in USP.ptId_aciChdIds:
            for cids in USP.ptId_aciChdIds[ptId].values():
                for cid in cids:
                    if USP.ptId_parDep[cid] not in USP.allowedDeps:
                        continue

                    cis = cis.update(USP.getTreeCis(cid))

        return cis

    def isMatchFromHead(chdPtId, cis):
        hci = USP.ptId_clustIdxStr[chdPtId][0]

        if hci not in cis:
            return False

        tcis = USP.getTreeCis(chdPtId)

        if tcis.isdisjoint(cis):
            return False

        return True

    def isMatch(chdPtId, arg):
        cis = USP.arg_cis[arg]

        if USP.isMatchFromHead(chdPtId, cis):
            return True

        if chdPtId in USP.ptId_aciChdIds:
            for cids in USP.ptId_aciChdIds[chdPtId].values():
                for cid in cids:
                    dep = USP.ptId_parDep[cid]

                    if (dep.startswith('conj') and not dep=='conj_negcc') or \
                            dep == 'appos':
                        if USP.isMatchFromHead(cid, cis):
                            return True

        return False

    def match(verbose=False):
        '''
            For each question set, get the cluster ID for the verb/relation, and
            the associated Parts for that cluster.
            For each question in the set, get the argument cluster for the type
            specified by getDep() and the one specified by getDep2()
                for each part associated with the question, check for a match
                with those argument clusters
        '''
        bad_qs = 0
        matches = 0

        for reltype, qs in USP.rel_qs.items():
            if reltype not in USP.rel_clustIdx:
                bad_qs += 1

                if verbose:
                    print("I don't understand this question: "
                          "{}".format(qs[0].toString()))
                continue

            clust_id = USP.rel_clustIdx[reltype]
            part_ids = Part.clustIdx_partRootNodeIds[clust_id]

            for q in qs:
                dep2 = 'nsubj'
                dep = q.getDep()

                if 'nsubj' in dep:
                    dep2 = 'dobj'

                if dep in USP.clustIdx_depArgClustIdx[clust_id]:
                    aci  = USP.clustIdx_depArgClustIdx[clust_id][dep]
                else:
                    if dep.startswith('nsubj'):
                        if 'nsubj' in USP.clustIdx_depArgClustIdx[clust_id]:
                            aci  = USP.clustIdx_depArgClustIdx[clust_id]['nsubj']
                        elif 'nsubjpass' in USP.clustIdx_depArgClustIdx[clust_id]:
                            aci  = USP.clustIdx_depArgClustIdx[clust_id]['nsubjpass']
                    elif 'obj' in dep:
                        if 'dobj' in USP.clustIdx_depArgClustIdx[clust_id]:
                            aci  = USP.clustIdx_depArgClustIdx[clust_id]['dobj']
                        elif 'obj' in USP.clustIdx_depArgClustIdx[clust_id]:
                            aci  = USP.clustIdx_depArgClustIdx[clust_id]['obj']

                if dep2 == 'nsubj':
                    if 'nsubj' in USP.clustIdx_depArgClustIdx[clust_id]:
                        aci2  = USP.clustIdx_depArgClustIdx[clust_id]['nsubj']
                    elif 'nsubjpass' in USP.clustIdx_depArgClustIdx[clust_id]:
                        aci2  = USP.clustIdx_depArgClustIdx[clust_id]['nsubjpass']
                elif dep2 == 'dobj':
                    if 'dobj' in USP.clustIdx_depArgClustIdx[clust_id]:
                        aci2  = USP.clustIdx_depArgClustIdx[clust_id]['dobj']
                    elif 'obj' in USP.clustIdx_depArgClustIdx[clust_id]:
                        aci2  = USP.clustIdx_depArgClustIdx[clust_id]['obj']

                for part_id in part_ids:
                    if part_id in USP.ptId_aciChdIds:
                        if aci  in USP.ptId_aciChdIds[part_id] and \
                           aci2 in USP.ptId_aciChdIds[part_id]:
                            matches += 1
                            USP.match_q(q, part_id, aci, aci2)

        if verbose:
            print("Unparsed questions: {}".format(bad_qs))
            print("Matched questions: {}".format(matches))

        return None

    def match_q(q, pid, aci, aci2):
        for x, cids in USP.ptId_aciChdIds[pid].items():
            if x == aci or x == aci2:
                continue
            else:
                if any([USP.ptId_parDep[cid] == 'neg' for cid in cids]):
                    return None

        isMatch = False

        for cid in USP.ptId_aciChdIds[pid][aci]:
            if USP.isMatch(cid, q.getArg()):
                isMatch = True
                break

        if isMatch:
            for cid in USP.ptId_aciChdIds[pid][aci2]:
                USP.findAns(q, cid)

        return None


    def getSentId(ptId):
        return ptId[:ptId.rfind(':')]

    def getArticleId(ptId):
        return ptId[:ptId.index(':')]

    def getSentIdx(ptId):
        return ptId[ptId.index(':')+1:ptId.rfind(':')]

    def getTknIdx(ptId):
        return int(ptId[ptId.rfind(':')+1:])

    def findAns(q, pid):
        sid = USP.getSentId(pid)
        aid = USP.getArticleId(pid)
        sIdx = USP.getSentIdx(pid)
        sent = art._sentences[sIdx]

        pid_minPid = dict()

        ans = USP.findAnsPrep(pid, pid_minPid)

        for a in ans:
            na = SortedSet()
            idx_prep = SortedDict()

            for i in a:
                tknIdx = USP.getTknIdx(i)
                na.add(tknIdx)

                detIdx = -1

                if tknIdx in sent._tkn_children:
                    for depChd in sent._tkn_children[tknIdx]:
                        if depChd[0] == 'det':
                            detIdx = depChd[1]
                            na.add(detIdx)
                            break

                if tknIdx in sent._tkn_par:
                    par = sent._tkn_par[tknIdx]
                    if par[0].startswith('case'):
                        parIdx = par[1]
                        parId = Utils.genTreeNodeId(aid, sIdx, parIdx)

                        if a.contains(parId):
                            prep = par[0]
                            mpid = pid_minPid[i]
                            midx = USP.getTknIdx(mpid)

                            if detIdx >= 0 and detIdx < midx:
                                midx = detIdx

                            idx_prep[midx] = prep

            s = ''

            for i in na:
                if len(idx_prep) > 0:
                    pidx = idx_prep[next(iter(idx_prep))]

                    if i >= pidx:
                        s = ' '.join([s, idx_prep.popitem(0)]).strip()

                word = sent._tokens[i].getForm()
                xid = Utils.genTreeNodeId(aid, sIdx, i)

                if xid in USP.ptId_clustIdxStr:
                    xs = USP.ptId_clustIdxStr[xid][1]

                    if ' ' in xs:
                        word = xs

                s = ' '.join([s, word]).strip()

            if q not in USP.qas:
                USP.qas[q] = set()

            USP.qas[q].add(Answer(sid, s))

        return None


    def findAnsPrep(pid, pid_minPid):
        ans = list()
        curr = list()
        z = SortedDict()
        z[pid] = 1
        curr.append(z)
        pid_minPid[pid] = pid

        if pid in USP.ptId_aciChdIds:
            for cids in USP.ptId_aciChdIds[pid].values():
                for cid in cids:
                    dep = USP.ptId_parDep[cid]

                    if (dep.startswith('conj') and not dep=='conj_negcc') or \
                            dep == 'appos':
                        y = findAnsPrep(cid, pid_minPid)
                        ans += y

                        if pid_minPid[cid] < pid_minPid[pid] < 0:
                            pid_minPid[pid] = pid_minPid[cid]
                    elif dep in USP.allowedDeps:
                        curr1 = list()
                        y = findAnsPrep(cid, pid_minPid)

                        if Utils.compareStr(pid_minPid[cid], pid_minPid[pid]) < 0:
                            pid_minPid[pid] = pid_minPid[cid]

                        for a in curr:
                            for b in y:
                                c = SortedDict(list(a.items())+list(b.items()))
                                curr1.append(c)
                        curr = curr1

        ans.append(curr)

        return ans

    # def preprocArgs():
    #     '''
    #         For each verb and associated set of questions,
    #             for each question, get known argument if not already processed
    #             get the lemma for that argument, get the list of ... NOT SURE
    #     '''
    #     for r, qs in USP.rel_qs.items():
    #         ignoredQs = set()

    #         for q in qs:
    #             if q.getArg() in USP.arg_cis:
    #                 continue

    #             cis = []
    #             x = []
    #             ts = q.getArg().split()
    #             isIgnored = False

    #             for f in ts:
    #                 z = SortedSet()
    #                 if f in ['the','of','in']:
    #                     continue

    #                 if f not in USP.form_lemma:
    #                     isIgnored = True
    #                     break
    #                 else:
    #                     ls = USP.form_lemma[f]

    #                     for l in ls:
    #                         if l in USP.lemma_clustIdxs:
    #                             z.update([str(ci) for ci in USP.lemma_clustIdxs[l]])
    #                     x.append(' '.join(z))

    #             if isIgnored:
    #                 ignoredQs.add(q)
    #                 continue

    #             cis.append(x)

    #             if len(ts) >= 2:
    #                 z = SortedSet()
    #                 hs = USP.form_lemma[ts[-1]]
    #                 ds = USP.form_lemma[ts[-2]]

    #                 for h in hs:
    #                     for d in ds:
    #                         if (h, d) in USP.headDep_clustIdxs:
    #                             z.add(USP.headDep_clustIdxs[(h, d)])

    #                 if len(z) > 0:
    #                     y = x[:-2]
    #                     y.append(' '.join(z))
    #                     cis.append(y)

    #             USP.arg_cis[q.getArg()] = cis

    #         qs = [x for x in qs if x not in ignoredQs]

    #     return None

    def preprocArgs():
        '''
            For each verb and associated set of questions,
                for each question, get known argument if not already processed
                get the lemma for that argument, get the list of ... NOT SURE
        '''
        for r, qs in USP.rel_qs.items():
            ignoredQs = set()

            for q in qs:
                if q.getArg() in USP.arg_cis:
                    continue

                cis = SortedSet()
                ts = q.getArg().split()
                isIgnored = False

                for f in ts:
                    if f in ['the','of','in']:
                        continue

                    if f not in USP.form_lemma:
                        isIgnored = True
                        break
                    else:
                        ls = USP.form_lemma[f]

                        for l in ls:
                            if l in USP.lemma_clustIdxs:
                                cis.update(USP.lemma_clustIdxs[l])

                if isIgnored:
                    ignoredQs.add(q)
                    continue

                if len(ts) >= 2:
                    hs = USP.form_lemma[ts[-1]]
                    ds = USP.form_lemma[ts[-2]]

                    for h in hs:
                        for d in ds:
                            if (h, d) in USP.headDep_clustIdxs:
                                cis.add(USP.headDep_clustIdxs[(h, d)])

                USP.arg_cis[q.getArg()] = cis

            qs = [x for x in qs if x not in ignoredQs]
            USP.rel_qs[r] = qs

        return None


def run():

    MLN.load_mln("{}/mln.pkl".format(USP.resultDir))

    USP.readQuestions(verbose=True)
    USP.readClust()
    USP.readPart()
    USP.preprocArgs()
    USP.match()
    USP.printAns()

    return None

if __name__ == '__main__':
    prs = argparse.ArgumentParser(description='Answer questions using an MLN '
                                     'knowledge base. \n'
                                     'Usage: python -m USP.py [-r results_dir] '
                                     ' [-e eval_dir]')
    prs.add_argument('-r', '--results_dir',
                        help='Directory of MLN results to read in from.')
    prs.add_argument('-p', '--eval_dir',
                        help='Directory to output evaluation files.')
    prs.add_argument('-q', '--query_file',
                        help='File containing the queries to test. Defaults '
                        'to "output_questions_QG-Net.pt.txt.prob.txt".')

    args = vars(prs.parse_args())

    # Default argument values
    params = {'eval_dir': settings.models_dir,
              'results_dir': settings.results_dir,
              'query_file': 'output_questions_QG-Net.pt.txt'}

    # If specified in call, override defaults
    for par in params:
        if args[par] is not None:
            params[par] = args[par]

    USP.query_file = params['query_file']

    if os.path.isabs(params['results_dir']):
        USP.resultDir = params['results_dir']
    else:
        USP.resultDir = os.path.join(os.getcwd(), params['results_dir'])

    if os.path.isabs(params['eval_dir']):
        USP.evalDir = params['eval_dir']
    else:
        USP.evalDir = os.path.join(os.getcwd(), params['eval_dir'])

    run()

