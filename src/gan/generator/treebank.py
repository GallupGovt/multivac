
# import os
# import pgf
# import stanfordnlp

# from multivac import settings
# from sortedcontainers import SortedDict, SortedSet

# #
# # ADD ERROR HANDLING AND MODEL DOWNLOAD IF stanfordnlp MODELS NOT
# # LOCAL
# # 
# nlp = stanfordnlp.Pipeline(models_dir=settings.stanf_nlp_dir,
#                            treebank='en_ewt', use_gpu=False,
#                            pos_batch_size=3000)

# #
# # Read in Questions
# # 
# #
# # Parse questions
# # 
# # write treebank
# #

# class stanford_token():
#     def __init__(self, text='', index=None, lemma='', pos='',
#                  ner='', dep='', head=None):
#         self.i = index              # index of token in sentence
#         self.text = text            # str verbatim token text
#         self.lemma = lemma          # str lemma
#         self.pos = pos              # str part of speech
#         self.ner = ner              # str named entity if any
#         self.dep = dep              # str dependency relation
#         self.head = head            # index of parent
#         self.has_children = False   # boolean whether has children
#         self.children = SortedSet() # indices of children
#         self.annot = None           # GrammaticalFramework Lexical annotation

#     def __repr__(self):
#         return "{}:{}=>{}:{}".format(self.i,
#                                      self.text,
#                                      self.dep,
#                                      self.head)

#     def __hash__(self):
#         return hash(self.__repr__())

#     def __eq__(self, other):
#         return self.compareTo(other) == 0

#     def __lt__(self, other):
#         return self.compareTo(other) < 0

#     def compareTo(self, other):
#         result = 0

#         if self.__repr__() != other.__repr__():
#             if self.__repr__() < other.__repr__():
#                 result -= 1
#             else:
#                 result += 1

#         return result

# class gf_tree():
#     grammar = None

#     def __init__(self, nlp_text, pgf_file=None):
#         self.tokens = [stanford_token(index=0,dep='root')]
#         self.post_order = {}
#         self.pre_order = {}

#         for w in nlp_text.tokens:
#             tok = stanford_token(text=w.words[0].text,
#                                  index=int(w.words[0].index),
#                                  lemma=w.words[0].lemma,
#                                  pos=w.words[0].upos,
#                                  #ner=w.words[0].ner,
#                                  dep=w.words[0].dependency_relation,
#                                  head=int(w.words[0].governor))

#             if tok.head not in self.post_order:
#                 self.post_order[tok.head] = SortedSet()

#             self.post_order[tok.head].add(tok.i)
#             self.pre_order[tok.i] = tok.head
#             self.tokens.append(tok)

#         for key in self.post_order:
#             self.tokens[key].has_children = True
#             self.tokens[key].children = self.post_order[key]

#         if pgf_file is not None:
#             gf_tree.set_grammar(pgf_file)

#     @classmethod
#     def set_grammar(cls, pgf_file):
#         gr = pgf.readPGF(pgf_file)
#         eng = gr.languages[list(gr.languages)[0]]
#         cls.grammar = eng

#     def string(self):
#         return ' '.join([x.text for x in self.tokens[1:]])

#     def lex_annot(self, tok):
#         tok.annot = gf_tree.grammar.lookupMorpho(tok.lemma)        

# gramm_cats = set()

# for entry in eng.fullFormLexicon():
#     annot = [x[0] for x in entry[1]]
#     annot = [x.split("_")[-1] for x in annot]
#     gramm_cats = gramm_cats.union(set(annot))


#     def traverse(self, func=None, depth=0, from_node=1, order='pre'):
#         if func is None:
#             func = lambda x: print("\t"*depth+"{} {} {} _ {}".format(x.dep, 
#                                                                      x.lemma, 
#                                                                      x.pos, 
#                                                                      x.i))

#         if order == 'pre':
#             # DO something at this stage - default is just print
#             tok = self.tokens[from_node]
#             func(tok)
#             depth += 1

#             for child in self.tokens[from_node].children:
#                 self.traverse(depth=depth, from_node=child, order=order)

#         if order=='post':
#             for child in self.tokens[from_node].children:
#                 self.traverse(from_node=child, order=order)

#             # DO something at this stage - default is just print
#             func(tok)



# def readQuestions(evalDir, query_file, nlp=None, verbose=False):
#     filename = os.path.join(evalDir, query_file)

#     with open(filename, "r") as f:
#         lines = f.readlines()

#     for line in lines:
#         q = gf_tree(nlp(line))


#     questions = [nlp(line) for line in lines]

#     for question in questions:
#         if len(question.tokens) == 0:
#             continue







# #
# # COPIED FOR STRUCTURE -- UPDATE EVERYTHING BELOW HERE
# # 

# if __name__ == '__main__':
#     prs = argparse.ArgumentParser(description='Answer questions using an MLN '
#                                      'knowledge base. \n'
#                                      'Usage: python -m USP.py [-r results_dir] '
#                                      ' [-e eval_dir]')
#     prs.add_argument('-r', '--results_dir',
#                         help='Directory of MLN results to read in from.')
#     prs.add_argument('-p', '--eval_dir',
#                         help='Directory to output evaluation files.')
#     prs.add_argument('-q', '--query_file',
#                         help='File containing the queries to test. Defaults '
#                         'to "output_questions_QG-Net.pt.txt.prob.txt".')

#     args = vars(prs.parse_args())

#     # Default argument values
#     params = {'eval_dir': settings.models_dir,
#               'results_dir': settings.mln_dir,
#               'query_file': 'output_questions_QG-Net.pt.txt'}

#     # If specified in call, override defaults
#     for par in params:
#         if args[par] is not None:
#             params[par] = args[par]

#     USP.query_file = params['query_file']

#     if os.path.isabs(params['results_dir']):
#         USP.resultDir = params['results_dir']
#     else:
#         USP.resultDir = os.path.join(os.getcwd(), params['results_dir'])

#     if os.path.isabs(params['eval_dir']):
#         USP.evalDir = params['eval_dir']
#     else:
#         USP.evalDir = os.path.join(os.getcwd(), params['eval_dir'])

#     run()






    
