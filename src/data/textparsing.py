####################################################################
## Load packages
####################################################################
import json
import ast
import pickle
import codecs
import pandas as pd
import re as reg
import numpy as np
import copy
import stanfordnlp
import random


from interruptingcow import timeout
from collections import OrderedDict
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations, implicit_multiplication_application)
from sympy.parsing.latex import parse_latex  
from sympy import *
####################################################################
## Global variables, mostly regex for LateX parsing and text cleaning
####################################################################

### Regex for cleaning 
re_citationsNumeric = reg.compile('(\[\d+)(,\s*\d+)*]')
re_url= reg.compile(r'((http|ftp|https):\/\/)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)')
author = r"(?:[A-Z][A-Za-z'`-]+)"
etal = r"(?:et al.?)"
additional = r"(?:,? (?:(?:and |& )?" + author + "|" + etal + "))"
year_num = r"(?:19|20)[0-9][0-9]"
page_num = r"(?:, p.? [0-9]+)?" 
year = "(?:, *"+year_num+page_num+"| *\("+year_num+page_num+"\))"
re_intextcite = reg.compile(r"((?:[A-Za-z][A-Za-z'`-éü-]+)(?:,? (?:(?:and |& )?(?:[A-Za-z][A-Za-z'`-éü-]+)|(?:et al.?)))*(?:,* *((?:19|20)[0-9][0-9][a-z]*)(\s*&\s*[0-9]*[a-z]*)*(, (\d+))*(?:, p.? [0-9]+)?| *\\((?:19|20)[0-9][0-9][a-z](\s*&)(?:, p.? [0-9]+)?\\)))")

re_emptyCite = reg.compile(r"\(([\s]*[;]+[\s]*)+\)")
re_emptyEg = reg.compile(r'\(e.g.[\s*;\s*]*[,]*\s*\)')
re_clickHere = reg.compile(r'Click here[^.]*\.')
re_cid=reg.compile(r"\(cid:\d+\)")
re_email = reg.compile(r"[\w.-]+@[\w.-]+")
re_emptyParens = reg.compile(r"\(\s*\)")
re_emptySee = reg.compile(r"\(see(\s)*\)")
re_sponsors = reg.compile(r'(This work was supported).+')
re_arxivHeader = reg.compile(r"(a r X i v).*?(?=[a-zA-Z]{2,})")
re_vixraHeader = reg.compile(r"^(\s?.?\s)+(v i X r a)")
re_hyphenatedWords = reg.compile(r'\S(?=\S*[-]\s)([a-zA-Z-]+)(\s)[A-za-z]+')


def clean_doc(doc, spacynlp):   
    '''
    Clean individual documents and remove citations, URLs, emails, other trivial content. Returns cleaned doc
    '''
    doc = reg.sub(re_cid, ' ', doc)
    doc = reg.sub(re_citationsNumeric, ' NumericCitation ', doc)
    doc = reg.sub(re_url, ' ', doc)
    doc = reg.sub(re_intextcite, ' Citation ', doc)
    doc = reg.sub(re_emptyCite, ' ', doc)
    doc = reg.sub(re_emptyEg, ' ', doc)
    doc = reg.sub(re_clickHere, ' ', doc)
    doc = reg.sub(re_email, ' ', doc)
    doc = reg.sub(re_emptyParens, ' ', doc)
    doc = reg.sub(re_emptySee, ' ', doc)
    doc = reg.sub(re_arxivHeader, ' ', doc)
    doc = reg.sub(re_vixraHeader, ' ', doc)
    
    #This work supported by --> all the way to end of document
    #Only remove this when it appears in the second half of the article
    removeSupported = False
    for m in reg.finditer(re_sponsors, doc):
        if m.start()>(len(doc)/2):
            #print('************',m.start(), len(doc))
            doc = reg.sub(re_sponsors, ' ', doc)
    
    #Handling hyphens - 2-28-2018
    for m in reg.finditer(re_hyphenatedWords, doc):
        match=m.group(0)
        
        mergedWord = match.replace(' ', '').replace('-','')
        if mergedWord in spacynlp.vocab: 
            
            doc = doc.replace(match, mergedWord)
        else:
            allWords = True
            for i in match.replace(' ', '').split('-'):
                allWords = allWords and (i in spacynlp.vocab)
            if allWords:
                doc = doc.replace(match,(match.replace(' ', '')) )
            else:
                doc = doc.replace(match, mergedWord)
    
    return doc