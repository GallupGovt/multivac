#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re as reg


def clean_doc(doc, spacynlp):
    '''
    Clean individual documents and remove citations, URLs, emails, other
    trivial content. Returns cleaned doc
    '''
    # Regex for cleaning
    re_citationsNumeric = reg.compile(r'(\[\d+)(,\s*\d+)*]')
    re_url = reg.compile(r'((http|ftp|https):\/\/)?[-a-zA-Z0-9@:%._\+~#=]"'
                         r'{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)')
    re_intextcite = reg.compile(r"((?:[A-Za-z][A-Za-z'`-éü-]+)(?:,? (?:(?:and |& )"
                                r"?(?:[A-Za-z][A-Za-z'`-éü-]+)|(?:et al.?)))*(?:,* "
                                r"*((?:19|20)[0-9][0-9][a-z]*)(\s*&\s*[0-9]*[a-z]*)"
                                r"*(, (\d+))*(?:, p.? [0-9]+)?| *\\((?:19|20)[0-9]"
                                r"[0-9][a-z](\s*&)(?:, p.? [0-9]+)?\\)))")

    re_emptyCite = reg.compile(r"\(([\s]*[;]+[\s]*)+\)")
    re_emptyEg = reg.compile(r'\(e.g.[\s*;\s*]*[,]*\s*\)')
    re_clickHere = reg.compile(r'Click here[^.]*\.')
    re_cid = reg.compile(r"\(cid:\d+\)")
    re_email = reg.compile(r"[\w.-]+@[\w.-]+")
    re_emptyParens = reg.compile(r"\(\s*\)")
    re_emptySee = reg.compile(r"\(see(\s)*\)")
    re_sponsors = reg.compile(r'(This work was supported).+')
    re_arxivHeader = reg.compile(r"(a r X i v).*?(?=[a-zA-Z]{2,})")
    re_vixraHeader = reg.compile(r"^(\s?.?\s)+(v i X r a)")
    re_hyphenatedWords = reg.compile(r'\S(?=\S*[-]\s)([a-zA-Z-]+)(\s)[A-za-z]+')

    # Actual cleaning
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

    # This work supported by --> all the way to end of document
    # Only remove this when it appears in the second half of the article
    for m in reg.finditer(re_sponsors, doc):
        if m.start() > (len(doc)/2):
            doc = reg.sub(re_sponsors, ' ', doc)

    # Handling hyphens - 2-28-2018
    for m in reg.finditer(re_hyphenatedWords, doc):
        match = m.group(0)

        mergedWord = match.replace(' ', '').replace('-', '')
        if mergedWord in spacynlp.vocab:

            doc = doc.replace(match, mergedWord)
        else:
            allWords = True
            for i in match.replace(' ', '').split('-'):
                allWords = allWords and (i in spacynlp.vocab)
            if allWords:
                doc = doc.replace(match, (match.replace(' ', '')))
            else:
                doc = doc.replace(match, mergedWord)

    # De-dup for PUBMED articles, where the main text is sometimes duplicated
    sliceText = doc[0:500]
    count = doc.count(sliceText)

    if count > 1:
        posDup = doc.find(sliceText, 1)
        doc = doc[0:posDup-1]

    return doc
