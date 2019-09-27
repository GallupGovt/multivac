## Ontology Construction
### Acquiring Data

To build a robust and diverse set of source models, MULTIVAC has a target of 2,000 scientific journal articles fitting set specifications and filters. These specifications are coded in a user-editable configuration file (by default, ‘multivac.cfg’), and cover both Search parameters (search terms, and sources to search) as well as Filter parameters, to weed out duplicates or unrelated content that might return in a more naïve match on the specified search terms.

To achieve the target sample document size, MULTIVAC targets three different online sources of epidemiological research: arXiv.org, PubMed, and Springer. MULTIVAC accesses these resources through source-specific APIs and authenticates with user API access keys. Each source is searched for articles containing the specified search terms, with matches saved to local disk by DOI number and metadata from those matches stored in JSON files. These three sources also serve to demonstrate MULTIVAC’s ability to work with a variety of data storage types: PubMed articles are ingested from XML, Springer from HTML, and arXiv from PDF. In total, MULTIVAC has scraped a combined 3,740 articles from these three sources: 940 from arXiv, 2,223 from PubMed, and 577 from Springer. This process is illustrated in a set of <a href='https://github.com/GallupGovt/multivac/blob/master/notebooks/scrape'>Jupyter Notebooks</a>.

The results are saved as a combined JSON file. Each article ID is a top-level key, with a Metadata key containing various metadata keys and values and a Text key containing the plain body text. This file then serves as the intermediate “source” datastore for subsequent analysis and processing, while the original raw documents are retained for reference.

### <a name='parsing'>Processing and Parsing</a>

Once the source data has been acquired and stored, some basic pre-processing and semantic metadata generation is required. MULTIVAC first calculates a basic term-frequency/inverse-document-frequency (TF-IDF) matrix for all the words in the vocabulary of the corpus of articles and saves this in the processed files directory as a sparse matrix for future analyses. 

MULTIVAC also trains a 300-dimensional domain-adapted Global Vectors (GloVe) word-embeddings model on the corpus and saves this file in the same folder. GloVe embeddings derive multi-dimensional vector spaces describing word associations based on calculations of word co-occurrences in a large corpus.<sup>[1](#1)</sup> We leverage a pre-trained 300-dimensional GloVe model incorporating 2 million terms found in the Common Crawl corpus, a collection of over 2 billion webpages scraped monthly.<sup>[2](#2)</sup> This model represents a best-in-class embedding model for generic English language text. However, given the specific and highly technical domain we are attempting to understand and model, much domain-specific semantic knowledge – not to mention domain-specific vocabulary – are not accounted for in this generic model. MULTIVAC augments this model by training a domain-specific model on our corpus, and combining embeddings using a nonlinear Canonical Correctional Analysis (CCA).<sup>[3](#3)</sup> This method aligns every pair of vector word representations (one domain specific, one generic) by finding linear combinations of the variables that maximally correlate. This alignment occurs on words that exist in the generic model vocabulary, but for terms that are entirely domain-specific the vector representations are passed as is, resulting in a domain-adapted model that encompasses all relevant terms and combines semantic meaning from both the domain and wider global context. 

MULTIVAC then iterates through all the texts in its corpus and parses all sentences and mathematical formulas into Universal Dependency trees using `spaCy` parsing for natural language sentences and sympy and ast for the formulas. Formulas are identified by LaTeX formatting codes for inline (“`\( \), $ $ or \begin{math} \end{math}`”) or display (“`\[ \], $$ $$, \begin{displaymath} \end{displaymath}` or `\begin{equation} \end{equation}`”) equation representations or `<math>` XML tags, depending on the raw format of the text being parsed. These are extracted separately from the natural language text given the fundamental differences in grammar and token relationships embedded in these sequences, though the locations of these sequences within the larger document are recorded to ensure that expressions can be mapped back to contextual information. The LaTeX representations are converted into a `symPy` representation that enables deconstructing expressions into a nested tree structure that contains a series of functions and arguments. For example, the expression `2x + (x*y)` would be expressed as `Add(Pow(Number(2), Entity('x')), Mul(Entity('x'), Entity('y')))` where `Add()`, `Pow()` and `Mul()` are functions; and `Number(2)` and `Symbol(‘x’)` are arguments. This representation aligns directly with the linked and nested universal dependencies for natural language text, and introduces “entity” as an additional POS tag for integration of these terms into our models’ vocabulary and token banks (“Number” is already a valid POS tag). 

A recursive tree traversal function will iterate over this tree structure to extract functions and arguments at each level of the nested expression. The traversal function will first start at the top level, identify arguments and recursively call itself for each argument until reaching a leaf node.

The outputs of this translation process are three sets of files: Dependency files, Morphology files, and Input files. Each file represents a parse of one article and is formatted in blocks, with one block for each sentence in the article. “Input” files record original word or punctuation as well as the part of speech (POS) tag, while “Morph” files record the token lemma, and each line contains a separate token. “Dep” files record the Stanford Universal Dependency relationships between pairs of words as well as the indices of the component words in the sentence. This process is illustrated in a <a href='https://github.com/GallupGovt/multivac/blob/master/notebooks/parse/Parsing.ipynb'>Jupyter Notebook</a>.

<table>
<tr>
<th>Inputs</th><th>Morphologies</th><th>Dependencies</th>
</tr>
<tr>
    <td nowrap>Glucocorticoid_NN <br> resistance_NN <br> in_IN <br> the_DT <br> squirrel_NN <br> monkey_NN <br> is_VBZ <br> associated_VBN <br> with_IN <br> overexpression_NN <br> of_IN <br> the_DT <br> immunophilin_NN <br> FKBP51_NN <br> ._.</td>
    <td> glucocorticoid <br> resistance <br> in <br> the <br> squirrel <br> monkey <br> be <br> associate <br> with <br> overexpression <br> of <br> the <br> immunophilin <br> fkbp51 <br> . </td>
    <td> nn(resistance-2, Glucocorticoid-1) <br> nsubjpass(associated-8, resistance-2) <br> det(monkey-6, the-4) <br> nn(monkey-6, squirrel-5) <br> prep_in(resistance-2, monkey-6) <br> auxpass(associated-8, is-7) <br> prep_with(associated-8, overexpression-10) <br> det(FKBP51-14, the-12) <br> nn(FKBP51-14, immunophilin-13) <br> prep_of(overexpression-10, FKBP51-14) </td>
</tr>
</table>

These files are organized as such to facilitate insertion into the unsupervised semantic parsing algorithm that induces a MLN knowledge base from the source data. This algorithm is codified in a 2009 paper by University of Washington researchers Dr. Hoifung Poon and Dr. Pedro Domingos entitled “Unsupervised Semantic Parsing,” and the original mechanism was created in Java.<sup>[4](#4)</sup> Gallup has refactored this software into Python for use in MULTIVAC both to ensure compatibility and maintainability of the code but also to introduce improvements and optimizations, leveraging advances in natural language processing and data management in the intervening years. This refactored system is named <a href='https://github.com/GallupGovt/multivac/tree/mln/pymln'>`pymln`</a>.

### End Notes
- <sup><a name='1'>1</a></sup> Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. “GloVe: Global Vectors for Word Representation.” Full text available at https://nlp.stanford.edu/pubs/glove.pdf <br>
- <sup><a name='2'>2</a></sup> See http://commoncrawl.org/connect/blog/ for up to date statistics on the corpus. As of this report the total is now 3.1 billion pages, though this has varied over time since project inception, and not simply increased monotonically. When the pre-trained GloVe model was created the corpus was closer to 2 billion pages in size. <br>
- <sup><a name='3'>3</a></sup> Prathusha K Sarma, YIngyu Liang, William A Sethares, “Domain Adapted Word Embeddings for Improved Sentiment Classification,” Submitted on 11 May 2018. arXiv:1805.04576 [cs.CL] Full text available at: https://arxiv.org/pdf/1805.04576 <br>
- <sup><a name='4'>4</a></sup> Hoifung Poon and Pedro Domingos. “Unsupervised Semantic Parsing.” In Proceedings of the 2009 Conference on Empirical Methods in Natural Language Processing, pages 1–10, Singapore, 2009. ACL. Full text available at: http://alchemy.cs.washington.edu/usp/poon09.pdf <br>


For more information please contact Principal Investigator, Benjamin Ryan (ben_ryan@gallup.com).

---
## Acknowledgements
This work is supported by the Defense Advanced Research Projects Agency (DARPA) under Agreement No. HR00111990008.

------


## Extraction Tool - GROBID:


### Installation:

https://grobid.readthedocs.io/en/latest/Install-Grobid/


#### GROBID Client:

Credit to: https://github.com/kermitt2/grobid-client-python


### Usage:

```bash
$ python extract_text.py -h

usage: extract_text.py [-h] [--input INPUT] [--output OUTPUT]
                       [--config CONFIG] [--n N] [--generateIDs]
                       [--consolidate_header] [--consolidate_citations]
                       [--force] [--teiCoordinates]
                       service

Client for GROBID services

positional arguments:
  service               one of [processFulltextDocument,
                        processHeaderDocument, processReferences]

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT         path to the directory containing PDF to process
  --output OUTPUT       path to the directory where to put the results
                        (optional)
  --config CONFIG       path to the config file, default is ./config.json
  --n N                 concurrency for service usage
  --generateIDs         generate random xml:id to textual XML elements of the
                        result files
  --consolidate_header  call GROBID with consolidation of the metadata
                        extracted from the header
  --consolidate_citations
                        call GROBID with consolidation of the extracted
                        bibliographical references
  --force               force re-processing pdf input files when tei output
                        files already exist
  --teiCoordinates      add the original PDF coordinates (bounding boxes) to
                        the extracted elements
```

```bash
## typical usage
python extract_text.py --input example-dir-with-pdfs/ --output example-dir-to-dump/ --consolidate_header --consolidate_citations processFulltextDocument

```

## Cleaning:

```bash

$ python clean_text.py -h

usage: clean_text.py [-h] --indir INDIR --outdir OUTDIR

Parser for XMLized scholarly publications.

optional arguments:
  -h, --help       show this help message and exit
  --indir INDIR    Path to the directory containing XMLs to process.
  --outdir OUTDIR  Path to output directory for processed files.
```

```bash
## typical usage
python clean_text.py --indir dir-to-tei-xmls/ --outdir dir-out/
```

