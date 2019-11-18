## Knowledge Graph Developments
Coming out of Phase I and our Lessons Learned reviews this past summer, it was clear that reducing the complexities of our knowledge representation system could pay significant dividends for performance and inference, as well as streamline work toward developing the GAN query generation system.

To that end, we tested several new techniques and approaches to ingesting and representing our domain knowledge to alleviate some of the complexities of our Phase I approach. Specifically, we addressed three key pain points in our existing process:
1) The need for multiple NLP parsing engines at different points in our process.
2) The computational hurdles in fully compiling and ingesting all of our domain knowledge
into a Markov Logic Network structure
3) The algorithmic complexity in mapping complex scientific queries to that structure
effectively.

For the first point, we first standardized our Stanford NLP implementations on the more mature Java-based system, using the official stanford-corenlp python wrapper module, eliminating usage of the newer all-python stanfordnlp module. Second, we eliminated the spaCy NLP engine in favor of Stanford’s NLP engine to consolidate on one tool throughout our process.

Originally, we had kept spaCy in our pipeline because it did allow for custom vocabulary. Our source texts produced two problems for our dependency parsing – some terms were hyphenated and split between rows, and our equation parsing system introduced “dummy” word tokens in otherwise standard English sentences. With spaCy, we were able to easily add equation “dummy” tokens and other custom terms to the engine’s dictionary and these were then processed as is, as well as identify and account for these hyphenated words.

However, this was a cumbersome solution and while our team developed some very innovative solutions toward addressing complicated and subtle errors in parsing, other performers in this effort can attest that this is a category of problem worthy of an entire project by itself. Rather than spend additional resources on parsing and managing data ingested with an additional suite of NLP tools, we opted to leverage an existing document parsing package called GROBID (GeneRation Of BIbliographic Data) that removed many of the extraneous complexities and burdens of managing the source articles themselves.

GROBID is a machine learning library for extracting, parsing and re-structuring raw documents such as PDF into structured XML/TEI encoded documents with a particular focus on technical and scientific publications. Using this system, we intelligently extract text from scholarly articles and identify metadata (authors, references, formulas, affiliations, patents, dates, etc.) and clean natural language text free of references, nested graph or tabular data, formulas, etc. GROBID is also a Java-based system, so much like Stanford’s NLP engine we start an instance of the server locally and use a custom-built RESTful API to call the GROBID service.

The benefits of this move are multiple. First, this solution streamlines ingestion of our source documents by natively cleaning and properly combining hyphenated terms and extracting equations from text even if not in LaTeX encoding tags, relieving us from much of that burden. Second, the additional cleaning and curation of the citations, tabular and other data in these documents removes several sources of pernicious noise in our knowledge graph system, reducing the computational requirements in inducing that component.

### Interim Knowledge Graph Structure
While reducing noise inputs to our knowledge graph structure is essential and a benefit in and of itself, it has not been sufficient to resolve the computational difficulties in learning a MLN representation of our domain knowledge, given the complexity of the relationships we are modeling. While we continue to work on implementing more efficient algorithms to surmount that challenge, our work with testing queries and query generation calls for an interim solution to stand in for the full MLN knowledge base in the meantime.

For the interim knowledge graph concept, we employ a semantic knowledge graph approach of extracting subject-predicate-object tuples from the documents in our corpus and representing them in the form of a graph, with subject and object entities as nodes and predicates as edges between them. The success of such knowledge graph representations depends in large part on the accuracy of the subject-predicate-object tuple extraction algorithm; incoherent or inaccurate tuples result in predictably poor performance. Given the complex grammatical nature of our source documents and the queries we hope to map to this knowledge graph, this represents a significant risk to the approach. Thus, we employ a flexible method of tuple extraction, opting either for traditional RDF- triple extraction using the OpenIE (Stanford Open Information Extraction) annotator included in the Stanford NLP suite,3 or a custom semantic component extraction algorithm to supply our subject-predicate-object tuples.

This custom algorithm extracts noun and verb phrases based on the semantic universal dependency tree structures identified by the Stanford dependency tree parser tool. A first pass identifies all nouns and adjective modifiers (and verb and adverb modifiers) while a second pass recursively combines these with nouns and verbs further up the tree structure. The resulting set of subject/object and predicate clusters are assigned to their respective roles in a quasi-RDF-triple entity, based on their roles and relations as defined by the dependency labels of their components. For instance, a noun/adjective cluster where one noun was marked as having the dependency relation “nsubj” would be assigned to the “subject” role, while one with an “dobj” dependency would be assigned to the “object” role. In grammatically complex sentences such as those often found in academic writing there may be multiple such sets, and this algorithm handles those cases natively by assigning and tracking related subject/predicate/object triples without overlapping or repeating them.

Once these triples are extracted from our corpus, they are clustered together using fast agglomerative clustering based on cosine similarity measures comparing the domain adapted GloVe embedding vectors (averaged for each subject/predicate/object cluster). This approach works toward the same objective as the iterative inference and clustering algorithm in our MLN knowledge graph but instead relies on more efficient vector and matrix algebra to do the associative work. By simplifying the induction of our knowledge graph while retaining as much rich semantic relational information as possible we hope to at least temporarily avoid the computational burden of inference and learning in an MLN on the way toward testing and developing our core query generation algorithms.

### Query Mapping
When considering queries extracted or generated from our source documents, we expect to find matching answers in our knowledge graph. However, when we map queries generated from our end state system (or novel queries submitted by human researchers) we expect frequent partial matches or perhaps no match at all to the existing knowledge base. Our query system needs to handle both situations.

Our custom triple extraction approach described above becomes even more important when we are mapping real world queries to our knowledge base. While the more standard OpenIE RDF- triple extraction algorithm often struggles to find such patterns within question sentences (which by their nature tend to both invert syntactic patterns of standard declarative sentences and often replace key components with question-word tokens), this algorithm handles such cases more seamlessly and inherently handles “incomplete” triples.

For any particular query we extract our triples and then search the knowledge graph for matches using cosine similarity measures once again. For each proposed triple, we can determine whether the proposition represented by the triple is supported by the knowledge graph (more colloquially, “is true”) if it is in fact contained within it. For those that are not strictly contained within it, however, we can use the knowledge graph to infer a quasi-likelihood of the proposition being true by leveraging graph embedding models. We use OpenKE as a toolkit that implements various graph embedding models in a unified framework. The primary difference between the various models is the scoring function used in predicting any particular triple. The scoring functions generally return lower values if a triple proposition is more likely to be supported based on other information in, and larger structure of, the graph.

As mentioned above, many queries will render “partial” triples upon extraction, with either subject, predicate, or object missing or in doubt. In this case, the system finds matches for the remaining triple components and then generates predictions and prediction scores for the missing component based on them. Thus, the return for any query is an “accuracy” score for a set of complete triples, whether that is a set of one – the originally supplied complete triple – or the proposed completions based on knowledge graph embedding driven inferences.

You can follow a walk-through <a href="https://github.com/GallupGovt/multivac/blob/master/kg_query_walkthrough.ipynb">here</a> for querying the new semantic knowledge graph. 

## Updated Query Generation
As developed in Phase I, MULTIVAC derives queries from source texts using a modified version of deep learning question-generation system called QG-Net.1 QG-Net is a recurrent neural network-based model that takes as inputs a "context" and an "answer" and outputs a question tailored to produce that answer. MULTIVAC uses the most important sentences in the abstract as the "contexts" and uses the most important words and phrases in the sentences as the "answers."

In order to make this system less dependent on thorough annotation of source material and less bound to questions about specific entities, MULTIVAC uses term frequency-inverse document frequency (TF-IDF) scores to determine which sentences – and which terms in those sentences – in the documents are truly important and differentiating. The sentences with the largest total sum of their terms' TF-IDF scores are used to generate queries. Within these sentences, the terms or phrases (by default, the system calculates TF-IDF scores for n-grams up to three words) with the highest TF-IDF scores are tagged as "answers."

Additionally, QG-Net subsystem uses GloVe embeddings to represent words as vectors; MULTIVAC augments the standard pre-trained embeddings with a domain-specific model trained on our corpus, and combines the embeddings using Canonical Correlational Analysis (CCA) on the intersection of tokens between the two models. The resulting domain-adapted model encompasses all terms in our corpus and combines semantic meaning from both the domain and wider global context.

From the initial results of MULTIVAC’s Phase I query generation/extraction, most of the queries extracted are relatively straightforward and “factual” in nature. This is not surprising given the motivation of the original QG-Net system, which was to auto-generate quiz questions for textbooks. This does present somewhat of a gap for our purposes between these results and the ideal range of depth, complexity and potential abstraction of scientific research queries we are ultimately interested in producing and modeling.

To expand our query library and attempt to improve the average query quality, we extracted literal questions that researchers had written in the research articles themselves. A total of 2,028 literal expert queries were extracted by parsing full texts into component sentences and simply selecting all sentences that ended in question marks.

To further enhance our query extraction capabilities, in Phase II we harnessed the power of TextRank – a graph-based ranking model used for identification of the most “important” sentences in a text. TextRank is used to build extractive summaries which is likely to identify key sentences in the entire article that may include model parameters or equations. This kind of graph-based ranking algorithm computes the importance of a particular vertex in a graph by recursively calculating the “votes” it gets from other vertices by virtue of being linked from them. These votes are weighted by the importance of their associated vertices, recursively calculated until reaching a point of convergence. After using TextRank to choose the most relevant sentences from our articles, we again use TF-IDF to find the most important word(s) in these key sentences to identify the “answers” and use the key sentences as the context to generate queries. A total of 2,999 additional queries were generated using this approach.

These preliminary “bootstrap” reviews have both expanded our baseline query library as well as provided pointers toward more performative algorithms for extracting well-formed and relevant queries from our source data. Comparing the outputs from these different approaches for query extraction, we can see how each differs in both the complexity (represented as word/token counts per query in the table below) and coherence (represented as the grammatical error rate per word/token) of the queries returned.

Table 1: MULTIVAC Query Set Statistics, by method of extraction
<table width=80%>
  <tr BGCOLOR="#89BC00">
    <th colspan=4>Token Counts*</th>
    <th colspan=3>Grammatical Error Rate**</th>
  </tr>
  <tr align="center" BGCOLOR="#e2efd6">
    <td></td>
    <td>Original</td>
    <td>TextRank</td>
    <td>Literal</td>
    <td>Original</td>
    <td>TextRank</td>
    <td>Literal</td>
  </tr>
  <tr>
    <td>count</td>
    <td>2804</td>
    <td>3000</td>
    <td>2028</td>
    <td>2804</td>
    <td>3000</td>
    <td>2028</td>
  </tr>
  <tr BGCOLOR="#e2efd6">
    <td>mean</td>
    <td>10.37</td>
    <td>10.63</td>
    <td>20.47</td>
    <td>0.13</td>
    <td>0.13</td>
    <td>0.08</td>
  </tr>
  <tr>
    <td>std</td>
    <td>2.95</td>
    <td>2.86</td>
    <td>52.01</td>
    <td>0.07</td>
    <td>0.07</td>
    <td>0.15</td>
  </tr>
  <tr BGCOLOR="#e2efd6">
    <td>min</td>
    <td>5</td>
    <td>5</td>
    <td>1</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
  </tr>
  <tr>
    <td>25%</td>
    <td>9</td>
    <td>9</td>
    <td>10</td>
    <td>0.09</td>
    <td>0.09</td>
    <td>0</td>
  </tr>
  <tr BGCOLOR="#e2efd6">
    <td>50%</td>
    <td>10</td>
    <td>10</td>
    <td>16</td>
    <td>0.1</td>
    <td>0.1</td>
    <td>0</td>
  </tr>
  <tr>
    <td>75%</td>
    <td>11</td>
    <td>12</td>
    <td>24</td>
    <td>0.14</td>
    <td>0.14</td>
    <td>0.11</td>
  </tr>
  <tr BGCOLOR="#e2efd6">
    <td>max</td>
    <td>90</td>
    <td>100</td>
    <td>2208</td>
    <td>1.17</td>
    <td>0.88</td>
    <td>2</td>
  </tr>
</table>
* Calculated as the number of words or tokens per query
** Calculated as the number of grammatical errors per token, per query

Using the TextRank approach does not substantially affect complexity or coherence in the queries returned compared of the original pure TF-IDF driven extraction, though this method makes more use of the complete semantic relationships in the text. However, compared to the literal queries pulled from the text, both the machine-driven methods return much less complex results (on the order of 10 words/tokens per query on average, compared with about 20 in the literal queries). They are also more likely to include grammatical errors, 60% more on average. 

While these results are not terribly surprising, they do point to a significant and ongoing problem with sequence generation approaches, namely that the complex and intertwined syntactical rules and semantic relationships in natural language text are still very difficult for even advanced deep learning systems to reliably reproduce. This concern drives our focus on tree based, syntax-aware GAN approaches for MULTIVAC.

## GAN System Development
As the central effort in Phase II, MULTIVAC will train a Generative Adversarial Network (GAN) to produce well-formed, novel expert queries without human intervention. While GAN modeling and development has historically been dominated by image generation efforts, in the past couple years there have been increasing numbers of attempts to develop GAN models for text generation. Development of GANs for sequence generation lagged behind in large part because the discrete outputs (words/tokens in a sequence) from the generative model make it difficult to pass the gradient update from the discriminative model to the generative model. The generative model wants to apply gradient loss to its outputs to improve, shifting the output incrementally in one direction or another. But where in continuous data such as images these incremental changes can still make a sort of sense, in the case of sequences of discrete tokens this approach is nonsensical. Similarly, gradient loss is typically only calculated for the entire generated sequence. 

In recent years, however, several approaches have been introduced to avoid or alleviate these problems in sequence generating GANs. In one of the first published attempts at generating English language sentences, instead of using the standard objective of GANs, researchers sought to match sequence feature distributions when training their generator and used various techniques to pre-train the model and handle discrete intermediate variables.

In the influential paper introducing SeqGAN, researchers tried modeling the data generator as a stochastic policy in reinforcement learning. This allowed the system to bypass the token/sequence gradient loss problem by directly performing gradient policy update. The reinforcement learning reward signal from the GAN discriminator scoring complete sequences is then passed back to the generator using Monte Carlo search. 

Subsequent attempts at sequence generation have generally adopted this basic loss function approach, adding features to encourage more accurate longer sequences through “leaking” information from the discriminator to the generator, or using variational autoencoders to encourage greater variety in the generated sequences. However, sequences generated from all of these approaches still suffer from significant quality issues in terms of syntactic coherence.

For MULTIVAC’s (and ASKE’s) goals, generating the kind of sequences we see above will not suffice. MULTIVAC’s output must be not only domain relevant but coherent, literate and complex if it is to be useful and interesting to advanced researchers. Our system will need to generate sequences in natural language in line with its corresponding grammar. 

### Next Generation Syntax-Aware GAN System

<img align="center" src="images/gan_design.png" alt="GAN Schematic">

Our approach builds off this recent work in graph neural networks and tree-based LSTM implementations. Building in knowledge about the dependency tree structures in our training data and desired outputs helps construct syntactically correct queries. Recent work in this area has produced promising results combining source texts with context-free grammar production rule sets to generate realistic SQL or Python programming code.

We begin with an architecture modeled after the recently developed TreeGAN system, employing a long short-term memory recurrent neural network system (LSTM) as our generator to produce parse trees which can then be translated to a valid English-language sequence in the given grammar. Our discriminator employs a tree-structured LSTM network to delineate generated trees from real parse trees. In the original TreeGAN system, grammar information is fed into the system by converting source Python and SQL code into their abstract syntax tree (AST) representations and then assembling a library of grammar production rules by induction. These rules then constrain the generator in the sequences it produces. 

We anticipated that the compilation of English grammar production rules would prove a complex and potentially laborious task. However, on further investigation this task was not terribly complex at all. The key factor in our favor here is that our grammar production rules do not have to be necessary and sufficient to reproduce all possible English sentences, or even all possible English questions. Rather, they have to be suitably comprehensive to produce the types of questions from which they were derived, which is no more or less than our objective.

To build our abridged English grammar system, we employ Stanford’s NLP engine once again to perform a constituency parse on each query in turn and represent this parse in an abstract syntax tree structure. The system then traverses the tree in a depth-first, left-to-right sequence defining each parent-child pair as a valid production rule and builds up a full set of unique production rules from the corpus of queries.

<table>
  <tr BGCOLOR=#89BC00>
    <th>Query</th>
    <th>Parse</th>
    <th>Production Rules</th>
  </tr>
  <tr>
    <td>What virus can lead to <br>certain cervical cancers?</td>
    <td>(SBARQ<br>
            &nbsp; &nbsp; (WHNP (WDT What) (NN virus))<br>
            &nbsp; &nbsp; (SQ (MD can)<br>
            &nbsp; &nbsp; &nbsp; &nbsp; (VP (VB lead)<br>
            &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; (PP (IN to)<br>
            &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; (NP (JJ certain) (JJ cervical) (NNS cancers)))))<br>
            &nbsp; &nbsp; (. ?)))</td>
    <td>(SBARQ) -> (WHNP), (SQ), (.)<br>
        (WHNP) -> (WDT), (NN)<br>
        (SQ) -> (MD), (VP)<br>
        (VP) -> (VB), (PP)<br>
        (PP) -> (IN), (NP)<br>
        (NP) -> (JJ), (JJ), (NNS)</td>
  </tr>
</table>

You can follow a walk-through of the QueryGAN training algorithm <a href="https://github.com/GallupGovt/multivac/blob/master/gan_training_illustration.ipynb">here</a>. 

### Knowledge Graph Embeddings for Directing Query Generation
Two other inputs are required for the generator portion of the system we are adapting, consisting of annotations (in the original TreeGAN, these were "docstrings" describing code functionality and/or outputs) and the actual target sequences. The system then performs a form of sequence-to-sequence translation, with the grammar production rules as constraints on the outputs.

Our system will seek to output English language scientific queries, so our training library of queries serves as the model for our target output sequences. For the left-hand side of this equation, we aim to supply the raw semantic compnents extracted from these queries - either as subject-object-predicate triples, or as . This sets up the system to relatively straight-forwardly build correct syntax around supplied bundles of semantic concepts in order to generate queries.

One persistent but unaddressed question behind the query generation task from the beginning has been how to constrain the solution space for generated queries to those that are both coherent and useful. But here, the practice of employing graph embeddings to infer missing portions of knowledge graphs comes to our rescue. If we treat any particular knowledge graph as an incomplete version of the “Platonic truth” graph that would exist in a state of perfect knowledge, we can attempt to recover those missing portions of the graph with a variety of techniques leveraging local and global graph structures. These “missing” network components can then form the complete or partial subject-predicate-object triples that feed into our query generation system, representing these automated inferences as well-formed natural language scientific questions.

While this system presupposes a knowledge graph constructed along the lines of our interim system and not a MLN, all of these can be adapted to use a MLN as the underlying knowledge graph as well. Whereas our interim system employs more conceptually simplistic subject- predicate-object triples, the MLN structure is fundamentally similar, with the exception of adding weights to the edges. This informs the process of learning and inference on the network, but not the basic mechanics of parsing queries and using the semantic components as inputs to be syntactically organized by our generator. And while most work on knowledge graph embedding has focused on more standard RDF-triple based graphs, recent work points to significant improvements on benchmark dataset performance in applying these techniques to MLN knowledge graphs.

You can follow a walk-through of knowledge-graph directed query generation <a href="https://github.com/GallupGovt/multivac/blob/master/pure_generation_walkthrough.ipynb">here</a>.

