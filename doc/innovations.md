# Key MULTIVAC Innovations
### Knowledge Graph Prediction
Knowledge graphs are defined as a structured representation of real-world facts in the form of entities and relations, used extensively in applications such as question-answering, information retrieval, recommender systems, and natural language processing. Any particular knowledge graph contains only a small subset of real-world information, even within a given target domain. Manually adding new information to knowledge graphs is costly and inefficient, and thus automated knowledge graph completion remains an area of continuing research.

State of the art approaches, however, are limited to identifying new links or triples from an existing vocabulary of entities or relations. Adding new, previously unobserved nodes to a knowledge graph is beyond their scope. Our knowledge graph completion approach is both simpler and more ambitious than the more standard link prediction efforts common today. Using Kronecker graph models, Gallup intends to predict unobserved portions of the knowledge graph and thereby identify RDF triples that are the most important for bridging between the observed and unobserved portions of the graph. 

Kronecker graph models generate synthetic graphs that exhibit these and other characteristics of real-world graphs. These models begin with an initiator graph and then recursively create larger and larger graphs by taking the “Kronecker product” of the initiator and the evolving graphs’ adjacency matrices. Implementing this deterministically produces precise fractal patterns with awkward “staircase” effects in degree distribution and other properties, so instead Kronecker graphs are typically generated stochastically from a matrix of probabilities.

We adapt this framework to knowledge graph completion by treating our knowledge graph as a simple un-directed graph with head and tail entities as un-labeled nodes and relations as un-labeled` edges between them, and predict a “completed” knowledge graph using Kronecker models of the observed one. We employ an algorithm called Kronecker Expectation Maximization (KronEM) on the masked graph to predict what the original complete graph would look like. KronEM uses Expectation Maximization (EM) to fit a model of graph structure using the observed portion, and then estimating the missing portion using the model. Just as EM helps reveal latent variables in Bayesian models, here it helps reveal hidden portions of our knowledge graph. 

This functionality is demonstrated in Jupyter notebooks accessible at:
-   https://github.com/GallupGovt/multivac/blob/master/doc/notebooks/key_triples_walkthrough.ipynb
-   https://github.com/GallupGovt/multivac/blob/master/doc/notebooks/kg_predict_walkthrough.ipynb 

### A Tree-Based Generative Adversarial Network for English Queries
GAN architectures are trained dialectically, first training the discriminator on the existing query library, then training the generator against a static discriminator. The discriminator is then trained again, accounting for examples on which it failed, and so on. 

While a great deal of work has been done on image-generation GANs, sequence generation GANs have lagged in performance. In the influential paper introducing SeqGAN, researchers tried modeling the data generator as a stochastic policy in reinforcement learning. This allowed the system to bypass the token/sequence gradient loss problem by directly performing gradient policy update. The reinforcement learning reward signal from the GAN discriminator scoring complete sequences is then passed back to the generator using Monte Carlo search.

Subsequent attempts at sequence generation have generally adopted this basic loss function approach, adding features to encourage more accurate longer sequences through “leaking” information from the discriminator to the generator, or using variational autoencoders to encourage greater variety in the generated sequences. However, sequences generated from all of these approaches still suffer from significant quality issues in terms of syntactic coherence.

Instead, QueryGAN builds off recent work in graph neural networks and tree-based LSTM implementations. QueryGAN’s generator employs a long short-term memory recurrent neural network system (LSTM) to encode input sequences of semantic components. These encodings are then decoded into sequences of actions that build parse trees, with each node either applying a rule or generating a token to produce parse trees which can then be translated to a valid sequence of a given grammar. The discriminator employs a standard sequence-classification convolutional neural network architecture to delineate generated trees from real parse trees. 

Crucially, the generator also employs an embedded grammar model to constrain the output tree-sequences to obey a set of grammatical production rules. To generat those rules, MULTIVAC employs Stanford’s NLP engine once again to perform a constituency parse on each query in the training library and represent this parse in an abstract syntax tree structure. The system then traverses the tree in a depth-first, left-to-right sequence defining each parent-child pair as a valid production rule and builds up a full set of unique production rules from the corpus of queries. 

To achieve adversarial training via reinforcement learning, QueryGAN begins with a batch of sample tree sequences, and then for each step st ∈ S up to the maximum time step of the longest sequence, regenerates another batch of sequences beginning from that step. Each of these sequences are scored by the discriminator and the scores are averaged for each step and then aggregated via roll-up into a scalar reward value. 

QueryGAN generates new sequences based on seeds sequences of “semantic components.” The intuition behind this structure is that the resulting system will take in an RDF-triple or set of RDF-triples and generate a question related to that content. In execution, QueryGAN selects random RDF-triples from the MULTIVAC knowledge graph as its sequence-generation inputs.

Training functionality for QueryGAN is demonstrated in a Jupyter Notebook accessible at: 
- https://github.com/GallupGovt/multivac/blob/master/doc/notebooks/gan_training_illustration.ipynb

### Domain-Adapted GloVe Model
MULTIVAC also trains a 300-dimensional domain-adapted Global Vectors (GloVe) word-embeddings model on the corpus and saves this file in the same folder. GloVe embeddings derive multi-dimensional vector spaces describing word associations based on calculations of word co-occurrences over a large corpus. 

MULTIVAC begins with a pre-trained 300-dimensional GloVe model incorporating 2 million terms found in the Common Crawl corpus, a collection of over 2 billion webpages scraped monthly.  This model represents a best-in-class embedding model for generic English language text. However, given the specific and highly technical domain we are attempting to understand and model, much domain-specific semantic knowledge – not to mention domain-specific vocabulary – are not accounted for in this generic model. MULTIVAC augments this model by training a domain-specific model on our corpus, and combining embeddings using Canonical Correctional Analysis (CCA) on the intersection of tokens between the two models.  The vectors for each token of the domain adapted GloVe embedding model are derived from a weighted average of the canonical vectors (N = 100) from the CCA analysis.

This alignment occurs on words that exist in both the domain-specific and generic model vocabularies, but for terms that are entirely domain-specific the vector representations are projected into the 100-dimensional canonical vector space from the CCA analysis via matrix multiplication and appended to the domain-adapted embedding vectors. The resulting domain-adapted model encompasses all terms in our corpus and combines semantic meaning from both the domain and wider global context. 

This functionality is demonstrated in Jupyter notebooks accessible at:
- https://github.com/GallupGovt/multivac/blob/master/doc/notebooks/Domain_Adapted_Glove.ipynb 

### Domain Agnostic Query Derivation and Extraction
In addition to expert submitted examples, MULTIVAC derives queries from source texts using a modified version of deep learning question-generation system called QG-Net (https://github.com/moonlightlane/QG-Net).  QG-Net is a recurrent neural network-based model that takes as inputs a "context" and an "answer" and outputs a question tailored to produce that answer. 

Term frequency-inverse document frequency scores (TF-IDF) are calculated to determine which terms/sentences in the documents are truly important and differentiating. The sentences with the largest total sum of their terms' TF-IDF scores are used to generate queries. Within these sentences, the terms or phrases (by default, the system calculates TF-IDF scores for n-grams up to three words) with the highest TF-IDF scores are tagged as "answers."

This approach differs from the method in the original QG-Net system, which depended on words listed in an index and named entity tags supplied by the Stanford NLP parser to select potential answers from context sentences. In research articles, as with the vast majority of texts which are not specifically published books, there is almost never an index or other lengthy annotated list of key words and phrases to guide this type of approach. Named entity recognition, on the other hand, can be limiting in terms of the types of information and relationships supplied for training question generators. TF-IDF scores, however, are format independent and can apply to any type of underlying semantic content, making them more versatile and powerful guides for programmatically generating queries. 

From these inputs, the question generator generates diverse question text word-by-word given all context word representations. For the alpha system, MULTIVAC generated 2,804 queries based on the source document research abstracts. MULTIVAC also extracts literal questions that researchers had written in the research articles themselves. During the alpha build, a total of 2,028 literal expert queries were extracted by parsing full texts into component sentences and simply selecting all sentences that ended in question marks.

To further enhance our query extraction capabilities, in our beta development phase we harnessed the power of TextRank – a graph-based ranking model used for identification of the most “important” sentences in a text.  After using TextRank to choose the most relevant sentences from our articles, we again use TF-IDF to find the most important word(s) in these key sentences to identify the “answers” and use the key sentences as the context to generate queries. A total of 2,999 additional queries were generated using this approach. 

This functionality is demonstrated in Jupyter notebooks accessible at:
- https://github.com/GallupGovt/multivac/blob/master/doc/notebooks/precooked_replication.ipynb

### Python Implementation of Markov Logic Network Knowledge Base
A Markov network is a set of random variables having a Markov property (where the conditional probability of future states is dependent solely on the present state) described by an undirected graph. In a MLN, the nodes of the network graph are atomic first-order formulas (atoms), and the edges are the logical connectives (here, dependencies) used to construct a larger formula. 

First-order logic (FOL), also known as first-order predicate calculus or first-order functional calculus, is a system in which each sentence, or statement, is broken down into a subject and a predicate. The predicate modifies or defines the properties of the subject. This system naturally mirrors the dependency tree parsing performed in the previous step.

Each formula is considered to be a clique (a subset of nodes in the graph such that every pair of nodes in the clique are connected), and the Markov blanket (the set of other nodes containing all the information necessary to determine the value of a given node) is the set of larger formulas in which a given atom appears. A “grounded atom” is an atomic formula with actual constants/values supplied to give the formula a “grounded” meaning. MLNs associate a weight with each formula, designated by the frequency with which that formula is “true” given its groundings in the available evidence (such as our corpus). Unlike in first-order logic knowledge bases, in a MLN when one clique or formula is violated (e.g., “Senators from Kansas are Republican”) the “world” described by that grounding is simply less probable, rather than impossible. 

The pymln subsystem reads in parse files generated from the initial source documents, detailing all the dependency parses and tokens therein, and compiles these into a network of nodes representing semantic tokens and edges representing their parent-child relationships. These nodes are assigned to initial semantic clusters which link them via their relationships and arguments to other nodes in the knowledge base. Each cluster maps these links via python dictionaries and sets encoding the various types of relationships, types of arguments, and specific Argument nodes in globally tracked indices.

The results of this process are a graphical model of nodes and edges governed by first-order logic formulas which embed all the statements, entities and relationships found in the source data. These first-order logic formulas are assigned weights according to the frequency of their occurrence in the corpus and placed in a Markov network structure to create a MLN. 

Further information and source code is accessible at: 
- https://github.com/GallupGovt/multivac/tree/master/pymln

### Semantic Integration of English Grammar and Scientific Formulae
MULTIVAC extends existing MLN ontology concepts by integrating the parsed model formulas along with the actual text, mapping both into the same shared ontological space. Thus, the dependencies and relationships in the models, as represented in the mathematical formulas associated with them, are also represented in the MLN ontology and enriched by the resulting relationships with the organic contextual knowledge provided by the natural language text. 

The first iteration of MULTIVAC text parsing relied on Stanford’s language processing engine (stanfordnlp) to construct dependency trees, tag parts of speech and lemmatize tokens. In the initial system, each sentence is processed individually to identify the dependency structure of its tokens. When LaTeX notation occurs in text the notation block is extracted and a "dummy" token is substituted, allowing the NLP dependency parsing to interpret the sentence as a proper English language construct. This is especially important for in-line LaTeX notations, which otherwise render many of the most important sentences in an article un-parseable.
 
The LaTeX equation itself is separately parsed and then re-inserted into the sentence, with the root of the LaTeX tree taking the place of the dummy token in the dependency structure. The LaTeX representations are parsed by converting them first into a sympy representation that enables deconstructing expressions into a nested tree structure that contains a series of functions and arguments. For example, the expression 2x + (x*y) would be expressed as: 

```Add(Pow(Number(2), Entity('x')), Mul(Entity('x'), Entity('y')))```

where Add(), Pow() and Mul() are functions; and Number(2) and Symbol(‘x’) are arguments. MULTIVAC transforms these nested parenthetical representations into a collapsed dependencies format and inserts the entire chain back into the source sentence, updating token indices as appropriate. The individual relationship and entity tokens from these equations are also expanded out in string representation and replace the LaTeX notation in the original text.

This functionality is demonstrated in Jupyter notebooks accessible at: 
- https://github.com/GallupGovt/multivac/blob/master/doc/notebooks/Parsing.ipynb 

