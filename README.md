# MULTIVAC
Gallup’s Meta-model Unification Learned Through Inquiry Vectorization and Automated Comprehension (MULTIVAC) will consist of an expert query generator trained on a corpus of historical expert queries and tuned dialectically with the use of a Generative Adversarial Network (GAN) architecture.

![alt text](https://github.com/GallupGovt/multivac/blob/master/images/multivac_concept.png 'MULTIVAC Concept Graphic')

For more information please contact Ben Ryan (ben_ryan@gallup.com).

# Initial System Design
## Phase I - Formal Representation/Interpretation of Queries
### Data Acquisition
The MULTIVAC prototype will operate on an Amazon Web Services (AWS) commercial EC2 instance with GPU capabilities and will scale as MULTIVAC’s complexity increases. MULTIVAC’s initial source data will comprise 2,000 articles with total data storage needs (including metadata) of approximately 500GB. We build our source dataset by utilizing a variety of means, including APIs, scraping, and bulk download options for tapping into epidemiological research articles from online sources such as arXiv’s Quantitative Biology repository. The use of arXiv is purposeful; the research is current and constantly updating; moreover, articles tend to be technical in nature, allowing for plentiful source material to train MULTIVAC.

### Parsing Article Data
Once the data has been acquired and stored, we perform some basic pre-processing and meta-data generation procedures. We calculate a term-frequency/inverse-document-frequency matrix for all the words in the vocabulary of the corpus of articles. We also train a domain-adapted GloVe word-embeddings model on the corpus.<sup>[1](#1)</sup>

We then read in and parse all sentences and formulas in the texts into Stanford Universal Dependency trees using both the Stanford Parser toolset for natural language sentences and custom rule-based systems for formulas. Dependency trees represent grammatical relations between words in a sentence using triples – name of the relation, governor, and dependent. An example parse of a natural language sentence is reproduced below.<sup>[2](#2)</sup> This parsing system will be written in Python leveraging Stanford’s official `corenlp` module, which acts as a wrapper to a Java backend implementation.<sup>[3](#3)</sup>

| “Bills on ports and immigration were<br> submitted by Senator Brownback,<br> Republican of Kansas.” | ![alt text](https://github.com/GallupGovt/multivac/blob/master/images/stanford_dependecies.png "Stanford sentence dependencies tree example") |
|:--------------:|:----:|

Formulas are identified by LaTeX formatting codes for inline (“`\( \), $ $` or `\begin{math} \end{math}`”) or display (“`\[ \], $$ $$, \begin{displaymath} \end{displaymath}` or `\begin{equation} \end{equation}`”) equation representations or `<math>` XML tags, depending on the raw format of the text being parsed. For PDF files, we leverage recent work in formula identification and extraction.<sup>[4](#4)</sup>

| <img src="https://github.com/GallupGovt/multivac/blob/master/images/formula.png" alt="Example formula" width="300"> | ![alt text](https://github.com/GallupGovt/multivac/blob/master/images/formula_dependencies.png "Formula dependencies tree example") |
|:--------------:|:----:|

These parse trees are then translated into first-order logic formulas. First-order logic, also known as first-order predicate calculus or first-order functional calculus, is a system in which each sentence, or statement, is broken down into a subject and a predicate. The predicate modifies or defines the properties of the subject. This system naturally mirrors the dependency tree parsing we perform in the previous step. These first-order logic formulas are then assigned weights according to the frequency of their occurrence in the corpus and placed in a Markov network structure to create a Markov Logic Network (MLN). Conversion to an MLN and subsequent ontology creation and manipulation will be written in Python, leveraging the `pracmln` module.<sup>[5](#5)</sup>

<table>
<tr>
<th>Dependencies</th>
<th>First Order Logic Formulas</th>
</tr>
<tr>
	<td nowrap>nsubjpass(submitted, Bills) <br> auxpass(submitted, were) <br> agent(submitted, Brownback) <br> nn(Brownback, Senator) <br> appos(Brownback, Republican) <br> prep_of(Republican, Kansas) <br> prep_on(Bills, ports) <br> conj_and(ports, immigration) <br> prep_on(Bills, immigration)</td>
	<td>&#955;x1&#955;x2.submitted(n1)/\agent(n1,x1)/\nsubjpass(n1,x2) <br>  &#955;x1&#955;x3.hasProperty(n2)/\nn(n2,x1)/\property(n2,x3) <br> &#955;x1&#955;x4.isA(n3)/\appos(n3,x1)/\category(n3,x4) <br> &#955;x1&#955;x5.belongsTo(n4)/\prep_of(n4,x1)/\category(n4,x5) <br> &#955;x2&#955;x6.hasProperty(n5)/\nn(n5,x2)/\property(n5,x6) <br> &#955;x2&#955;x7.hasProperty(n5)/\nn(n5,x2)/\property(n5,x7) <br><br> (x1=Brownback; x2=Bills; x3=Senator; x4=Republican; x5=Kansas; x6=ports; x7=immigration)</td>
</tr>
</table>

### Markov Logic Networks
A Markov network is a set of random variables having a Markov property (where the conditional probability of future states are dependent solely on the present state) described by an undirected graph. In a Markov Logic Network, the nodes of the network graph are atomic first-order formulas (atoms), and the edges are the logical connectives (here, dependencies) used to construct a larger formula. Each formula is considered to be a clique (a subset of nodes in the graph such that every pair of nodes in the clique are connected), and the Markov blanket (the set of other nodes containing all the information necessary to determine the value of a given node) is the set of larger formulas in which a given atom appears. A “grounded atom” is an atomic formula with actual constants/values supplied to give the formula a “grounded” meaning. MLNs associate a weight with each formula, designated by the frequency with which that formula is “true” given its groundings in the available evidence (such as our corpus). Unlike in first-order logic knowledge bases, in an MLN when one clique or formula is violated (e.g., “Senators from Kansas are Republican”) the “world” described by that grounding is simply less probable, rather than impossible.<sup>[6](#6)</sup>

### Ontology Construction
MULTIVAC translates all sentences and formulas in our corpus into first-order logic forms, and integrates  them into a unified ontology. This integration is done by finding the parse that maximizes the a posteriori distributions of the network. To construct the domain’s model ontology, we cluster together semantically interchangeable formulas into more generalized versions, identified as formulas which can be combined to improve the log-likelihood of observing the given set of formulas, determined by the sum of the weights. For example, in the above sentence the formula for “bills were submitted by Brownback” (passive voice):

<p align='center'> <i> &#955;x1&#955;x2.submitted(n1)/\agent(n1,Brownback)/\nsubjpass(n1,bills) </i> </p>

is semantically the same as the formula for “Brownback submitted bills” (active voice): 

<p align='center'> <i> &#955;x1&#955;x2.submitted(n1)/\nsubj(n1,Brownback)/\dobj(n1,bills) </i> </p>

and these formulas would then be merged into one formula cluster for the concept “&#955;x1&#955;x2.submitted” that abstracts away the active/passive voice distinction. 

MULTIVAC extends the Alchemy MLN ontology concept<sup>[7](#7)</sup> by including the parsed model formulas along with the actual text, mapping both into a shared ontological space. An important element of our integration is the co-mingling of formula parameters in the text of the containing article, where authors explain their models and approach. This allows our domain-adapted word embedding model to assign word vectors to model parameters that relate them to other natural language words in the corpus, and for the ontology generation to associate these model parameters not only with formula-specific MLN nodes but also with nodes generated from the natural language text around those formulas, linking the two portions of the MLN network together. 

The result is a domain ontology represented as a Markov Logic Network grounded on the models found in our domain’s scientific literature. The MLN represents a meta-model ontology architecture that can be queried not just for facts but for cause and effect inference, counter-factual explorations, and uncertainty quantification across the domain.

### Confirm Execution of Queries as Models
MULTIVAC will extract literal expert queries, identified by parsing full texts into component sentences and selecting sentences that end in question marks, as well as deriving expert queries from abstracts using a modified version of deep learning query-generation system QG-Net.<sup>[8](#8)</sup> QG-Net is a recurrent neural network (RNN)-based model that takes as inputs a “context” (typically a limited amount of natural language text containing relevant information) and an “answer” (a specific fact or figure found within the context) and outputs a question tailored to produce that answer. Our adaptation of QG-Net will be written in Python, in line with the current implementation and most of the rest of the MULTIVAC system.

We will provide epidemiology abstracts to the QG-Net context reader as the “context” data, and selected parts of the abstracts as the “answer” data. These parts will be selected based on two heuristics:
* Words that occur in the same context (sentence or paragraph) as model parameters and descriptions in the main body of the article or that occur in the article title signify phrases that can serve as feasible answers.
* Words with a high TF-IDF score for a particular article signify phrases that can serve as feasible answers.

Given context and answer inputs, QG-Net generates different questions that focus on the relevant contextual information that different answers provide. More specifically, it uses the GloVe algorithm to represent words as vectors coupled with speech tag POS, name entity NER, and word case CAS from the Stanford natural language processing toolkit as inputs in the context reader to generate diverse context word representations. Finally, a question generator generates the question text word-by-word given all context word representations.

We modify QG-Net for use in MULTIVAC to leverage the first-order logic information in our MLN representation. The MLN-ontology provides a rich meta-model architecture and a source of parameter value distributions to model query outcomes against. Rather than generating “factual” questions (answering “what,” “when,” “where,” or “who” type queries), our adaptation of the QG-Net system will generate “whether,” “how” and “how much” type queries that interrogate the logical structure of the answer phrases and statements. In addition to the word embedding and lexical metadata features listed above, the first-order logic formulas will be fed in as features to train the RNN.

We will apply automated sensitivity analysis by building a gradient of result sets by repeating our query modeling over a range of parameter values and applying the method of finite differences. This sensitivity analysis serves two purposes. The first is uncertainty propagation. If the inputs have a given level of uncertainty, it is crucial to understand how this manifests in uncertainty in the outcomes. The second purpose is to search for regions of the parameter space where modeled outcomes are subject to drastic changes. Results will include estimates of these potential tipping points and – in hypothetical real-time monitoring configurations – threshold alerts and confidence-banded forecasts.

## Phase II - Human Review and Machine Generation of Queries
### Human Expert Query Execution and System Reviews
Gallup will recruit a cohort of six to eight top experts in epidemiological modeling via our established connections with leading international organizations, NGOs and government agencies (e.g., Defense Threat Reduction Agency, Gates Foundation, Johns Hopkins). Once recruited, we will elicit novel meta-model queries from each expert for MULTIVAC to translate. We will then present the expert cohort with sets of queries containing translated versions of queries drawn from the literature alongside those from human experts and those generated de novo by MULTIVAC. The experts will be asked to approve these representations or suggest modifications as appropriate, providing an important revision and validation for the ontology.

Our experts will communicate with MULTIVAC remotely, most likely through a dynamic web interface such as Jupyter Notebooks. The involvement of human experts in this phase serves two purposes. First, human expert review of the extracted queries and outputs provides a crucial check on system performance and helps identify areas most in need of improvement or optimization. Second, novel queries submitted by actual human experts provides an “out of sample” test for a system trained on a finite corpus. This review of derived queries and supply of novel human queries helps to build our training set of “authentic” queries for our final task of teaching MULTIVAC to generate entirely novel, well-formed queries on its own.

### Machine Generation of Expert Queries
Above, we describe how we will build a library of queries (extracted and expert-supplied) and a domain-specific ontology in the form of a Markov Logic Network. With these resources as its training data, MULTIVAC will train a Generative Adversarial Network (GAN) to produce well-formed, novel expert queries without human intervention, adapting recent work from Stanford.<sup>[9](#9)</sup> GANs comprise two main components, the generator and the discriminator. The more traditional discriminator network is a standard convolutional neural network that learns the boundaries between classes — for instance, well-formed expert queries and nonsense queries — by training on real-world examples. The generator network is an inverse convolutional network that models the distribution of individual classes in terms of their features. Thus, the generator network generates new query instances, while the discriminator evaluates them for validity. 

The discriminator network will be trained using these queries as our labeled training data. Meanwhile, the generator ingests models, parameters, factors and relationships and returns a “query” constructed from them. We will prime the generator network by having it compile the queries from the formulas in our MLN ontology using Markov-Chains to mimic the semantic query grammars embedded there, along with a random component to ensure novel combinations and variations. This fills out a meta-process model structure with models, parameters and factors taken from the accumulated ontology, and then sends this query to the generator network. This novel query is fed to the discriminator along with the existing set of curated expert queries. The discriminator considers both these real and generated queries and assigns probabilities of their authenticity, gradually learning to assign higher probabilities to “authentic” queries and lower ones to inauthentic queries.

<p align='center'><img src="https://github.com/GallupGovt/multivac/blob/master/images/gan.png" alt="GAN Design Graphic" width="600"></p>

This GAN architecture will be trained dialectically, first training the discriminator on the existing ontology and query library, then training the generator against a static discriminator. The discriminator will then be trained again, accounting for examples on which it failed, and so on. The discriminator will also be augmented by a “real-world” feedback loop; when the generator produces a query, the discriminator scores it, but the query is also submitted against the computational model simulation. If it produces results, the query is added to the discriminator training set as a valid expert query, regardless of the initial score given by the discriminator. Thus, new queries and query types can be added to the training library from successful novel queries. In the final iteration, the system will include a hypothesis evaluation loop looking at the explanatory power of a given machine-generated hypothesis and weighting up those that are novel, have a potentially high explanatory power and are plausible in the current context. This GAN implementaiton will be written in Python leveraging the Keras API with a TensorFlow backend.

### Model Output Interpretation
The ontology we build to map natural language queries to modeling configurations will serve as the basis for presenting the results in human readable ways. Generically, the results will consist of numerical point estimates, p-values, variable importance based on sensitivity analysis, etc. Our ontology for translating natural language queries into model parameters will also be general enough to include semantic notions of model outputs. Using this framework, we can abstractly map the outputs of our meta-models to quantitative results. 

### Related Codebases
- Stanford CoreNLP: https://github.com/stanfordnlp/python-stanford-corenlp 
- PracMLN: https://github.com/danielnyga/pracmln 
- QG-Net: https://github.com/moonlightlane/QG-Net 
- A-NICE-MC: https://github.com/ermongroup/a-nice-mc 

### Bibliography
<sup><a name='1'>1</a></sup> https://arxiv.org/abs/1805.04576 <br>
<sup><a name='2'>2</a></sup> https://nlp.stanford.edu/static/software/stanford-dependencies.shtml <br>
<sup><a name='3'>3</a></sup> https://github.com/stanfordnlp/python-stanford-corenlp <br>
<sup><a name='4'>4</a></sup> http://www.iapr-tc11.org/archive/icdar2011/fileup/PDF/4520b419.pdf <br>
<sup><a name='5'>5</a></sup> http://pracmln.org/ <br>
<sup><a name='6'>6</a></sup> https://homes.cs.washington.edu/~pedrod/papers/mlj05.pdf <br>
<sup><a name='7'>7</a></sup> https://homes.cs.washington.edu/~pedrod/papers/acl10.pdf <br>
<sup><a name='8'>8</a></sup> http://www.princeton.edu/~shitingl/papers/18l@s-qgen.pdf <br>
<sup><a name='9'>9</a></sup> https://arxiv.org/abs/1706.07561 <br>
