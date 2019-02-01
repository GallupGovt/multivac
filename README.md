# MULTIVAC
DARPA’s Information Innovation Office’s Automating Scientific Knowledge Extraction (ASKE) program seeks to develop approaches to make it easier for scientists to build, maintain and reason over rich models of complex systems — which could include physical, biological, social, engineered or hybrid systems. By interpreting and exposing scientific knowledge and assumptions in existing model code and documentation, researchers can identify new data and information resources automatically, extracting useful information from these sources, and integrating this useful information into machine-curated expert models for robust modeling.

![alt text](https://github.com/GallupGovt/multivac/blob/master/images/MULTIVAC_schematic.png 'MULTIVAC Concept Graphic')

Gallup’s Meta-model Unification Learned Through Inquiry Vectorization and Automated Comprehension (MULTIVAC) effort supports these goals by developing a system that absorbs scientific knowledge — in the form of facts, relationships, models and equations — from a particular domain corpus into a Markov Logic Network (MLN) ontology and learns to query that ontology in order to accelerate scientific exploration within the target domain. MULTIVAC will consist of an expert query generator trained on a corpus of historical expert queries and tuned dialectically with the use of a Generative Adversarial Network (GAN) architecture. As a prototype system, MULTIVAC will focus on the domain of epidemiological research, and specifically the realm of SIR/SEIR (Susceptible-Infected-Recovered, often with an additional “Exposed” element) compartmental model approaches. It is Gallup’s intent that this system includes a “human-in-the-loop” element, especially during training, to ensure that the system is properly tuned and responsive to the needs and interests of the human researchers it is intended to augment.

## Ontology Construction
- <a href='https://github.com/GallupGovt/multivac/tree/master/src/data'>Acquiring Data</a>
- <a href='https://github.com/GallupGovt/multivac/tree/master/src/data#parsing'>Processing and Parsing</a>
- <a href='https://github.com/GallupGovt/multivac/tree/master/pymln'>Ontology Induction</a>

## Query Training
- <a href='https://github.com/GallupGovt/multivac/tree/master/notebooks/query_gen'>Confirm Execution of Queries as Models</a>

### Human in the Loop - Expert Review
In phase one of this project, the queries generated from MULTIVAC’s modified QG-net implementation will stand unreviewed, with the understanding that the initial version is a proof of concept rather than a more robust prototype system. In the next phase of this project, Gallup will recruit a cohort of six to eight top experts in epidemiological modeling via our established connections with leading international organizations, NGOs and government agencies (e.g., Defense Threat Reduction Agency, Gates Foundation, Johns Hopkins). Once recruited, Gallup will elicit novel meta-model queries from each expert for MULTIVAC to translate. We will then present the expert cohort with sets of queries containing translated versions of queries drawn from the literature alongside those from human experts and those generated de novo by MULTIVAC. The experts will be asked to approve these representations or suggest modifications as appropriate, providing an important revision and validation for the ontology.

### Machine Generation of Queries
- <a href='https://github.com/GallupGovt/multivac/tree/master/notebooks/query_gen#gan'>Generative Adversarial Networks</a>

### Model Output Interpretation
MULTIVAC’s MLN knowledge base is built to map natural language queries to modeling configurations and will serve as the basis for presenting the results in human-readable ways. Generically, the results will consist of numerical point estimates, p-values if appropriate, variable importance based on sensitivity analysis and other metrics as required. The graphical model architecture for translating natural language queries into model parameters will also be general enough to include semantic notions of model outputs. Using this framework, MULTIVAC can abstractly map the outputs of its meta-models to quantitative results.

## Conclusion
The MULTIVAC system is still in development along several lines of effort, with a target date of April 1st, 2019, for an “alpha” code release. Subsequent to that milestone the development process will expose MULTIVAC to expert epidemiological research community for evaluation and improvement. Additionally, Gallup will explore a variety of avenues for optimization and improvements in terms of computational efficiencies, semantic representational precision and accuracy, and usability. MULTIVAC’s prototype iterations will be largely command line driven, but a more fully formed version should include web and/or graphical interfaces to encourage a broader user community across domains.

For more information please contact Lead Data Scientist Benjamin Ryan (ben_ryan@gallup.com).
