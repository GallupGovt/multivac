# MULTIVAC
Gallup’s Meta-model Unification Learned Through Inquiry Vectorization and Automated Comprehension (MULTIVAC) will consist of an expert query generator trained on a corpus of historical expert queries and tuned dialectically with the use of a Generative Adversarial Network (GAN) architecture.

For more information please contact Ben Ryan (ben_ryan@gallup.com).

## Approach
We first ingest natural language expert queries drawn from scientific literature and instantiate them as formalized computational models through the application of syntactic and semantic parsing and mapping tokens to model components. We confirm the system can execute these queries and produce desired results from the relevant models. We then use this ontology as well as live human expert input, to iteratively train a GAN to produce novel, well-formed and reasonable expert queries.

![alt text](https://github.gallup.com/ben-ryan/multivac/blob/master/images/multivac_concept.png "MULTIVAC Concept Graphic")

## Context
Even with state of the art computational modeling and simulation systems, experts have to translate their own queries into parameters and models by hand. Similarly, evaluation of existing models is laborious and time consuming. There is no systematic mechanism for assessment of model validity by independent users of the modeling code, and new hypotheses are always limited by human biases. Additionally, associated modeling code exhibits wide variation in structure, format and ease of access, and formulating hypotheses as parameter sets and program executions requires input from the original researchers.

## Impact
Training a system to understand “natural language” queries delivers two primary benefits: making models easier and more intuitive to operate by a wider range of scientists who might not be expert in manipulating modeling software, and giving the system the tools to retrieve additional queries automatically from scans of related scientific literature by parsing texts in the wild for new potential queries. This process results in the creation of a new ontology that maps decision space concepts to detailed parameter sets.

Ultimately, the MULTIVAC approach results in a “virtual scientist” system, capable of learning from the existing literature and generating queries that humans might not think of, greatly expanding the search space of possible model phenomena and potentially leading to discovery of many unknown unknowns.

## Resources
- MULTIVAC Slack: https://aske-multivac.slack.com/
- BioRxiv: https://arxiv.org/archive/q-bio
- Stanford CoreNLP: https://github.com/dasmith/stanford-corenlp-python
- Generative Adversarial Network (GAN) implementations using Keras: https://github.com/eriklindernoren/Keras-GAN

## Bibliography
- Asimov, I. (1956). The Last Question. Multivax. Retrieved from http://www.multivax.com/last_question.html

### Related NLP/AI/Deep Learning Research
- E. Cambria, S. Poria, D. Hazarika, T. Young. October 2018. "Recent Trends in Deep Learning Based Natural Language Processing," arXiv:1708.02709 [cs.CL] https://arxiv.org/pdf/1708.02709.pdf
- Song, J., Zhao, S., & Ermon, S. (2017). A-NICE-MC: Adversarial Training for MCMC. Proceedings of the 31st Annual Conference on Neural Information Processing Systems. Long Beach, CA. https://arxiv.org/abs/1706.07561
- W. Yih, X. He, C. Meek. "Semantic Parsing for Single-Relation Question Answering," Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics (Short Papers), pages 643–648, June 2014. Association for Computational Linguistics. http://acl2014.org/acl2014/P14-2/pdf/P14-2105.pdf
- Chen, Liu, Song. March 2018. "Tree-to-tree Neural Networks for Program Translation," https://openreview.net/forum?id=rkxY-sl0W
- "Deep Learning for NLP: An Overview of Recent Trends" August 2018. (Informal, basic overview of major types of DL NLP approaches and tasks) https://medium.com/dair-ai/deep-learning-for-nlp-an-overview-of-recent-trends-d0d8f40a776d

### Epidemiology Research
- GM Nakamura, ND Gomes, GC Cardoso, AS Martinez. July 2018. "Robust parameter determination in epidemic models with analytical descriptions of uncertainties," arXiv:1807.05301 [q-bio.PE] https://arxiv.org/abs/1807.05301
- Merler, S., Ajelli, M., Fumanelli, L., Gomes, M. F. C., Piontti, A. P. y, Rossi, L., … Vespignani, A. (2015). Spatiotemporal spread of the 2014 outbreak of Ebola virus disease in Liberia and the effectiveness of non-pharmaceutical interventions: a computational modelling analysis. The Lancet Infectious Diseases, 15(2), 204–211. https://doi.org/10.1016/S1473-3099(14)71074-6
- (1927). A Contribution to the Mathematical Theory of Epidemics. Proceedings of the Royal Society of London A, 115, 700-721. http://rspa.royalsocietypublishing.org/content/115/772/700
- Kostylenko, O., Rodrigues, H. S., & Torres, D. F. M. (2018). Banking Risk as an Epidemiological Model: An Optimal Control Approach. In A. I. F. Vaz, J. P. Almeida, J. F. Oliveira, & A. A. Pinto (Eds.), Operational Research (pp. 165–176). Springer International Publishing. https://link.springer.com/chapter/10.1007/978-3-319-71583-4_12
- Khan. (2014, September). A Computer Virus Propagation Model Using Delay Differential Equations With Probabilistic Contagion And Immunity. International Journal of Computer Networks & Communications (IJCNC) 6(5). Retrieved from the arXiv database. https://arxiv.org/abs/1410.5718
- Budak, C., Agrawal, D., & Abbadi. A.E. (2011). Limiting the spread of misinformation in social networks. Proceedings of the 20th international conference on World wide web, 665-674. Association for Computing Machinery. http://www.ra.ethz.ch/CDstore/www2011/proceedings/p665.pdf
- Jenness, S.M., Goodreau, S.M., & Morris, M. (2018). EpiModel: An R Package for Mathematical Modeling of Infectious Disease over Networks. Journal of Statistical Software, 84(8), 1-47. <doi:10.18637/jss.v084.i08>.
- Lawrance, C.E., & Croft, A.M. (2014). Do mosquito coils prevent malaria? A systematic review of trials. Journal of travel medicine, 11(2), 92-96. https://www.ncbi.nlm.nih.gov/pubmed/15109473
