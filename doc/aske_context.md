# MULTIVAC in the ASKE Context
![ASKE Schematic v1.0](https://github.com/GallupGovt/multivac/blob/master/images/aske_schematic_v1.png)
![ASKE Schematic v1.5](https://github.com/GallupGovt/multivac/blob/master/images/aske_schematic_v1.5.png)

Gallup’s MULTIVAC system extracts scientific knowledge — in the form of facts, relationships, equations — from a given domain corpus consisting of natural language text and formal mathematical equations. The system then compiles this knowledge into a curated probabilistic graphical model (specifically, a Markov Logic Network) knowledgebase. Finally, the system learns to query that knowledge base in order to accelerate scientific exploration within the target domain. 

With reference to the first ASKE program schematic on the previous slide, MULTIVAC is more or less vertically integrated across the discovery/extraction, curation, and inference. 

The end objective, however, is hypothesis generation. This feature situates the most novel contribution of MULTIVAC essentially outside these levels, at the top of the more process-oriented second schematic on the previous slide. In effect, MULTIVAC’s “inference” component inverts the standard intention and, instead of using the work done in the extraction and curation layers to arrive at new inferences, learns through observation and experimentation how to ask it’s own novel questions that then require more standard inference solutions to answer. Other projects in the program have presented innovative ways of automating or enhancing execution of human inquiries. Our system seeks to automate the production and evolution of those queries in the first place.

The final goal of a MULTIVAC system for any given domain is to generate new scientific queries relevant to that domain that have not been asked before by humans. These inquiries, properly formatted, could in theory even act as inputs to many of the other TA2 systems.

### Wait, but Why?
- The glacial pace of evolution in paradigms and modes of inquiry within domains. 
- Stove-pipes within and between domains of scientific inquiry


## ASKE Potential Use Cases
### Modernizing and consolidating old research:
- While much research is available in digital form today, vast archives exist in hard copy in various forms that are far less searchable. Using an ASKE system to ingest and compile/curate these types of repositories could help revitalize forgotten areas of research.

### Breaking stovepipes:
- Sometimes research fields become balkanized between different communities based on approaches, terminologies, or simply favored publication venues. An ASKE system that can comprehend a field at scale across these artificial segmentations could help break irrational logjams and cross-pollinate discoveries. 

### Revitalizing stagnant areas of research:
- Occasionally research fields lose momentum or interest, as consensus emerges on “big questions” or as unknowns become more apparently “unknowable.” Paradigm shifts can happen that help break this stagnation and revolutionize fields, but this can take a great deal of time and is never guaranteed. A system that can analyze a field of research and produce novel questions or avenues of inquiry can help inject new creativity and perspectives and revitalize research.
