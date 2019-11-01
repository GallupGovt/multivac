
from multivac.src.gan.gen_pyt.common.registerable import Registrable
from multivac.src.gan.gen_pyt.datasets.utils import ExampleProcessor
from multivac.src.gan.gen_pyt.datasets.english.dataset import English
from multivac.src.rdf_graph.rdf_parse import StanfordParser

@Registrable.register('english_example_processor')
class EnglishExampleProcessor(ExampleProcessor):
    def __init__(self, transition_system):
        self.transition_system = transition_system
        self.parser = StanfordParser(annots = "tokenize ssplit parse")

    def pre_process_utterance(self, utterance):
        toks, text, parse_str, tree = English.canonicalize_example(utterance,
                                                                   self.parser)
        return toks, tree

    def post_process_hypothesis(self, hyp):
        """traverse the AST and replace slot ids with original strings"""
        text = self.transition_system.ast_to_surface_text(hyp.tree)
        return text
