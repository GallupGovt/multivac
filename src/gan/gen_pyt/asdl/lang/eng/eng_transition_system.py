# coding=utf-8

from asdl.lang.py.py_asdl_helper import asdl_ast_to_python_ast, python_ast_to_asdl_ast
from multivac.src.rdf_graph.rdf_parse import tokenize_text
from asdl.transition_system import TransitionSystem, GenTokenAction

from common.registerable import Registrable


@Registrable.register('english')
class EnglishTransitionSystem(TransitionSystem):
    def tokenize_text(self, text, mode=None):
        return tokenize_text(text, mode)

    def surface_text_to_ast(self, text):
        py_ast = ast.parse(text).body[0]
        return python_ast_to_asdl_ast(py_ast, self.grammar)

    def ast_to_surface_text(self, asdl_ast):
        py_ast = asdl_ast_to_python_ast(asdl_ast, self.grammar)
        text = astor.to_source(py_ast).strip()

        return text

    def compare_ast(self, hyp_ast, ref_ast):
        hyp_text = self.ast_to_surface_text(hyp_ast)
        ref_reformatted_text = self.ast_to_surface_text(ref_ast)

        ref_text_tokens = tokenize_text(ref_reformatted_text)
        hyp_text_tokens = tokenize_text(hyp_text)

        return ref_text_tokens == hyp_text_tokens

    def get_primitive_field_actions(self, realized_field):
        actions = []
        if realized_field.value is not None:
            if realized_field.cardinality == 'multiple':  # expr -> Global(identifier* names)
                field_values = realized_field.value
            else:
                field_values = [realized_field.value]

            tokens = []
            if realized_field.type.name == 'string':
                for field_val in field_values:
                    tokens.extend(field_val.split(' ') + ['</primitive>'])
            else:
                for field_val in field_values:
                    tokens.append(field_val)

            for tok in tokens:
                actions.append(GenTokenAction(tok))

        return actions

    def is_valid_hypothesis(self, hyp, **kwargs):
        try:
            hyp_text = self.ast_to_surface_text(hyp.tree)
            ast.parse(hyp_text)
            self.tokenize_text(hyp_text)
        except:
            return False
        return True
