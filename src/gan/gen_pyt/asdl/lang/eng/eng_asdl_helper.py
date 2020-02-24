# coding=utf-8

from multivac.src.gan.gen_pyt.asdl.asdl import (ASDLCompositeType,
                                                ASDLConstructor,
                                                ASDLPrimitiveType,
                                                ASDLProduction, Field)
from multivac.src.gan.gen_pyt.asdl.asdl_ast import (AbstractSyntaxTree,
                                                    RealizedField)


def find_match_paren(s):
    count = 0

    for i, c in enumerate(s):
        if c == "(":
            count += 1
        elif c == ")":
            count -= 1

        if count == 0:
            return i


def english_ast_to_asdl_ast(text, depth=0, debug=False):
    ''' Takes a constituency parse string of an English sentence and creates
        an AbstractSyntaxTree object from it.

        Example input:
        '(ROOT (SBARQ (WHADVP (WRB Why)) (SQ (VBP do) (NP (NNS birds)) (ADVP
        (RB suddenly)) (VP (VB appear) (SBAR (WHADVP (WRB whenever)) (S (NP
        (PRP you)) (VP (VBP are) (ADJP (JJ near))))))) (. ?)))'
    '''

    if debug:
        print(("\t" * depth + "String: '{}'".format(text)))

    try:
        tree_str = text[text.index("(") + 1:text.rfind(")")]
    except ValueError:
        print(("Malformatted parse string: '{}'".format(text)))
        raise ValueError

    all_fields = []
    next_idx = tree_str.index(" ")

    if "(" in tree_str:
        node_type = ASDLCompositeType(tree_str[:next_idx])
        node_fields = []

        while "(" in tree_str:
            tree_str = tree_str[tree_str.index("("):]
            next_idx = find_match_paren(tree_str) + 1
            child = english_ast_to_asdl_ast(tree_str[:next_idx], depth+1, debug)

            if isinstance(child, AbstractSyntaxTree):
                asdl_field = Field(child.production.type.name,
                                   child.production.type,
                                   'single')
                all_fields.append(RealizedField(asdl_field, value=child))
            else:
                asdl_field = child.field
                all_fields.append(child)

            node_fields.append(asdl_field)
            tree_str = tree_str[next_idx + 1:]

        field_str = ', '.join(["({})".format(f.name) for f in node_fields])
        rule_str = node_type.name + " -> " + field_str
        constructor = ASDLConstructor(rule_str, node_fields)
        production = ASDLProduction(node_type, constructor)

        result = AbstractSyntaxTree(production, realized_fields=all_fields)
    else:
        node_type = ASDLPrimitiveType(tree_str[:next_idx])
        result = RealizedField(Field(node_type.name, node_type, 'single'),
                               value=tree_str[next_idx + 1:])

    return result


def asdl_ast_to_english(asdl_ast_node):
    tokens = []

    for field in asdl_ast_node.fields:
        # for composite node
        field_value = None

        if isinstance(field.type, ASDLCompositeType) and field.value:
            field_value = asdl_ast_to_english(field.value)
        else:
            field_value = field.value

        tokens.append(field_value)

    return ' '.join([x if x else '<None>' for x in tokens])
