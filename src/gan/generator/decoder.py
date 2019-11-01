
import re
import traceback

from multivac.src.gan.generator.lang.eng.unaryclosure import compressed_ast_to_normal
from multivac.src.gan.generator.lang.eng.grammar import BRACKET_TYPES
from multivac.src.gan.generator.model import *

def decode_tree_to_string(decode_tree):
    compressed_ast_to_normal(decode_tree)
    decode_tree = decode_tree.children[0]
    result = [x.value for x in decode_tree.get_leaves()]

    return re.sub("<eos>", "", ' '.join(result)).strip()

def decode_english_dataset(model, dataset, cfg):
    verbose = cfg['verbose']

    if verbose:
        print('decoding [{}] set, num. examples: {}'.format(dataset.name, 
                                                            dataset.count))

    decode_results = []
    cum_num = 0

    for example in dataset.examples:
        cand_list = model.decode(example, 
                                 dataset.grammar, 
                                 dataset.terminal_vocab,
                                 beam_size=cfg['beam_size'], 
                                 max_time_step=cfg['decode_max_time_step'])

        exg_decode_results = []

        for cid, cand in enumerate(cand_list[:10]):
            try:
                text = decode_tree_to_string(cand.tree)
                exg_decode_results.append((cid, cand, text))
            except:
                if verbose:
                    print("Exception in converting tree to code:")
                    print(('-' * 60))
                    print(('raw_id: %d, beam pos: %d' % (example.raw_id, cid)))
                    traceback.print_exc(file=sys.stdout)
                    print(('-' * 60))

        cum_num += 1
        if cum_num % 50 == 0 and verbose:
            print(('%d examples so far ...' % cum_num))

        decode_results.append(exg_decode_results)

    return decode_results
