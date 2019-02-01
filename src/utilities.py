import os


def mkdir(directory):
    """Make a directory if it doesn't already exist."""
    if not os.path.exists(directory):
        print('Make directory %s' % directory)
        os.makedirs(directory)
    else:
        print('Directory %s already exists' % directory)



def dict_str(my_dict, results=''):
	'''Poor man's version of a pretty-ish print function for dictionaries to
	   expose the basic structure without printing all the values. '''
	results += '{'
	results += ', '.join(['{}: {}'.format(k,type(v)) if not isinstance(v, dict) else '{}:\n \t {} \n'.format(k, dict_str(v)) for k, v in my_dict.items()])

	# for key in my_dict:
	# 	results += str(key) + ': '

	# 	if isinstance(my_dict[key], dict):
	# 		results += dict_str(my_dict[key], results)
	# 	else:
	# 		results += str(type(my_dict[key]))

	results += '}'

	return results

