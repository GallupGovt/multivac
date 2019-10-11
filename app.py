from flask import (
	Flask,
	request,
	render_template,
	redirect,
	url_for
)

# import the multivac query utility
# this is an exampe substitute for the query utility
def dummy_utility(args_dict):
	hash1 = hash(args_dict['arg1'])
	hash2 = hash(args_dict['arg2'])
	return f'OUTPUT: res1: {hash1}, res2: {hash2}'


app = Flask(__name__)
app.debug = True


@app.route('/')
def query():
    return render_template('query.html')


@app.route('/results')
def results():

	if request.method == 'GET':
		# populate inputs based on utility run args_dict
		inp1 = request.values.get('input-1')
		inp2 = request.values.get('input-2')

		# this is a dummy example using dummy_utility
		example_args_dict = {
			'arg1': inp1,
			'arg2': inp2
		}
		example_results = dummy_utility(example_args_dict)

		return render_template(
			'result.html',
			query=example_args_dict,
			results=example_results
		)

	else:
		return redirect(url_for('query'))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)