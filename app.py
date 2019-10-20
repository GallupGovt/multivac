from flask import (
	Flask,
	request,
	render_template,
	redirect,
	url_for
)

from src.rdf_graph import map_queries


app = Flask(__name__)
app.debug = True


@app.route('/')
def query():

    return render_template(
    	'query.html'
    )


@app.route('/results')
def results():

	if request.method == 'GET':

		args_dict = {
			'docker_folder_structure': [x for x in os.walk(os.getcwd())],
			'dir': request.values.get('dir-input'),
			'model': request.values.get('model-type-input'),
			'out': request.values.get('out-input'),
			'run': request.values.get('run-input'),
			'threshold': request.values.get('threshold-input'),
			'verbose': request.values.get('verbosity-input'),
			'num_top_rel': request.values.get('num-top-input'),
			'search': request.values.get('search-input'),
		}

		# results = map_queries.run(args_dict)

		return args_dict

	else:
		return redirect(url_for('query'))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)