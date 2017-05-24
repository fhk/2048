import pickle

from flask import Flask, send_file, jsonify, abort, request

import numpy as np
import scipy.spatial

from words import nouns

app = Flask(__name__)

MODEL = None
VECTORS = None


def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx], idx


@app.route("/")
def index():
    return send_file("templates/index.html")


@app.route('/api/v1.0/words', methods=['POST'])
def create_word():
    if not request.json or not 'data' in request.json:
        abort(400)

    data = request.json
    words = data['data']['words']
    if len(words) == 2:
        indx_0 = [i for i, n in enumerate(nouns) if n == words[0]]
        indx_1 = [i for i, n in enumerate(nouns) if n == words[1]]

        added_vec = np.add(VECTORS[0][indx_0], VECTORS[0][indx_1])

        result = nouns[MODEL.query(added_vec[0])[1]]

        return jsonify({'data': result}), 201

if __name__ == '__main__':
    with open('word_vecs.pkl', 'r') as in_pkl:
        VECTORS = pickle.load(in_pkl)
        MODEL = scipy.spatial.cKDTree(VECTORS[0], leafsize=100)
    app.run(debug=True, host="0.0.0.0")
