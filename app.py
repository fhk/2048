from flask import Flask, send_file, jsonify, abort, request

import gensim
import numpy as np

app = Flask(__name__)

MODEL = None


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
        result = MODEL.most_similar_cosmul(
            positive=words, topn=5)
        for i, r in enumerate(result):
            if r[0] + 's' in words:
                continue
            else:
                return jsonify({'data': result[i][0]}), 201

if __name__ == '__main__':
    MODEL = gensim.models.KeyedVectors.load_word2vec_format("../../Downloads/GoogleNews-vectors-negative300.bin.gz", binary=True)
    app.run(debug=True, host="0.0.0.0")
