from flask import Flask
from flask import jsonify
from flask import make_response
from flask_cors import CORS

from server.poems_table import Poems

app = Flask(__name__)
CORS(app)

poems = Poems()


@app.route('/poem')
def get_poem():
    poem_text = poems.get_random_poem()[:1000]
    response = make_response(jsonify(text=poem_text))
    response.headers['content-type'] = 'text/plain'
    return response


if __name__ == '__main__':
    app.run(port=8080)
