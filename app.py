from flask import Flask, request
from flask_restx import Resource, Api, reqparse, Namespace
from MrBanana_generation import generator

app = Flask(__name__)
api = Api(app,
        version='1.0',
        title="AI Relay Novel Generation program",
        description="Relay Novel generation program API with Korean GPT-2",
        contact="mari970@naver.com",
        license='GIST'
        )
#
# Relay = Namespace('Relay')
# api.add_namespace(Relay, '/relay_generation')

# parser = reqparse.RequestParser()
# parser.add_argument('name', type=str, help= 'user name')
# parser.add_argument('input sentence', type = str, help= 'input korean sentence')
# args = parser.parse_args()

@api.route('/')
class Inference(Resource):
    def post(self):
        # address = 'http://210.125.85.141:5000/'

        # novel[idx] = requests.json.get(args['input sentence'])
        sentence = request.form['sentence'] # 'http://210.125.85.141/novel'
        out = generator(sentence)
        # novel[idx+1] = requests.json.get(out)
        output = {'response': out}

        return output, 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
