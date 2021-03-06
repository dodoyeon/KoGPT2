from flask import Flask, request
from flask_restx import Resource, Api, reqparse, Namespace
from MrBanana_generation import generator

# Inference SERVER
app = Flask(__name__)
api = Api(app,
        version='1.0',
        title="AI Relay Novel Generation program",
        description="Relay Novel generation program API with Korean GPT-2",
        contact="mari970@naver.com",
        license='GIST'
        )

# Relay = Namespace('Relay')
# api.add_namespace(Relay, '/relay_generation')
#
# parser = reqparse.RequestParser()
# parser.add_argument('name', type=str, help= 'user name')
# parser.add_argument('input sentence', type = str, help= 'input korean sentence')
# args = parser.parse_args()

@api.route('/')
class Inference(Resource):
    def post(self):
        # novel[idx] = requests.json.get(args['sentence'])
        sentence = request.get_json()['sentence']
        print(sentence)
        out = generator(sentence)
        # novel[idx+1] = requests.json.get(out)
        output = {'response': out}
        print(output)
        return output # , 200 = HTTP에서 성공을 의미하는 status code

if __name__ == "__main__":
    app.run(host='172.26.37.206', port=5000, debug=True)
