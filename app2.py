from flask import Flask, render_template, request
from flask_restx import Resource, Api
import requests

# Hompage SERVER
app = Flask(__name__)
api = Api(app,
        version='1.0',
        title="AI Relay Novel Generation program",
        description="Relay Novel generation program API with Korean GPT-2",
        contact="mari970@naver.com",
        license='GIST'
        )

novel = {}
count = 1

@api.route('/')
class Relay(Resource):
    def front(self):
        return render_template('submit.html')
    
    # @api.route('/inference')
    def post(self):
        global novel
        sentence = request.form['sent']
        data = {'sentence' : sentence}
        novel += sentence
        result = request.post(, data=data)
        
        result = result.json()
        response = result['response']
        return {
            'sentence':sentence,
            'result':response
        }

if __name__ == "__main__":
    app.run(host='0.0.0.0')