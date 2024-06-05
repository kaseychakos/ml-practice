from flask import Flask, json, jsonify, request
from lib.sentiment_analysis import SentimentAnalysis
from util.validate import validate

app = Flask(__name__)
sentiment = SentimentAnalysis()

@app.route('/', methods=["GET"])
def status():
    return jsonify(success=True)

@app.route('/query', methods=["POST"])
def handle_query():
    # validate input
    valid, err = validate(request)
    if not valid:
        return err
    
    req = request.json

    # standardize input then predict sentiment and return results
    standardized, tokenized = sentiment.standardize(req['query'])
    result = sentiment.predict(req['query'], standardized, tokenized)
    data = {
        "query" : req['query'],
        "result" : result.to_json(orient='records')
    }
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=False)