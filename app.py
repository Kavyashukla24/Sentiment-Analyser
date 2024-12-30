from flask import Flask, render_template, request, jsonify
from transformers import pipeline

# Initialize the Flask app
app = Flask(__name__)

# Initialize the sentiment analysis pipeline from Huggingface
sentiment_analyzer = pipeline('sentiment-analysis')

# Home route to render the frontend
@app.route('/')
def home():
    print("Home route is working!")  # Print message to the console
    return render_template('index.html')

# API route for sentiment analysis
@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    text = request.json.get('text', '')  # Get the text input from the frontend
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Perform sentiment analysis
    result = sentiment_analyzer(text)

    # Extract the sentiment result (label and score)
    sentiment = result[0]['label']
    score = result[0]['score']

    # Return the result as JSON
    return jsonify({'sentiment': sentiment, 'score': score})

if __name__ == '__main__':
    print("Starting Flask server...")  # Print message to indicate server is starting
    app.run(debug=True)
