import os
import requests
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

HUGGING_FACE_API_KEY = os.getenv("HF_API_KEY")  # Set this in your hosting platform

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    # Get the user input
    user_input = request.form.get('prompt')
    if not user_input.strip():
        return jsonify({"error": "Prompt cannot be empty!"}), 400

    # Send input to Hugging Face API
    headers = {"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"}
    payload = {"inputs": user_input}
    try:
        response = requests.post(
            "https://api-inference.huggingface.co/models/gpt2",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        data = response.json()
        generated_text = data[0]["generated_text"]
        return jsonify({"response": generated_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
