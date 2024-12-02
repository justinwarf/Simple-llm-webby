import os
import requests
from flask import Flask, request, jsonify, render_template, make_response

app = Flask(__name__)

HUGGING_FACE_API_KEY = os.getenv("HF_API_KEY")
API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"  # Use a better model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    # Get the user input and session history
    user_input = request.form.get('prompt', '').strip()
    if not user_input:
        return jsonify({"error": "Prompt cannot be empty!"}), 400

    # Get the session history, but only keep the most recent conversation (user + model)
    session_history = request.cookies.get('history', '')
    
    # Only append the most recent user input and the previous response to the session history
    session_history = f"User: {user_input}\n"  # Only include the most recent user input for context

    # Prepare the payload
    payload = {
        "inputs": session_history,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.7,
            "top_p": 0.9,
        },
    }

    # Call the Hugging Face API
    headers = {"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"}
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

        # Extract the model's response
        model_response = data[0]["generated_text"].strip()

        # Remove "User:" from the end of the model's response if it appears
        if model_response.endswith("User:"):
            model_response = model_response[:-5].strip()

        # Save only the last response and question to the session history
        session_history = f"User: {user_input}\nLLM: {model_response}\n"

        # Send the response back to the client
        resp = make_response(jsonify({"response": model_response}))
        resp.set_cookie("history", session_history)
        return resp
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
