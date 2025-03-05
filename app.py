from flask import Flask, request, render_template
import openai
from openai import OpenAI


app = Flask(__name__)

# Define the two models
MODEL_A = "gpt-4o"
MODEL_B = "gpt-3.5-turbo"

@app.route("/", methods=["GET", "POST"])
def index():
    conversation_history = []  # Stores the conversation messages

    if request.method == "POST":
        api_key = request.form.get("api_key")
        initial_message = request.form.get("initial_message")
        client = OpenAI(api_key=api_key)
        num_turns = int(request.form.get("num_turns", 5))

        if not api_key or not initial_message:
            return render_template("index.html", error="Please enter an API key and an initial message.")

        message = initial_message

        for i in range(num_turns):
            # Alternate between the two models
            model = MODEL_A if i % 2 == 0 else MODEL_B

            try:
                response = client.chat.completions.create(model=model,
                messages=[{"role": "user", "content": message}])
                # Extract the content
                message = response.choices[0].message.content
                # Store conversation in our history list
                conversation_history.append((model, message))

            except openai.OpenAIError as e:
                return render_template("index.html", error=f"API Error: {str(e)}")

    return render_template("index.html", conversation_history=conversation_history)

if __name__ == "__main__":
    # Print a message so you know the server is running
    print("Flask server starting on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
