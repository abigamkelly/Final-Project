from flask import Flask, request, render_template
from openai import OpenAI  # Correct import for OpenAI SDK

app = Flask(__name__)

# Define the two models
MODEL_A = "gpt-4o-mini"  # Critic
MODEL_B = "gpt-4o-mini"  # Initiator

# System instructions for each model
SYSTEM_INITIATOR = (
    #"You are the Initiator. You start or continue a conversation with creative ideas, "
    #"questions, or statements."
    "keep the response short: "
)
SYSTEM_INITIATOR2 = (
    " Based on the this response by a critic (do not respond to the critic), update your response (you may disagree as well). "
   )
SYSTEM_CRITIC = (
        "You are the Critic. considering your previous critics, criticize this response. keep the response short: "
)

@app.route("/", methods=["GET", "POST"])
def index():
    conversation_history1 = []  # Stores Initiator conversation
    conversation_history2 = []  # Stores Critic conversation
    conversation_history = []  # Stores conversation for display


    if request.method == "POST":
        api_key = request.form.get("api_key")
        initial_message = request.form.get("initial_message")
        num_turns = int(request.form.get("num_turns", 2))

        if not api_key or not initial_message:
            return render_template("index.html", error="Please enter an API key and an initial message.")

        client = OpenAI(api_key=api_key)

        message = initial_message  # Store the initial message
        message2 = ""  # Store critic's response
        conversation_history.append({"User": message})
        for i in range(num_turns):
            if i % 2 == 0:
                # **Initiator Block**
                model = MODEL_B
                system_prompt = SYSTEM_INITIATOR if i == 0 else SYSTEM_INITIATOR2
                message1 = system_prompt + (message if i == 0 else message2)
                
                conversation_history1.append({"role": "user", "content": message1})
                

                try:
                    response1 = client.chat.completions.create(
                        model=model,
                        messages=conversation_history1
                    )
                    # Extract the response content
                    message1 = response1.choices[0].message.content
                    conversation_history.append({"Initiator": message1})
                    conversation_history1.append({"role": "assistant", "content": message1})

                except Exception as e:
                    return render_template("index.html", error=f"API Error: {str(e)}")

            else:
                # **Critic Block**
                model = MODEL_A
                system_prompt = SYSTEM_CRITIC
                message2 = system_prompt + message1  # Provide Initiator's response to the Critic
                
                conversation_history2.append({"role": "user", "content": message2})

                try:
                    response2 = client.chat.completions.create(
                        model=model,
                        messages=conversation_history2
                    )
                    # Extract the response content
                    message2 = response2.choices[0].message.content
                    conversation_history.append({"Critic": message2})
                    conversation_history2.append({"role": "assistant", "content": message2})

                except Exception as e:
                    return render_template("index.html", error=f"API Error: {str(e)}")

    return render_template("index.html", conversation_history=conversation_history)

if __name__ == "__main__":
    print("Flask server starting on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
