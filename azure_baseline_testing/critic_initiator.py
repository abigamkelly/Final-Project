import csv
import io
from openai import AzureOpenAI

def main():
    # Read the API key from a text file
    with open("api_key.txt", "r") as file:
        api_key = file.readline().rstrip()

    CSV_FILE = "data/tweets.csv"
    OUTPUT_CSV = "classified_tweets.csv"
    MAX_TWEETS = 10

    # Initial threshold for the conversation trigger
    CONVERSATION_THRESHOLD = 0.85

    # System prompt for the initiator
    SYSTEM_INITIATOR = (
        '''I am a researcher interested in studying disaster response events using Graph Neural Networks (GNNs).
            My goal is to analyze historical disaster events through tweets to better understand the timeliness 
            and effectiveness of responses to specific event types. Each tweet I receive includes a disaster-related hashtag, 
            but not all tweets are relevant to my research objective. I want a model to classify tweets as either: 1 (Useful for disaster response analysis), 
            or 0 (Not useful for disaster response analysis). Also provide a confidence percentage for your classification represented as a decimal between 
            0 and 1 and your reasoning for the classification (regardless of classification label, high confidence should be close to 1 
            and low confidence should be close to 0, for example if something is classified in class 0 and you are very confident, the confidence should be close to 1
            ).  Return your response as the classification, the percentage decimal, 
            and your reasoning separated by '\n' (e.g,. 1\n 0.4\n reasoning).'''
    )

    # System prompt for the critic (if conversation is triggered)
    SYSTEM_CRITIC = (
        "You are a discerning critic with expertise in disaster response tweet analysis. "
        "Review the following classification, its confidence, and reasoning. "
        "Provide a detailed critique and, if appropriate, suggest an alternative classification along with improved reasoning."
    )

    conversation_history = []  # For display and record

    # Azure OpenAI configuration
    endpoint = "https://sv-openai-research-group4.openai.azure.com/"
    model_name = "gpt-4o"
    deployment1 = "gpt-4o-2"  # critic model
    deployment2 = "gpt-4o-2"  # initiator model
    api_version = "2024-12-01-preview"

    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=api_key,
    )

    results = []
    tweet_count = 0

    # Open and read CSV file
    with open(CSV_FILE, "r", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            # Stop after processing MAX_TWEETS
            if tweet_count >= MAX_TWEETS:
                break

            if not row or len(row) == 0:
                continue

            tweet = row[0].strip()
            if not tweet:
                print("skipping an empty tweet")
                continue

            # Prepare messages for the initiator
            system_message = {"role": "system", "content": SYSTEM_INITIATOR}
            user_message = {"role": "user", "content": f"Tweet: {tweet}"}
            messages = [system_message, user_message]

            try:
                # Initial classification call with the initiator model
                response = client.chat.completions.create(
                    model=deployment2,
                    messages=messages,
                    max_tokens=50,
                    temperature=0.5,
                    top_p=0.5
                )
                answer = response.choices[0].message.content.strip()
                # Expected answer format: "classification\nconfidence\nreasoning"
                parts = answer.split('\n')
                if len(parts) == 3:
                    classification = parts[0].strip()
                    confidence = parts[1].strip()
                    reasoning = parts[2].strip()
                else:
                    classification = "N/A"
                    confidence = "N/A"
                    reasoning = "N/A"

                conversation_transcript = f"Initiator initial response: {answer}"

                # Convert confidence to float for comparison if possible
                try:
                    confidence_value = float(confidence)
                except ValueError:
                    confidence_value = 1.0  # if conversion fails, assume high confidence

                # If confidence is low, trigger a conversation between the models
                if confidence_value < CONVERSATION_THRESHOLD:
                    # Critic conversation
                    critic_system_message = {"role": "system", "content": SYSTEM_CRITIC}
                    critic_user_message = {"role": "user", "content": (
                        f"Review the following classification for the tweet:\n"
                        f"Tweet: {tweet}\n"
                        f"Classification: {classification}\n"
                        f"Confidence: {confidence}\n"
                        f"Reasoning: {reasoning}\n"
                        "The confidence is low. Please provide your critique and suggestions."
                    )}
                    critic_messages = [critic_system_message, critic_user_message]

                    critic_response = client.chat.completions.create(
                        model=deployment1,
                        messages=critic_messages,
                        max_tokens=50,
                        temperature=0.5,
                        top_p=0.5
                    )
                    critic_answer = critic_response.choices[0].message.content.strip()
                    conversation_transcript += f"\nCritic response: {critic_answer}"

                    # Initiator follow-up conversation
                    followup_user_message = {"role": "user", "content": (
                        f"Critic feedback: {critic_answer}\n"
                        "Please provide an updated classification, confidence, and reasoning in the same format."
                    )}
                    followup_messages = [system_message, followup_user_message]

                    followup_response = client.chat.completions.create(
                        model=deployment2,
                        messages=followup_messages,
                        max_tokens=50,
                        temperature=0.5,
                        top_p=0.5
                    )
                    followup_answer = followup_response.choices[0].message.content.strip()
                    conversation_transcript += f"\nInitiator follow-up: {followup_answer}"

                    # Parse follow-up response if possible
                    followup_parts = followup_answer.split('\n')
                    if len(followup_parts) == 3:
                        classification = followup_parts[0].strip()
                        confidence = followup_parts[1].strip()
                        reasoning = followup_parts[2].strip()

                results.append({
                    "tweet": tweet,
                    "classification": classification,
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "conversation": conversation_transcript
                })
                print(f"Processed tweet {tweet_count + 1}: {tweet}")
            except Exception as e:
                print(f"Error processing tweet: {tweet}\nError: {str(e)}")
            tweet_count += 1

    # Write the results to a new CSV file with updated fieldnames
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as out_file:
        fieldnames = ["tweet", "classification", "confidence", "reasoning", "conversation"]
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Classification complete. Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
