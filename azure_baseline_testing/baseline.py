# This file contains the code that gets a baseline accuracy for the Azure model
# A CSV file containing the tweet, confidence level, and classification is produced

import csv
import io
from openai import AzureOpenAI

def main():

    # Read the API key from a text file
    with open("api_key.txt", "r") as file:
        api_key = file.readline()
        api_key = api_key.rstrip()

    CSV_FILE = "data/tweets.csv"
    OUTPUT_CSV = "classified_tweets.csv"
    MAX_TWEETS = 100

    # Azure OpenAI configuration
    endpoint = "https://sv-openai-research-group4.openai.azure.com/"
    deployment = "gpt-4o-2"
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

            # Prepare prompt messages for classification
            system_message = {
                "role": "system",
                "content": (
                    "I am a researcher interested in studying disaster response events using Graph Neural Networks (GNNs). My goal is to analyze historical disaster events through tweets to better understand the timeliness and effectiveness of responses to specific event types. Each tweet I receive includes a disaster-related hashtag, but not all tweets are relevant to my research objective. I want a model to classify tweets as either: 1 (Useful for disaster response analysis), or 0 (Not useful for disaster response analysis). Also provide a confidence percentage for your classification represented as a decimal between 0 and 1.  Return your response as the classification and the percentage decimal separated by a comma (e.g,. 1, 0.4)."
                )
            }
            user_message = {"role": "user", "content": f"Tweet: {tweet}"}
            messages = [system_message, user_message]

            try:
                response = client.chat.completions.create(
                    model=deployment,
                    messages=messages,
                    max_tokens=50,
                    temperature=0.5,
                    top_p=0.5
                )
                answer = response.choices[0].message.content.strip()
                # Expected answer format: "classification, percentage"
                parts = answer.split(',')
                if len(parts) == 2:
                    classification = parts[0].strip()
                    confidence = parts[1].strip()
                else:
                    classification = "N/A"
                    confidence = "N/A"

                results.append({
                    "tweet": tweet,
                    "classification": classification,
                    "confidence": confidence
                })
                print(f"Processed tweet {tweet_count + 1}: {tweet}")
            except Exception as e:
                print(f"Error processing tweet: {tweet}\n  Error: {str(e)}")
            tweet_count += 1

    # Write the results to a new CSV file
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as out_file:
        fieldnames = ["tweet", "classification", "confidence"]
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Classification complete. Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

