import csv
import io
import os
import re
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from openai import AzureOpenAI

def preprocess_tweet(tweet):
    """
    Preprocess tweet text by removing URLs, mentions, special characters,
    and normalizing whitespace.
    """
    # Remove URLs
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    
    # Remove mentions
    tweet = re.sub(r'@\w+', '', tweet)
    
    # Remove HTML tags
    tweet = re.sub(r'<.*?>', '', tweet)
    
    # Remove special characters and numbers
    tweet = re.sub(r'[^\w\s]', '', tweet)
    tweet = re.sub(r'\d+', '', tweet)
    
    # Normalize whitespace
    tweet = re.sub(r'\s+', ' ', tweet).strip()
    
    return tweet

def load_and_preprocess_data(csv_file, max_tweets=None, test_size=0.2):
    """
    Load and preprocess the dataset, splitting into train and validation sets.
    Returns preprocessed dataframes and the original dataframe.
    """
    try:
        # Load the dataset
        df = pd.read_csv(csv_file)
        
        # Limit the number of tweets if specified
        if max_tweets and max_tweets > 0:
            df = df.sample(min(max_tweets, len(df)), random_state=42)
        
        # Make a copy of the original dataframe
        original_df = df.copy()
        
        # Preprocess the tweets
        df['processed_text'] = df['text'].apply(preprocess_tweet)
        
        # Check if 'target' column exists (for training data)
        if 'target' in df.columns:
            # Split into train and validation sets
            train_df, val_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df['target'])
            return train_df, val_df, original_df
        else:
            # For test data without target
            return df, None, original_df
            
    except Exception as e:
        print(f"Error loading or preprocessing data: {str(e)}")
        return None, None, None

def visualize_data(df):
    """
    Create visualizations for the dataset to understand its characteristics.
    """
    # Create a directory for visualizations if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    # 1. Distribution of disaster vs non-disaster tweets
    if 'target' in df.columns:
        plt.figure(figsize=(8, 6))
        sns.countplot(x='target', data=df)
        plt.title('Distribution of Disaster vs Non-Disaster Tweets')
        plt.xlabel('Target (0: Non-Disaster, 1: Disaster)')
        plt.ylabel('Count')
        plt.savefig('visualizations/target_distribution.png')
        plt.close()
    
    # 2. Tweet length distribution
    plt.figure(figsize=(10, 6))
    df['tweet_length'] = df['text'].apply(len)
    sns.histplot(data=df, x='tweet_length', bins=50, kde=True)
    plt.title('Distribution of Tweet Lengths')
    plt.xlabel('Tweet Length (characters)')
    plt.ylabel('Count')
    plt.savefig('visualizations/tweet_length_distribution.png')
    plt.close()
    
    # 3. Word count distribution
    plt.figure(figsize=(10, 6))
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
    sns.histplot(data=df, x='word_count', bins=30, kde=True)
    plt.title('Distribution of Word Counts in Tweets')
    plt.xlabel('Word Count')
    plt.ylabel('Count')
    plt.savefig('visualizations/word_count_distribution.png')
    plt.close()
    
    # 4. Most common keywords if available
    if 'keyword' in df.columns:
        plt.figure(figsize=(12, 8))
        top_keywords = df['keyword'].value_counts().head(20)
        sns.barplot(x=top_keywords.values, y=top_keywords.index)
        plt.title('Top 20 Keywords')
        plt.xlabel('Count')
        plt.ylabel('Keyword')
        plt.savefig('visualizations/top_keywords.png')
        plt.close()
    
    print("Visualizations created and saved in 'visualizations' directory.")

def normalize_classification(classification, reasoning=""):
    """
    Normalize classification to ensure it's either "0" or "1".
    Uses reasoning text to help determine the correct classification if needed.
    """
    # First, try to clean and extract just the digit
    cleaned = re.sub(r'[^01]', '', classification.strip())
    if cleaned in ["0", "1"]:
        return cleaned
    
    # If that fails, try to interpret based on reasoning
    lower_reasoning = reasoning.lower()
    if any(term in lower_reasoning for term in ["not disaster", "not a disaster", "non-disaster", "irrelevant", 
                                               "metaphor", "joke", "not related", "not about disaster"]):
        return "0"
    elif any(term in lower_reasoning for term in ["disaster", "emergency", "crisis", "evacuation", 
                                                 "fire", "flood", "earthquake", "hurricane"]):
        return "1"
    
    # If still can't determine, default to "0" (non-disaster) as a safer choice
    return "0"

def evaluate_model(true_labels, predicted_labels, confidence_scores=None):
    """
    Evaluate the model using various metrics and return the results.
    """
    # Print raw predictions for debugging
    print("Raw predictions:", predicted_labels[:5], "...")
    
    # Normalize and validate predictions
    normalized_predictions = []
    valid_indices = []
    
    for i, (pred, true) in enumerate(zip(predicted_labels, true_labels)):
        # Normalize the prediction to ensure it's "0" or "1"
        norm_pred = normalize_classification(str(pred))
        normalized_predictions.append(norm_pred)
        
        # Keep track of valid indices
        if norm_pred in ["0", "1"]:
            valid_indices.append(i)
    
    # Update predicted_labels with normalized values
    for i, norm_pred in enumerate(normalized_predictions):
        predicted_labels[i] = norm_pred
    
    # Print normalized predictions for debugging
    print("Normalized predictions:", normalized_predictions[:5], "...")
    print(f"Valid predictions: {len(valid_indices)}/{len(predicted_labels)}")
    
    # If no valid predictions, return empty metrics
    if not valid_indices:
        print("Warning: No valid predictions found for evaluation.")
        return {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'confusion_matrix': np.array([[0, 0], [0, 0]]),
            'avg_confidence': None
        }
    
    # Filter the labels and scores
    filtered_true_labels = [true_labels[i] for i in valid_indices]
    filtered_predicted_labels = [normalized_predictions[i] for i in valid_indices]
    
    # Convert labels to integers
    filtered_true_labels = [int(label) for label in filtered_true_labels]
    filtered_predicted_labels = [int(label) for label in filtered_predicted_labels]
    
    # Calculate metrics
    accuracy = accuracy_score(filtered_true_labels, filtered_predicted_labels)
    precision = precision_score(filtered_true_labels, filtered_predicted_labels, zero_division=0)
    recall = recall_score(filtered_true_labels, filtered_predicted_labels, zero_division=0)
    f1 = f1_score(filtered_true_labels, filtered_predicted_labels, zero_division=0)
    
    # Create confusion matrix
    cm = confusion_matrix(filtered_true_labels, filtered_predicted_labels)
    
    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Disaster', 'Disaster'],
                yticklabels=['Non-Disaster', 'Disaster'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('visualizations/confusion_matrix.png')
    plt.close()
    
    # If confidence scores are provided, analyze them
    if confidence_scores:
        # Filter confidence scores
        filtered_confidence_scores = [confidence_scores[i] for i in valid_indices]
        
        # Plot confidence distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(filtered_confidence_scores, bins=20, kde=True)
        plt.title('Distribution of Confidence Scores')
        plt.xlabel('Confidence Score')
        plt.ylabel('Count')
        plt.savefig('visualizations/confidence_distribution.png')
        plt.close()
        
        # Calculate average confidence
        avg_confidence = sum(filtered_confidence_scores) / len(filtered_confidence_scores) if filtered_confidence_scores else None
    else:
        avg_confidence = None
    
    # Return all metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'avg_confidence': avg_confidence
    }
    
    return metrics

def dynamic_threshold_adjustment(confidence_value, history=None):
    """
    Dynamically adjust the conversation threshold based on historical performance.
    
    Args:
        confidence_value: Current confidence value
        history: Optional list of historical confidence values and their outcomes
        
    Returns:
        Boolean indicating whether to trigger conversation
    """
    # Base threshold
    BASE_THRESHOLD = 0.85
    
    # If no history is provided, use the base threshold
    if not history:
        return confidence_value < BASE_THRESHOLD
    
    # Calculate adaptive threshold based on history
    correct_predictions = [h for h in history if h['correct']]
    if correct_predictions:
        avg_confidence_correct = sum([h['confidence'] for h in correct_predictions]) / len(correct_predictions)
        # Adjust threshold based on historical performance
        adaptive_threshold = (BASE_THRESHOLD + avg_confidence_correct) / 2
        return confidence_value < adaptive_threshold
    else:
        return confidence_value < BASE_THRESHOLD

def multi_round_conversation(client, tweet, initial_classification, initial_confidence, initial_reasoning, 
                            system_initiator, system_critic, deployment1, deployment2, 
                            max_rounds=3, target_confidence=0.9):
    """
    Conduct multiple rounds of conversation between initiator and critic models until
    target confidence is reached or max rounds are completed.
    
    Args:
        client: OpenAI client
        tweet: The tweet text
        initial_classification: Initial classification (0 or 1)
        initial_confidence: Initial confidence value
        initial_reasoning: Initial reasoning
        system_initiator: System prompt for initiator
        system_critic: System prompt for critic
        deployment1: Critic model deployment
        deployment2: Initiator model deployment
        max_rounds: Maximum number of conversation rounds
        target_confidence: Target confidence threshold to stop conversation
        
    Returns:
        final_classification, final_confidence, final_reasoning, conversation_transcript
    """
    # Initialize with initial values
    current_classification = initial_classification
    current_confidence = initial_confidence
    current_reasoning = initial_reasoning
    conversation_transcript = f"Initiator initial response: {initial_classification}\n{initial_confidence}\n{initial_reasoning}"
    
    # Track conversation rounds
    round_count = 0
    
    # Continue conversation until target confidence is reached or max rounds completed
    while round_count < max_rounds:
        try:
            # Try to convert confidence to float for comparison
            confidence_value = float(current_confidence)
            if confidence_value >= target_confidence:
                conversation_transcript += f"\nTarget confidence of {target_confidence} reached after {round_count} rounds."
                break
        except (ValueError, TypeError):
            # If confidence can't be converted to float, continue with next round
            pass
            
        round_count += 1
        
        # Critic conversation
        critic_system_message = {"role": "system", "content": system_critic}
        critic_user_message = {"role": "user", "content": (
            f"Review the following classification for the tweet:\n"
            f"Tweet: {tweet}\n"
            f"Classification: {current_classification}\n"
            f"Confidence: {current_confidence}\n"
            f"Reasoning: {current_reasoning}\n"
            f"Round {round_count} of conversation. Please provide your critique and suggestions."
        )}
        critic_messages = [critic_system_message, critic_user_message]
        
        critic_response = client.chat.completions.create(
            model=deployment1,
            messages=critic_messages,
            max_tokens=300,  # Increased to allow for more detailed critique
            temperature=0.5,
            top_p=0.5
        )
        critic_answer = critic_response.choices[0].message.content.strip()
        conversation_transcript += f"\nRound {round_count} Critic response: {critic_answer}"
        
        # Initiator follow-up conversation
        followup_user_message = {"role": "user", "content": (
            f"Critic feedback (Round {round_count}): {critic_answer}\n"
            "Please provide an updated classification, confidence, and reasoning in the same format."
        )}
        system_message = {"role": "system", "content": system_initiator}
        followup_messages = [system_message, followup_user_message]
        
        followup_response = client.chat.completions.create(
            model=deployment2,
            messages=followup_messages,
            max_tokens=150,  # Increased for more detailed reasoning
            temperature=0.5,
            top_p=0.5
        )
        followup_answer = followup_response.choices[0].message.content.strip()
        conversation_transcript += f"\nRound {round_count} Initiator follow-up: {followup_answer}"
        
        # Parse follow-up response if possible
        followup_parts = followup_answer.split('\n')
        if len(followup_parts) >= 3:
            current_classification = followup_parts[0].strip()
            current_confidence = followup_parts[1].strip()
            current_reasoning = '\n'.join(followup_parts[2:]).strip()
            
            # Check if target confidence is reached
            try:
                if float(current_confidence) >= target_confidence:
                    conversation_transcript += f"\nTarget confidence of {target_confidence} reached after {round_count} rounds."
                    break
            except (ValueError, TypeError):
                # If confidence can't be converted to float, continue with next round
                pass
        
    # Normalize the final classification
    current_classification = normalize_classification(current_classification, current_reasoning)
    
    # Return final results
    return current_classification, current_confidence, current_reasoning, conversation_transcript

def main():
    # Read the API key from a text file
    try:
        with open("api_key.txt", "r") as file:
            api_key = file.readline().rstrip()
    except FileNotFoundError:
        print("Error: api_key.txt file not found. Please create this file with your API key.")
        return
    
    # Configuration
    CSV_FILE = "data/tweets.csv"  # Updated to use the tweets file
    OUTPUT_CSV = "classified_tweets.csv"
    MAX_TWEETS = 100  # Increased from 10 to get better statistics
    TEST_SIZE = 0.2   # 20% of data for validation
    
    # Multi-round conversation settings
    MAX_CONVERSATION_ROUNDS = 3
    TARGET_CONFIDENCE = 0.9
    
    # Initial threshold for the conversation trigger
    CONVERSATION_THRESHOLD = 0.9
    
    # System prompt for the initiator - enhanced with clearer confidence score instructions
    SYSTEM_INITIATOR = (
        '''I am a researcher interested in studying disaster response events using Graph Neural Networks (GNNs).
My goal is to analyze historical disaster events through tweets to better understand the timeliness 
and effectiveness of responses to specific event types. Each tweet I receive includes a disaster-related hashtag, 
but not all tweets are relevant to my research objective. I want a model to classify tweets as either: 1 (Useful for disaster response analysis), 
or 0 (Not useful for disaster response analysis). 

Consider these guidelines for classification:
- Tweets about actual disasters, emergencies, or crises should be classified as 1
- Tweets using disaster-related terms metaphorically or in non-emergency contexts should be classified as 0
- News reports about disasters should be classified as 1
- Jokes, memes, or casual mentions using disaster terminology should be classified as 0

IMPORTANT ABOUT CONFIDENCE SCORES:
- Provide a confidence score between 0 and 1 that represents how certain you are about your classification
- A score close to 1 means you are very confident in your decision (whether that decision is 0 or 1)
- A score close to 0.5 means you are uncertain about your classification
- A score close to 0 should never be used (always express your confidence level from 0.5 to 1)

Return your response in exactly this format:
1. First line: ONLY "0" or "1" (your classification)
2. Second line: Your confidence score as a decimal between 0.5 and 1 (e.g., 0.85)
3. Third line and beyond: Your reasoning for the classification

Example for a disaster tweet:
1
0.95
This tweet describes an actual emergency evacuation due to flooding.

Example for a non-disaster tweet:
0
0.9
This tweet uses disaster terminology metaphorically with no actual emergency.'''
    )
    
    # System prompt for the critic - enhanced with clearer confidence score instructions
    SYSTEM_CRITIC = (
        "You are a discerning critic with expertise in disaster response tweet analysis. "
        "Review the following classification, its confidence, and reasoning. "
        "Consider these specific aspects in your critique:\n"
        "1. Is the tweet actually about a real disaster or emergency situation?\n"
        "2. Does the tweet contain actionable information useful for disaster response?\n"
        "3. Is the confidence level appropriate given the content of the tweet?\n"
        "4. Are there any ambiguities or contextual clues that might have been missed?\n\n"
        "IMPORTANT ABOUT CONFIDENCE SCORES:\n"
        "- A confidence score represents how certain the classifier is about their decision\n"
        "- High confidence (close to 1) means high certainty in the classification (whether 0 or 1)\n"
        "- Low confidence (close to 0) means uncertainty about the classification\n"
        "- Suggest appropriate confidence levels in your critique\n\n"
        "Provide a detailed critique and, if appropriate, suggest an alternative classification along with improved reasoning.\n"
        "keep response short"
    )
    
    conversation_history = []  # For display and record
    confidence_history = []    # For threshold adjustment
    
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
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    train_df, val_df, original_df = load_and_preprocess_data(CSV_FILE, MAX_TWEETS, TEST_SIZE)
    
    if train_df is None:
        print(f"Error: Could not load data from {CSV_FILE}")
        return
    
    # Visualize the data
    print("Creating data visualizations...")
    visualize_data(original_df)
    
    results = []
    true_labels = []
    predicted_labels = []
    confidence_scores = []
    tweet_count = 0
    
    # Process tweets from the validation set if available, otherwise use the training set
    df_to_process = val_df if val_df is not None else train_df
    
    print(f"Processing {len(df_to_process)} tweets...")
    
    # Process each tweet
    for index, row in df_to_process.iterrows():
        # Get the tweet text and preprocess it
        tweet = row['processed_text']
        
        # Store the true label if available
        if 'target' in row:
            true_label = row['target']
            true_labels.append(true_label)
        else:
            true_label = None
        
        # Prepare messages for the initiator
        system_message = {"role": "system", "content": SYSTEM_INITIATOR}
        user_message = {"role": "user", "content": f"Tweet: {tweet}"}
        messages = [system_message, user_message]
        
        try:
            # Initial classification call with the initiator model
            response = client.chat.completions.create(
                model=deployment2,
                messages=messages,
                max_tokens=150,  # Increased from 50 to allow for more detailed reasoning
                temperature=0.5,
                top_p=0.5
            )
            answer = response.choices[0].message.content.strip()
            # Expected answer format: "classification\nconfidence\nreasoning"
            parts = answer.split('\n')
            if len(parts) >= 3:
                classification = parts[0].strip()
                confidence = parts[1].strip()
                reasoning = '\n'.join(parts[2:]).strip()  # Join all remaining parts as reasoning
            else:
                classification = "N/A"
                confidence = "0"  # Default confidence
                reasoning = answer  # Use the whole answer as reasoning
            
            # Normalize the classification to ensure it's "0" or "1"
            classification = normalize_classification(classification, reasoning)
            
            # Store the predicted label and confidence score
            predicted_labels.append(classification)
            
            # Convert confidence to float for comparison if possible
            try:
                confidence_value = float(confidence)
                # Ensure confidence is between 0.5 and 1
                confidence_value = max(0.5, min(1.0, confidence_value))
                confidence_scores.append(confidence_value)
            except (ValueError, TypeError):
                confidence_value = 0.5  # Default to medium confidence
                confidence_scores.append(confidence_value)
            
            # Check if the prediction is correct (if true label is available)
            if true_label is not None:
                try:
                    is_correct = str(true_label) == classification
                    confidence_history.append({
                        'confidence': confidence_value,
                        'correct': is_correct
                    })
                except:
                    # Skip if there's an issue with comparison
                    pass
            
            # Dynamically determine if conversation should be triggered
            should_converse = dynamic_threshold_adjustment(confidence_value, confidence_history)
            
            # If confidence is low or dynamic threshold suggests conversation, trigger multi-round conversation
            if should_converse:
                # Multi-round conversation between models
                updated_classification, updated_confidence, updated_reasoning, conversation_transcript = multi_round_conversation(
                    client=client,
                    tweet=tweet,
                    initial_classification=classification,
                    initial_confidence=confidence,
                    initial_reasoning=reasoning,
                    system_initiator=SYSTEM_INITIATOR,
                    system_critic=SYSTEM_CRITIC,
                    deployment1=deployment1,
                    deployment2=deployment2,
                    max_rounds=MAX_CONVERSATION_ROUNDS,
                    target_confidence=TARGET_CONFIDENCE
                )
                
                # Update the predicted label and confidence score
                predicted_labels[-1] = updated_classification
                try:
                    updated_confidence_value = float(updated_confidence)
                    # Ensure confidence is between 0.5 and 1
                    updated_confidence_value = max(0.5, min(1.0, updated_confidence_value))
                    confidence_scores[-1] = updated_confidence_value
                except (ValueError, TypeError):
                    pass
                
                # Update classification, confidence, and reasoning for results
                classification = updated_classification
                confidence = updated_confidence
                reasoning = updated_reasoning
                
                # Update confidence history if true label is available
                if true_label is not None:
                    try:
                        is_correct = str(true_label) == updated_classification
                        confidence_history[-1] = {
                            'confidence': updated_confidence_value,
                            'correct': is_correct
                        }
                    except:
                        # Skip if there's an issue with comparison
                        pass
            else:
                # If no conversation was triggered, create a simple transcript
                conversation_transcript = f"Initiator response: {answer}"
            
            results.append({
                "tweet": tweet,
                "original_tweet": row['text'],
                "classification": classification,
                "confidence": confidence,
                "reasoning": reasoning,
                "conversation": conversation_transcript,
                "true_label": true_label
            })
            print(f"Processed tweet {tweet_count + 1}: {tweet[:50]}...")
        except Exception as e:
            print(f"Error processing tweet: {tweet[:50]}...\nError: {str(e)}")
        tweet_count += 1
    
    # Evaluate model performance if true labels are available
    if true_labels:
        print("\nEvaluating model performance...")
        metrics = evaluate_model(true_labels, predicted_labels, confidence_scores)
        
        print("\nModel Performance Metrics:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        if metrics['avg_confidence'] is not None:
            print(f"Average Confidence: {metrics['avg_confidence']:.4f}")
        
        # Add metrics to results for saving
        metrics_result = {
            "tweet": "METRICS",
            "original_tweet": "METRICS",
            "classification": "METRICS",
            "confidence": "METRICS",
            "reasoning": f"Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1 Score: {metrics['f1_score']:.4f}",
            "conversation": "",
            "true_label": ""
        }
        results.append(metrics_result)
    
    # Write the results to a new CSV file with updated fieldnames
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as out_file:
        fieldnames = ["tweet", "original_tweet", "classification", "confidence", "reasoning", "conversation", "true_label"]
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Classification complete. Results saved to {OUTPUT_CSV}")
    
    # Generate a summary report
    with open("classification_summary.txt", "w", encoding="utf-8") as summary_file:
        summary_file.write("TWEET CLASSIFICATION SUMMARY\n")
        summary_file.write("==========================\n\n")
        
        summary_file.write(f"Total tweets processed: {tweet_count}\n")
        if true_labels:
            summary_file.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            summary_file.write(f"Precision: {metrics['precision']:.4f}\n")
            summary_file.write(f"Recall: {metrics['recall']:.4f}\n")
            summary_file.write(f"F1 Score: {metrics['f1_score']:.4f}\n")
            if metrics['avg_confidence'] is not None:
                summary_file.write(f"Average Confidence: {metrics['avg_confidence']:.4f}\n")
            
            summary_file.write("\nConfusion Matrix:\n")
            summary_file.write("                 Predicted Non-Disaster  Predicted Disaster\n")
            summary_file.write(f"Actual Non-Disaster       {metrics['confusion_matrix'][0][0]}                  {metrics['confusion_matrix'][0][1]}\n")
            summary_file.write(f"Actual Disaster           {metrics['confusion_matrix'][1][0]}                  {metrics['confusion_matrix'][1][1]}\n")
        
        summary_file.write("\nConversation Statistics:\n")
        conversations_triggered = sum(1 for r in results if "Critic response:" in r.get("conversation", ""))
        if tweet_count > 0:
            summary_file.write(f"Conversations triggered: {conversations_triggered} ({conversations_triggered/tweet_count*100:.2f}%)\n")
        else:
            summary_file.write("No tweets were processed.\n")
        
        # Count multi-round conversations
        multi_round_conversations = sum(1 for r in results if "Round 2" in r.get("conversation", ""))
        if conversations_triggered > 0:
            summary_file.write(f"Multi-round conversations: {multi_round_conversations} ({multi_round_conversations/conversations_triggered*100:.2f}% of conversations)\n")
        else:
            summary_file.write("No multi-round conversation metrics can be calculated.\n")
        
        # Add information about target confidence achievement
        target_confidence_reached = sum(1 for r in results if f"Target confidence of {TARGET_CONFIDENCE} reached" in r.get("conversation", ""))
        if conversations_triggered > 0:
            summary_file.write(f"Conversations reaching target confidence: {target_confidence_reached} ({target_confidence_reached/conversations_triggered*100:.2f}% of conversations)\n")
        else:
            summary_file.write("No conversations were triggered, so target confidence metrics cannot be calculated.\n")
        
        # Add information about visualizations
        summary_file.write("\nVisualizations:\n")
        summary_file.write("- Tweet length distribution: visualizations/tweet_length_distribution.png\n")
        summary_file.write("- Word count distribution: visualizations/word_count_distribution.png\n")
        if true_labels:
            summary_file.write("- Target distribution: visualizations/target_distribution.png\n")
            summary_file.write("- Confusion matrix: visualizations/confusion_matrix.png\n")
            summary_file.write("- Confidence distribution: visualizations/confidence_distribution.png\n")
        if 'keyword' in original_df.columns:
            summary_file.write("- Top keywords: visualizations/top_keywords.png\n")
    
    print("Summary report generated: classification_summary.txt")

if __name__ == "__main__":
    main()
