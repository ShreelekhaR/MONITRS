import json
import os
import numpy as np
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import re
import time
import argparse
import google.generativeai as genai

def ask_gemini(question, predicted, ground_truth, model):
    """
    Ask Gemini to be LLM-as-Judge for satellite image QA evaluation
    
    Args:
        question: The original question
        predicted: The model's predicted answer
        ground_truth: The ground truth answer
        model: The Gemini model to use
        
    Returns:
        Dictionary with scores
    """
    # Prepare the prompt for Gemini
    prompt = f"""You are evaluating an answer to a satellite imagery question. You will be presented with:
1. A question about satellite imagery
2. A ground truth expert answer
3. A predicted answer to evaluate

Please score the predicted answer on the following criteria (0-5 scale):

1. FACTUAL ACCURACY (0-5):
   - 0: Completely contradicts ground truth
   - 5: Perfectly aligned with ground truth
   
2. COMPLETENESS (0-5):
   - 0: Misses all key points in ground truth
   - 5: Covers all key points in ground truth
   
3. SPECIFICITY (0-5):
   - 0: Extremely vague compared to ground truth
   - 5: As specific or more specific than ground truth
   
4. VISUAL EVIDENCE UTILIZATION (0-5):
   - 0: No reference to visual elements unlike ground truth
   - 5: References visual evidence as well as ground truth
   
5. UNCERTAINTY HANDLING (0-5):
   - 0: Expresses inappropriate certainty/uncertainty
   - 5: Expresses appropriate uncertainty like ground truth

For each criterion, provide:
- Score (0-5)
- Brief justification explaining your reasoning

Think step by step before scoring. First analyze the ground truth to identify key points and characteristics. Then methodically compare the predicted answer to these benchmarks.

After providing scores for each criterion, also provide an OVERALL SCORE (0-5) that represents your holistic evaluation.

Return in the following format:
SCORE: <score>/5
FACTUAL ACCURACY: <factual_accuracy>/5
COMPLETENESS: <completeness>/5
SPECIFICITY: <specificity>/5
VISUAL EVIDENCE UTILIZATION: <visual_evidence>/5
UNCERTAINTY HANDLING: <uncertainty_handling>/5
OVERALL SCORE: <overall_score>/5

Nothing more.

QUESTION: {question}
GROUND TRUTH: {ground_truth}
PREDICTED ANSWER: {predicted}

EVALUATION:"""

    # Call Gemini API
    try:
        response = model.generate_content(prompt)
        response_text = response.text
    except Exception as e:
        print(f"Error calling Gemini API: {str(e)}")
        # Return default scores in case of API error
        return {
            "factual_accuracy": 0,
            "completeness": 0,
            "specificity": 0,
            "visual_evidence": 0,
            "uncertainty_handling": 0,
            "overall_score": 0
        }
    
    # Parse the evaluation response
    return parse_evaluation(response_text)

def parse_evaluation(evaluation_text):
    """
    Parse the evaluation text from Gemini to extract scores
    
    Args:
        evaluation_text: Raw evaluation text from Gemini
        
    Returns:
        Dictionary with scores
    """
    # Define the criteria to look for
    criteria = {
        "factual_accuracy": r"FACTUAL ACCURACY:\s*(\d+)/5",
        "completeness": r"COMPLETENESS:\s*(\d+)/5",
        "specificity": r"SPECIFICITY:\s*(\d+)/5", 
        "visual_evidence": r"VISUAL EVIDENCE UTILIZATION:\s*(\d+)/5",
        "uncertainty_handling": r"UNCERTAINTY HANDLING:\s*(\d+)/5",
        "overall_score": r"OVERALL SCORE:\s*(\d+)/5"
    }
    print(f"Evaluation text: {evaluation_text}")
    # Extract scores
    scores = {}
    
    for criterion, pattern in criteria.items():
        match = re.search(pattern, evaluation_text)
        if match:
            scores[criterion] = int(match.group(1))
        else:
            scores[criterion] = 0  # Default if pattern not found

    print(f"Parsed scores: {scores}")
   
    
    return scores

def main():
    """
    Main function to evaluate QA pairs and print simple statistics
    """
    parser = argparse.ArgumentParser(description="Simple evaluation of satellite image QA using Gemini as LLM-Judge")
    parser.add_argument("--qa_json", type=str, default="qa.json",
                        help="Path to the questions JSON file")
    parser.add_argument("--answers_path", type=str, default="answers.json",
                        help="Path to the predicted answers JSON file")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--api_key", type=str, default="your_api_key_here",
                        help="Gemini API key")
    
    args = parser.parse_args()
    
    # Set up Gemini
    genai.configure(api_key=args.api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.qa_json} and {args.answers_path}...")
    qa_json = json.load(open(args.qa_json))
    answers_json = json.load(open(args.answers_path))

    # print the number of questions
    print(f"Number of questions in QA JSON: {len(qa_json)}")
    print(f"Number of questions in answers JSON: {len(answers_json)}")

    
    # Initialize results storage
    all_scores = []
    
    # Process each question
    print("Evaluating answers...")
    for qid in tqdm(list(answers_json.keys())):  # Limit to 10 for faster testing - remove slice for full evaluation
        try:
            predicted = answers_json[qid]["predicted"]
            if len(predicted) == 0:
                predicted = 'F'
            
            # Get the ground truth and question
            real_qid = qid.split("_")[0]
            
            question = qa_json[int(real_qid)]['conversations'][0]['value']
            if '>' in question:
                question = question.split('>')[1]
            ground_truth = qa_json[int(real_qid)]['conversations'][1]['value']
            
            # Get evaluation
            scores = ask_gemini(question, predicted, ground_truth, model)
            
            # Add question ID to scores
            scores["qid"] = qid
            scores["real_qid"] = real_qid
            
            # Store in list for later use
            all_scores.append(scores)
            
        except Exception as e:
            print(f"Error processing question {qid}: {str(e)}")
    
    # Create DataFrame for all scores
    scores_df = pd.DataFrame(all_scores)
    
    # Calculate average scores
    metric_columns = ["factual_accuracy", "completeness", "specificity", 
                     "visual_evidence", "uncertainty_handling", "overall_score"]
    
    avgs = {col: scores_df[col].mean() for col in metric_columns if col in scores_df.columns}
    
    # Save scores to CSV
    csv_path = os.path.join(args.output_dir, "evaluation_scores.csv")
    scores_df.to_csv(csv_path, index=False)
    print(f"Individual scores saved to {csv_path}")
    
    # Print the simple table row
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Total questions evaluated: {len(all_scores)}")
    print("\nAverage Scores (0-5 scale):")
    for metric, avg in avgs.items():
        print(f"{metric.replace('_', ' ').title()}: {avg:.2f}")
    
    # Generate a single row for a table (tab-separated)
    table_row = "\t".join([f"{avg:.2f}" for metric, avg in avgs.items()])
    
    print("\nTable Row (tab-separated):")
    print(table_row)
    
    # Also show in markdown format
    header_row = "| " + " | ".join([metric.replace("_", " ").title() for metric in avgs.keys()]) + " |"
    separator_row = "|" + "|".join(["-------" for _ in avgs.keys()]) + "|"
    data_row = "| " + " | ".join([f"{avg:.2f}" for avg in avgs.values()]) + " |"
    
    print("\nMarkdown Table Row:")
    print(header_row)
    print(separator_row)
    print(data_row)
    
    # Save summary to a text file
    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("EVALUATION RESULTS\n")
        f.write("="*50 + "\n")
        f.write(f"Total questions evaluated: {len(all_scores)}\n\n")
        f.write("Average Scores (0-5 scale):\n")
        for metric, avg in avgs.items():
            f.write(f"{metric.replace('_', ' ').title()}: {avg:.2f}\n")
        
        f.write("\nTable Row (tab-separated):\n")
        f.write(table_row + "\n\n")
        
        f.write("Markdown Table:\n")
        f.write(header_row + "\n")
        f.write(separator_row + "\n")
        f.write(data_row + "\n")
    
    print(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    main()
