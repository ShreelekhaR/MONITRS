# Read in JSON in the following format: "0_": {"predicted": "C", "ground_truth": "A", "task": "multiple_choice"}, "1_": {"predicted": "B", "ground_truth": "B", "task": "multiple_choice"}, "2_": {"predicted": "A", "ground_truth": "A", "task": "multiple_choice"}, "3_": {"predicted": "D", "ground_truth": "B", "task": "multiple_choice"}, "4_": {"predicted": "B", "ground_truth": "A", "task": "multiple_choice"}, "5_": {"predicted": "A", "ground_truth": "C", "task": "multiple_choice"}, "6_": 
import numpy as np
import json
from tqdm import tqdm
import os
import scipy.stats as stats



def calculate_accuracy_mcq(answers):
    """
    Calculate accuracy for each question based on predicted and ground truth answers.
    Args:
    answers (dict): Dictionary containing predicted and ground truth answers.
    Returns:
    dict: Accuracy percentage overall for all valid questions
    list: Boolean array of correct/incorrect predictions for McNemar's test
    """
    accurate_count = 0
    correct_predictions = []

    for qid in answers.keys():
        predicted = answers[f"{qid}"]["predicted"]
        if len(predicted) == 0:
            predicted = 'F'
        else:
            predicted = answers[f"{qid}"]["predicted"][0]
        # get the ground truth
        ground_truth = answers[f"{qid}"]["ground_truth"][0]
        # check if the predicted answer is correct ignoring case
        is_correct = predicted.lower() == ground_truth.lower()
        correct_predictions.append(is_correct)
        
        if is_correct:
            accurate_count += 1
    
    accuracy = accurate_count / len(answers) * 100
    print(f"Accuracy: {accuracy:.2f}%")
    
    return accuracy, correct_predictions

def mcnemars_test(model1_correct, model2_correct, model1_name, model2_name):
    """
    Perform McNemar's test to compare two models.
    Args:
    model1_correct (list): Boolean array indicating correct/incorrect predictions for model 1
    model2_correct (list): Boolean array indicating correct/incorrect predictions for model 2
    model1_name (str): Name of model 1
    model2_name (str): Name of model 2
    Returns:
    float: p-value from McNemar's test
    """
    # Create contingency table
    b = sum(np.logical_and(model1_correct, np.logical_not(model2_correct)))  # model1 correct, model2 incorrect
    c = sum(np.logical_and(np.logical_not(model1_correct), model2_correct))  # model1 incorrect, model2 correct
    
    # Apply McNemar's test
    # Use exact binomial test for better accuracy with small counts
    if b + c < 20:
        # p_value = stats.binom_test(b, b + c, p=0.5)
        p_value = stats.binomtest(b, b + c, p=0.5).pvalue
    else:
        # Use chi-square approximation with continuity correction
        chi2 = (abs(b - c) - 1)**2 / (b + c)
        p_value = stats.chi2.sf(chi2, 1)
    
    print(f"\nMcNemar's test results for {model1_name} vs {model2_name}:")
    print(f"Number of cases where only {model1_name} was correct: {b}")
    print(f"Number of cases where only {model2_name} was correct: {c}")
    print(f"p-value: {p_value:.6f}")
    
    if p_value < 0.001:
        print("The difference is statistically significant (p < 0.001)")
    elif p_value < 0.01:
        print("The difference is statistically significant (p < 0.01)")
    elif p_value < 0.05:
        print("The difference is statistically significant (p < 0.05)")
    else:
        print("The difference is not statistically significant (p > 0.05)")
    
    return p_value

def calculate_nlp_metrics(answers):
    """
    Calculate BLEU-1,BLEU-2,BLEU-3,BLEU-4, Meteor and Rouge-L score for each question based on predicted and ground truth answers.
    Args:
    answers (dict): Dictionary containing predicted and ground truth answers.
    Returns:
    dict: Dictionary containing BLEU, METEOR, and ROUGE-L scores for each question.
    """
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from rouge import Rouge
    import nltk
    
    # Download necessary NLTK data (only need to run once)
    try:
        nltk.data.find('wordnet')
    except LookupError:
        nltk.download('wordnet')
        nltk.download('punkt')

    # Initialize results dictionary
    results = {}
    
    # Initialize Rouge scorer
    rouge = Rouge()
    
    # Initialize BLEU smoothing function
    smoothie = SmoothingFunction().method1
    
    print("\n===== Evaluation Scores =====")
    print(f"{'Question ID':<15} {'BLEU-1':>8} {'BLEU-2':>8} {'BLEU-3':>8} {'BLEU-4':>8} {'METEOR':>8} {'ROUGE-L':>8}")
    print("-" * 70)
    
    for q_id, answer_data in answers.items():
        # Extract predicted and ground truth answers
        predicted = answer_data.get('predicted', '')
        ground_truth = answer_data.get('ground_truth', '')

    # Tokenize the answers
        predicted_tokens = nltk.word_tokenize(predicted.lower())
        ground_truth_tokens = nltk.word_tokenize(ground_truth.lower())
        
        # Calculate BLEU scores
        bleu_1 = sentence_bleu([ground_truth_tokens], predicted_tokens, 
                              weights=(1, 0, 0, 0), smoothing_function=smoothie)
        bleu_2 = sentence_bleu([ground_truth_tokens], predicted_tokens, 
                              weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
        bleu_3 = sentence_bleu([ground_truth_tokens], predicted_tokens, 
                              weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
        bleu_4 = sentence_bleu([ground_truth_tokens], predicted_tokens, 
                              weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
        
        # Calculate METEOR score
        meteor = meteor_score([ground_truth_tokens], predicted_tokens)
        
        # Calculate ROUGE scores
        # Handle empty strings to avoid ROUGE errors
        if not predicted or not ground_truth:
            rouge_scores = {'rouge-l': {'f': 0.0}}
        else:
            try:
                rouge_scores = rouge.get_scores(predicted, ground_truth)[0]
            except:
                # Fallback if ROUGE fails
                rouge_scores = {'rouge-l': {'f': 0.0}}
        
        rouge_l = rouge_scores['rouge-l']['f']
        
        # Store results
        results[q_id] = {
            'bleu-1': bleu_1,
            'bleu-2': bleu_2,
            'bleu-3': bleu_3,
            'bleu-4': bleu_4,
            'meteor': meteor,
            'rouge-l': rouge_l
        }
       
    # Print average scores
    if results:
        avg_bleu1 = sum(r['bleu-1'] for r in results.values()) / len(results)
        avg_bleu2 = sum(r['bleu-2'] for r in results.values()) / len(results)
        avg_bleu3 = sum(r['bleu-3'] for r in results.values()) / len(results)
        avg_bleu4 = sum(r['bleu-4'] for r in results.values()) / len(results)
        avg_meteor = sum(r['meteor'] for r in results.values()) / len(results)
        avg_rouge_l = sum(r['rouge-l'] for r in results.values()) / len(results)
        
        print("-" * 70)
        print(f"{'AVERAGE':^15} {avg_bleu1:>8.4f} {avg_bleu2:>8.4f} {avg_bleu3:>8.4f} {avg_bleu4:>8.4f} {avg_meteor:>8.4f} {avg_rouge_l:>8.4f}")
    
    return results

if __name__ == "__main__":
    with open("answers.json", "r") as f:
        answers_json = json.load(f)

    # # calculate accuracy
    # accuracy = calculate_accuracy_mcq(answers_json)


    # # calculate nlp metrics
    # nlp_metrics = calculate_nlp_metrics(answers_json)


    # Load results for each model
    model_files = {
        'Ours': 'Your_mcq_file_path.json',
        'TEOChat': 'Your_mcq_file_path.json',
        'VideoLLaVA': 'Your_mcq_file_path.json',
        'Gemini': 'Your_mcq_file_path.json'
    }

    # Dictionary to hold results from all models
    model_results = {}

    # Process each model's results
    for model_name, file_path in model_files.items():
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                answers = json.load(f)
                accuracy, correct_predictions = calculate_accuracy_mcq(answers)
                model_results[model_name] = {
                    'accuracy': accuracy,
                    'correct_predictions': correct_predictions
                }
                print(f"{model_name} loaded and processed.")
        else:
            print(f"Warning: File not found for {model_name}: {file_path}")


    # Ensure all models have same number of predictions
    # (Important for valid McNemar's test)
    models_to_compare = list(model_results.keys())
    if len(models_to_compare) >= 2:
        lengths = [len(model_results[model]['correct_predictions']) for model in models_to_compare]
        if len(set(lengths)) > 1:
            print("Warning: Models have different numbers of predictions.")
            print("Lengths:", lengths)
            print("McNemar's test requires the same examples to be evaluated by both models.")
        else:
            # Perform McNemar's test for all pairs of models
            for i in range(len(models_to_compare)):
                for j in range(i+1, len(models_to_compare)):
                    model1 = models_to_compare[i]
                    model2 = models_to_compare[j]
                    
                    mcnemars_test(
                        model_results[model1]['correct_predictions'],
                        model_results[model2]['correct_predictions'],
                        model1,
                        model2
                    )
    else:
        print("Need at least two models to perform McNemar's test.")