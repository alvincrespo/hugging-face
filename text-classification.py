from transformers import pipeline

def my_pipe(task, model, text, **kwargs):
    return pipeline(task=task, model=model)(text, **kwargs)

# -----------------------------------------------------------------------------------------------------
# Sentiment Analysis
# -----------------------------------------------------------------------------------------------------
print("\n---Sentiment Analysis---")
# https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english
classification_model = "distilbert-base-uncased-finetuned-sst-2-english"

classification_task = "text-classification"

positive_text = "I love using transformers!"
negative_text = "I hate using transformers!"

positive_result = my_pipe(classification_task, classification_model, positive_text)
negative_result = my_pipe(classification_task, classification_model, negative_text)

print(f"{positive_result}")
print(f"{negative_result}")
# -----------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------------
# Grammar Correctness
# -----------------------------------------------------------------------------------------------------
# https://huggingface.co/abdulmatinomotoso/English_Grammar_Checker
print("\n---Grammar Correctness---")
grammar_model = "abdulmatinomotoso/English_Grammar_Checker"

grammar_task = "text-classification"

correct_text = "I love ham and cheese sandwiches!"
incorrect_text = "Me no like ham and cheese sammiches!"

positive_result = my_pipe(grammar_task, grammar_model, correct_text)
negative_result = my_pipe(grammar_task, grammar_model, incorrect_text)

print(f"{positive_result}")
print(f"{negative_result}")
# -----------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------------
# QNLI (Question Natural Language Inference)
# -----------------------------------------------------------------------------------------------------
# https://huggingface.co/cross-encoder/qnli-electra-base
print("\n---QNLI---")

qnli_model = "cross-encoder/qnli-electra-base"
qnli_task = "text-classification"

qnli_sentence = "Are tomatoes native to Italy? No, tomatoes are native to South America."

qnli_result = my_pipe(qnli_task, qnli_model, qnli_sentence)

print(f"{qnli_result}")
# -----------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------------
# Dynamic Category Assignment
# -----------------------------------------------------------------------------------------------------
# https://huggingface.co/facebook/bart-large-mnli
print("\n---Dynamic Category Assignment---")

dynamic_model = "facebook/bart-large-mnli"
dynamic_task = "zero-shot-classification"

dynamic_sentence = "Are you ready for 3D? Well, here we go! Nintendo 64 will be releasing on June 23, 1996."
categories = ["marketing", "technology", "gaming"]

dynamic_result = my_pipe(dynamic_task, dynamic_model, dynamic_sentence, candidate_labels=categories)

print(f"Top Label: {dynamic_result['labels'][0]} with score: {dynamic_result['scores'][0]}")
# -----------------------------------------------------------------------------------------------------
