import os

def triviaQA(example):
    question = example["question"]
    context = example["search_results"]["description"]
    answer_list = example["answer"]["normalized_aliases"]
    return question, context, answer_list