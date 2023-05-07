import os
import openai
import pandas as pd
from collections import defaultdict
import numpy as np


def read_questions_from_file(file_path):
    all_questions = []
    all_answers = []
    with open(file_path, "r") as file:
        lines = file.readlines()
        questions, answers = lines[::2], lines[1::2]
        assert len(questions) == len(answers)
        for q, a in zip(questions, answers):
            all_questions.append(q.split("Q: ")[1].strip())
            all_answers.append(a.split("A: ")[1].strip())

    return all_questions, all_answers


def evaluate_completion_models(models, questions, answers):
    results = defaultdict(list)

    np.random.seed(42)
    # Prompt the model with a random subset of the questions (don't include these in the evaluated questions)
    prompt_indicies = np.random.choice(range(len(questions)), size=8, replace=False)
    prompt_postfix = ""
    for idx in prompt_indicies:
        prompt_postfix += f"\nQ: {questions[idx]}\nA: {answers[idx]}"

    filtered_questions_and_answers = [
        (q, a) for i, (q, a) in enumerate(zip(questions, answers)) if i not in prompt_indicies
    ]
    questions, answers = zip(*filtered_questions_and_answers)
    results["questions"] = questions
    results["gt"] = answers

    for model in models:
        for question in questions:
            prompt = f"What follows is a set of statments and then and factual answer as to whether the statment is true or not. The answers are only ever True or False.{prompt_postfix}\nQ: {question}\nA:"
            response = openai.Completion.create(
                engine=model,
                prompt=prompt,
                max_tokens=1,
                n=1,
                stop=None,
                temperature=0.0,
            )
            answer = response.choices[0].text.strip()
            assert answer in ["True", "False"]
            print(f"Model: {model}, Question: {question}, Answer: {answer}")
            results[model].append(answer)

    return results


def main():
    openai.api_key = os.getenv("OPENAI_API_KEY")
    models = [
        "ada",
        "babbage",
        "curie",
        "davinci",
        "text-ada-001",
        "text-babbage-001",
        "text-curie-001",
        "text-davinci-001",
    ]
    questions, answers = read_questions_from_file("questions/combined.txt")

    evaluation_results = evaluate_completion_models(models, questions, answers)
    df = pd.DataFrame(evaluation_results, index=None)
    df.to_csv("model_evaluation.csv", index=True)


if __name__ == "__main__":
    main()
