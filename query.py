import openai
import os
import sys

# Function to query OpenAI API
def query_openai_api(prompt: str, model: str = "text-davinci-002", max_tokens: int = 100) -> str:
    """Queries the OpenAI API using the given prompt.

    Args:
        prompt (str): The prompt to send to the OpenAI API.
        model (str, optional): The OpenAI model to use. Defaults to "text-davinci-002".
        max_tokens (int, optional): The maximum number of tokens in the response. Defaults to 100.

    Returns:
        str: The generated response from the OpenAI API.
    """
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=0.5,
    )

    responses = [choice.text.strip() for choice in response.choices]
    return responses


def query_openai_api_chat(prompt: str, model: str = "gpt-3.5-turbo", max_tokens: int = 100) -> str:

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{prompt}"},
        ],
    )

    response = [choice.message.content.strip() for choice in response.choices]
    return response
