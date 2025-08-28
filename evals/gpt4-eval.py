#!/usr/bin/env python3

# testing batch API from https://github.com/openai/openai-cookbook/blob/main/examples/batch_processing.ipynb
# pip install openai --upgrade
# pip install python-dotenv
# note: use this code only for small scale tests
# for larger files, convert to jsonl and upload to batch API
# see notebook in notebooks for an example

import json
import os

import openai
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

tqdm.pandas()


def initialize_openai_client():
    load_dotenv(".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key not loaded")

    # Initializing OpenAI client - see https://platform.openai.com/docs/quickstart?context=python
    client = openai.OpenAI()
    return client


def define_general_prompt():
    general_prompt = """
    
    'Only use results from the genomic data commons in your response and provide frequencies \
     as a percentage in the result. Report the result in the following output JSON format, strictly using \
     the structure "The final answer is: <frequency %>", followed by top references to publications from which you \
     obtained your response:

    {
        result: The final answer is: <frequency %>
        references: <list of references>
    }

    """
    return general_prompt


def get_gpt4_response(client, general_prompt, question):
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        temperature=0.0,
        seed=2000,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": general_prompt},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content


def run_eval(row, general_prompt, client):
    question = row["questions"]
    response = json.loads(get_gpt4_response(client, general_prompt, question))
    gpt4_base_output = response["result"]
    gpt4_references = response["references"]

    return pd.Series([gpt4_base_output, gpt4_references, response])


def main():
    print("init client")
    client = initialize_openai_client()
    print("openai client initialized")
    # check model availability
    print(client.models.list())

    eval_dataset = pd.read_csv("csvs/tests.csv")
    # eval_dataset = eval_dataset.head(n=2)
    general_prompt = define_general_prompt()

    eval_dataset[["gpt4_base_output", "gpt4_references", "response"]] = (
        eval_dataset.progress_apply(
            lambda x: run_eval(x, general_prompt, client), axis=1
        )
    )

    print("done generation questions , dumping to CSV")
    eval_dataset.to_csv("csvs/gpt4.tests.results.csv")


if __name__ == "__main__":
    main()
