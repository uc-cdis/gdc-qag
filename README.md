# gdc-qag
This repo contains code for Query Augmented Generation on the Genomic Data Commons as described in the up-coming paper. 
To run this code, you need access to the `meta-llama/Llama-3.2-3B-Instruct` model. This is a gated model.
You also need access to a V100 GPU with 16 GB GPU RAM.

## Installation
Use the `pyproject.toml` file to install dependencies using `poetry`. You can install `poetry` using `pip3 install poetry`.
This project has been tested on Python 3.10.13. Please install this Python version, example using `pyenv install 3.10.13` and then switching to this version in your terminal using `pyenv local 3.10.13`.

To install all dependencies run `poetry install --no-root`.
Then switch to the environment using `eval $(poetry env activate)`.
If you want to install any new dependency, you can add using `poetry add <dependency>` and this will update the environment, the toml and the lock file.

To run Jupyter notebooks run the following:

```
poetry run pip install jupyter ipykernel
poetry run python -m ipykernel install --user --name venv-3.10.13 --display-name "Python Poetry venv-3.10.13
```
If using VSCode, restart it, and select `Python Poetry venv-3.10.13` in kernel under Jupyter kernel

## HF token
Once approval is obtained for the llama model, please add the token to your environment as an environment variable
`HF_TOKEN`=<token_value>

## Running the analysis

Test out an example from `csvs` directory

You can either run on an input file with questions, or on a single question

To test on an input file:

```
(gdc-qag-py3.10) aartiv@kl35-gpu-3:~/gdc-qag$ python gdc_pipeline.py --input-file csvs/baby_tests.csv
```

To test on a single question:

```
(gdc-qag-py3.10) aartiv@kl35-gpu-3:~/gdc-qag$ python gdc_pipeline.py --question "What is the co-occurence frequency of somatic homozygous deletions in CDKN2A and CDKN2B in the mesothelioma project TCGA-MESO in the genomic data commons?"
```
Results are written to `csvs` directory. This example question executes in ~20 seconds if successful. Check out results for the example csv file, `baby_tests.results.csv` to compare with your test run.

Please retry the query if there is a failure, e.g. GDC API failures, or connection aborts

## Structure of this repository and useful files

- `csvs`: contains csv files for testing
- `evals`: contains code for evaluation of `Qwen-4B` model and low level eval of `GPT-4o` model. Full evaluation of `GPT-4o` see `notebooks`
- `figs`: figures from paper
- `methods`: helper methods and utility functions for QAG

`notebooks`:  
- `plot.ipynb` contains code to reproduce figures, the data is available in the supplement
- `open_ai_batch.ipynb`: contains code for batch submission for `GPT-4o` evaluation
- `other_applications.ipynb`: contains code for testing out novel features of QAG
- `synthetic_data_query_intent_train_bert.ipynb`: code for training BERT model. Data available as a HF Dataset, refer to paper
- `llm_evaluation_dataset.ipynb`: contains code templates for creating evaluation dataset
- `survival_analysis.ipynb`: contais code for plotting survival data, data is downloaded from GDC portal
- `eval-bert-model.ipynb`: contains code for testing BERT model performance on the test set. Data available in the supplement

`gdc_pipeline.py`: this is the entrypoint code for launching QAG

Note that vllm is pinned to 0.7.2 and transformers to 4.49.0

## Hugging Face model and datasets
The datasets and BERT model used by QAG are available on hugging face, please refer to the paper

## Link to paper
Coming soon


