#!/usr/bin/env python3
import sys

import uvicorn
from fastapi import FastAPI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langserve import add_routes

from gdc_pipeline import execute_pipeline, setup_args


def create_prompt():
    template = """You are an assistant with access to a tool named `run_pipeline_on_file`. Your job is to run this tool on a file questions.csv.

    {tools}

    Thought: I should call the tool {tool_names}
    Action: run_pipeline_on_file
    Action Input: questions.csv

    Begin!

    Question: {input}
    {agent_scratchpad}
    """.strip()

    prompt = PromptTemplate.from_template(template)
    return prompt


def wrapped_execute_pipeline(input_file: str):
    sys.argv = ["prog", "--input-file", "dummy.csv"]
    args = setup_args()
    print(f"Tool received: {input_file}")
    return execute_pipeline(
        input_file=input_file,
        intent_model_path=args.intent_model_path,
        path_to_gdc_genes_mutations=args.path_to_gdc_genes_mutations_file,
        hf_token_path=args.hf_token_path,
    )


mcp_tool = Tool(
    name="run_pipeline_on_file",
    func=wrapped_execute_pipeline,
    description="Run the pipeline on a CSV file. Input is a file 'questions.csv'",
)


agent_llm = FakeListLLM(
    responses=[
        """Thought: I should use the tool
    Action: run_pipeline_on_file
    Action Input: questions.csv"""
    ]
)


prompt = create_prompt()

agent = create_react_agent(llm=agent_llm, tools=[mcp_tool], prompt=prompt)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=[mcp_tool],
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=1,
)

app = FastAPI()
add_routes(app, agent_executor, path="/agent")
