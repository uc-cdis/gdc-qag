# start the mcp server with tool calling enabled
# run this script from cli

# download tool-calling template
# this one does not work
# !wget https://github.com/vllm-project/vllm/blob/main/examples/tool_chat_template_llama3.1_json.jinja
# try this one below:
# https://huggingface.co/datasets/aisi-whitebox/non_sandbagging_llama_31_8b_instruct_sec_qa_v2/raw/main/tool_chat_template_llama3.1_json.jinja

import os
import subprocess
from pathlib import Path

proj_root = Path(__file__).resolve().parent.parent


p_vllm = subprocess.Popen(
    [
        "vllm",
        "serve",
        "meta-llama/Llama-3.2-3B-Instruct",
        "--dtype",
        "float16",
        "--max-model-len",
        "8184",
        "--enforce-eager",
        "--trust-remote-code",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "llama3_json",
        "--chat-template",
        os.path.join(proj_root, "methods", "tool_chat_template_llama3.1_json.jinja"),
    ]
)
