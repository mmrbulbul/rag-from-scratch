import os

import torch
import yaml

from rag_systems.utils.prompt_templates import *

LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"


def initialize(langchain=False, huggingface=True, openai=False):
    """Load api keys"""

    with open("api.yaml", "r") as file:
        config = yaml.safe_load(file)

    if huggingface:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = config["huggingface"]
    if openai:
        os.environ["OPENAI_API_KEY"] = config["openai"]
    if langchain:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = LANGCHAIN_ENDPOINT
        os.environ["LANGCHAIN_API_KEY"] = config["langchain"]


def supports_flash_attention(device_id):
    """
    Check if a GPU supports FlashAttention.
    ref: https://github.com/huggingface/transformers/issues/28188#issuecomment-1906901375
    """
    major, minor = torch.cuda.get_device_capability(device_id)

    # Check if the GPU architecture is Ampere (SM 8.x) or newer (SM 9.0)
    is_sm8x = major == 8 and minor >= 0
    is_sm90 = major == 9 and minor == 0

    return is_sm8x or is_sm90


def get_config(filename="config.yaml"):
    with open(filename, "r") as file:
        config = yaml.safe_load(file)
    return config


def call_llm(llm_agent, prompt):
    """_summary_

    Args:
        llm_agent : Hugging face pipeline
        prompt : prompt to generate
    """

    output = llm_agent(prompt)
    return output[0]['generated_text']


def create_prompt(prompt_template=default_prompt,
                  system="", question="", context=""):

    if system != "":
        prompt_template[0].update({"content": system})
    if question != "":
        prompt_template[1].update({"content": question})
    if context != "":
        prompt_template[2].update({"content": context})

    return prompt_template
