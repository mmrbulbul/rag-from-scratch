import os

import yaml

LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"


def initialize(langchain=False, huggingface=True, openai=False):
    """Load api keys"""

    with open('api.yaml', 'r') as file:
        config = yaml.safe_load(file)

    if huggingface:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = config['huggingface']
    if openai:
        os.environ["OPENAI_API_KEY"] = config['openai']
    if langchain:
        os.environ["LANGCHAIN_TRACING_V2"] = 'true'
        os.environ["LANGCHAIN_ENDPOINT"] = LANGCHAIN_ENDPOINT
        os.environ["LANGCHAIN_API_KEY"] = config['langchain']
