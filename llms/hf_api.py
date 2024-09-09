"""
REF:
    1. https://huggingface.co/microsoft/Phi-3-small-128k-instruct
    2. https://huggingface.co/blog/4bit-transformers-bitsandbytes#advanced-usage
"""
import warnings

import langchain
import torch
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, pipeline)

warnings.filterwarnings('ignore')

torch.random.manual_seed(0)
langchain.debug = True

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-128k-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
    quantization_config=nf4_config,
    # attn_implementation="flash_attention_2"

)

print("model loaded\n")

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

generation_args = {
    "max_new_tokens": 600,
    "return_full_text": False,
    "do_sample": False,
    "temperature": 0.0

}

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    **generation_args
)


# llm = HuggingFacePipeline(pipeline=pipe)


# # system_template = "You are a helpful AI assistant."
messages = [
    {"role": "system", "content": "You are a helpful AI assistant. Answer only what you have been asked"},
    {"role": "user", "content": "Solve 2x + 3 = 7"}
]


# print(llm.invoke(messages))

# print("*"*100)
# print("\n")
output = pipe(messages)
print(output[0]['generated_text'])
