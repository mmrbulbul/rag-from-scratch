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
    "max_new_tokens": 512,
    "return_full_text": False,
    "do_sample": True,
    "temperature": 0.7,

}


messages = [
    {
        "role": "system",
        "content": "You are GoodBot, You are a helpful assistant.",
    },
    {
        "role": "user",
        "content": "who are you?"
    }

]

tokenizer.use_default_system_prompt = True
tokenizer.chat_template = "{{ bos_token }}{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + '<|end|>' }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + '<|end|>' }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + '<|end|>' }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"


llm_agent = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    **generation_args
)


# llm = HuggingFacePipeline(pipeline=pipe)


if __name__ == "__main__":
    pass
