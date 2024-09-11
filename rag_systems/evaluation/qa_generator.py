"""
Scripts to generate random questin and answer based on knowledge base
ref: https://huggingface.co/learn/cookbook/en/rag_evaluation
"""
import random

from tqdm import tqdm

QA_generation_prompt = """
Your task is to write a factoid question and an answer given a context.
Your factoid question should be answerable with a specific, concise piece of factual information from the context.
Your factoid question should be formulated in the same style as questions users could ask in a search engine.
This means that your factoid question MUST NOT mention something like "according to the passage" or "context".

Provide your answer as follows:

Output:::
Factoid question: (your factoid question)
Answer: (your answer to the factoid question)

Now here is the context.

Context: {context}\n
Output:::"""


def call_llm(llm_client, prompt):

    raise NotImplementedError


class QAGenerator:
    def __init__(self, n_quations):
        self.n_quations = n_quations  # number of qa pair to generate

    def generate_qa(self, llm_client, docs_processed):

        print(f"Generating {self.n_quations} QA couples...")

        outputs = []
        for sampled_context in tqdm(random.sample(docs_processed, self.n_quations)):
            # Generate QA couple
            output_QA_couple = call_llm(llm_client, QA_generation_prompt.format(
                context=sampled_context.page_content))
            try:
                question = output_QA_couple.split(
                    "Factoid question: ")[-1].split("Answer: ")[0]
                answer = output_QA_couple.split("Answer: ")[-1]
                assert len(answer) < 300, "Answer is too long"
                outputs.append(
                    {
                        "context": sampled_context.page_content,
                        "question": question,
                        "answer": answer,
                        "source_doc": sampled_context.metadata["source"],
                    }
                )
            except:
                continue

        return outputs
