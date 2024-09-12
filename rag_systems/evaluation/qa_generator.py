"""
Scripts to generate random questin and answer based on knowledge base
ref: https://huggingface.co/learn/cookbook/en/rag_evaluation
"""
import random

from tqdm import tqdm

from rag_systems.utils.prompt_templates import QA_generation_prompt
from rag_systems.utils.utils import call_llm, create_prompt


class QAGenerator:
    def __init__(self, n_quations):
        self.n_quations = n_quations  # number of qa pair to generate

    def generate_qa(self, llm_client, docs_processed):

        print(f"Generating {self.n_quations} QA couples...")

        outputs = []
        for sampled_context in tqdm(random.sample(docs_processed,
                                                  self.n_quations)):
            # Generate QA couple
            output_QA_couple = call_llm(
                llm_client,
                create_prompt(
                    prompt_template=QA_generation_prompt,
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
