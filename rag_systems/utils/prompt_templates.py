import copy

default_prompt = messages = [
    {
        "role": "system",
        "content": "You are GoodBot, You are a helpful assistant.",
    },
    {
        "role": "user",
        "content": ""
    },
    {
        "role": "assistant",
        "content": ""

    }

]

QA_generation_prompt_sys = """Your task is to write a factoid question and an answer given a context.
Your factoid question should be answerable with a specific, concise piece of factual information from the context.
Your factoid question should be formulated in the same style as questions users could ask in a search engine.
This means that your factoid question MUST NOT mention something like "according to the passage" or "context".

Provide your answer as follows:

Output:::
Factoid question: (your factoid question)
Answer: (your answer to the factoid question)

Now here is the context.

"""

question_groundedness_critique_prompt_sys = """
You will be given a context and a question.
Your task is to provide a 'total rating' scoring how well one can answer the given question unambiguously with the given context.
Give your answer on a scale of 1 to 5, where 1 means that the question is not answerable at all given the context, and 5 means that the question is clearly and unambiguously answerable with the context.

Provide your answer as follows:

Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here are the question and context.
"""

question_relevance_critique_prompt_sys = """
You will be given a question.
Your task is to provide a 'total rating' representing how useful this question can be to machine learning developers building NLP applications with the Hugging Face ecosystem.
Give your answer on a scale of 1 to 5, where 1 means that the question is not useful at all, and 5 means that the question is extremely useful.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here is the question.
"""


question_standalone_critique_prompt_sys = """
You will be given a question.
Your task is to provide a 'total rating' representing how context-independant this question is.
Give your answer on a scale of 1 to 5, where 1 means that the question depends on additional information to be understood, and 5 means that the question makes sense by itself.
For instance, if the question refers to a particular setting, like 'in the context' or 'in the document', the rating must be 1.
The questions can contain obscure technical nouns or acronyms like Gradio, Hub, Hugging Face or Space and still be a 5: it must simply be clear to an operator with access to documentation what the question is about.

For instance, "What is the name of the checkpoint from which the ViT model is imported?" should receive a 1, since there is an implicit mention of a context, thus the question is not independant from the context.

Provide your answer as follows:

Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here is the question.
"""


rag_prompt_sys = """Using the information contained in the context,
give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
Provide the number of the source document when relevant.
If the answer cannot be deduced from the context, do not give an answer
"""


QA_generation_prompt = copy.deepcopy(default_prompt)
QA_generation_prompt[0].update({"content": QA_generation_prompt_sys})


question_groundedness_critique_prompt = copy.deepcopy(default_prompt)
question_groundedness_critique_prompt[0].update(
    {"content": question_groundedness_critique_prompt_sys})

question_relevance_critique_prompt = copy.deepcopy(default_prompt)
question_relevance_critique_prompt[0].update(
    {"content": question_relevance_critique_prompt_sys})

question_standalone_critique_prompt = copy.deepcopy(default_prompt)
question_standalone_critique_prompt[0].update(
    {"content": question_standalone_critique_prompt_sys})

rag_prompt = copy.deepcopy(default_prompt)

rag_prompt[0].update(
    {"content": rag_prompt_sys})
