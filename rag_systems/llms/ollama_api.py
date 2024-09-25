from langchain_ollama.llms import OllamaLLM


def get_model(model_name="phi3:medium-128k", temperature=0, nx_ctx=32768):
    return OllamaLLM(model=model_name, temperature=temperature, nx_ctx=nx_ctx)


if __name__ == "__main__":
    llm = get_model()
    print(llm.invoke("solve 2x+3 =7"))
