from langchain_openai import ChatOpenAI

def get_llm(seed: int | None = None):
    """
    Returns a LangChain LLM instance connected to the local
    llama.cpp server running GPT-OSS 20B Q4_K_M.
    Uses the langchain-openai package's ChatOpenAI class, which
    properly implements bind_tools() for structured tool calling.
    The llama.cpp server exposes an OpenAI-compatible API at
    /v1/chat/completions.
    """
    extra_body: dict = {"reasoning_effort": "medium"}
    if seed is not None:
        extra_body["seed"] = seed

    llm = ChatOpenAI(
        base_url="http://127.0.0.1:8080/v1",
        api_key="not-needed",
        model="gpt-oss-20b-q4_k_m",
        temperature=1.0,
        max_tokens=4096,
        extra_body=extra_body,
    )

    return llm
