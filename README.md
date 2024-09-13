# langchain-ragie

Ragie (https://ragie.ai) integration for LangChain

## Install

```bash
pip install langchain-ragie
```

## Usage

If you need asyncio, see [this example.](https://github.com/ragieai/langchain-ragie/blob/main/examples/async.py).

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import langchain_ragie

template = """Answer the question based only on the following context:

{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI()
retriever = langchain_ragie.RagieRetriever()


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

response = chain.invoke("What do the besties think about Davos?")

print(response)

```

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.
