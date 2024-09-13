# langchain-ragie

This is a langchain integration for Ragie (https://ragie.ai)

## Install

```bash
pip install langchain-ragie
```

## Usage

If you need asyncio, see [this example.](https://github.com/ragieai/langchain-ragie/blob/main/tests/aintegration.py).

```python
import os

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
retriever = langchain_ragie.RagieRetriever(
    api_key=os.getenv("RAGIE_API_KEY"), filter={"type": "Book"}, rerank=False
)


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

response = chain.invoke("Should the United States contain China?")

print(response)

```
