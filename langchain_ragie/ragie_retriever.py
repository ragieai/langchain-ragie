from typing import Annotated, Any, Dict, List, Optional
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
import pydantic
from ragie import Ragie, RetrieveParams


def _to_documents(retrieval) -> List[Document]:
    return [
        Document(
            id=chunk.document_id,
            metadata=chunk.document_metadata,
            page_content=chunk.text,
        )
        for chunk in retrieval.scored_chunks
    ]


class RagieRetriever(BaseRetriever):
    api_key: str
    r"""The API key for Ragie."""
    top_k: Optional[int]
    r"""The maximum number of chunks to return. Defaults to 8."""
    filter: Annotated[Optional[Dict[str, Any]], pydantic.Field(alias="filter")]
    r"""The metadata search filter on documents. Returns chunks only from documents which match the filter. The following filter operators are supported: $eq - Equal to (number, string, boolean), $ne - Not equal to (number, string, boolean), $gt - Greater than (number), $gte - Greater than or equal to (number), $lt - Less than (number), $lte - Less than or equal to (number), $in - In array (string or number), $nin - Not in array (string or number). The operators can be combined with AND and OR. Read [Metadata & Filters guide](https://docs.ragie.ai/docs/metadata-filters) for more details and examples."""
    rerank: Optional[bool]
    r"""Reranks the chunks for semantic relevancy post cosine similarity. Will be slower but returns a subset of highly relevant chunks. Best for reducing hallucinations and improving accuracy for LLM generation."""
    max_chunks_per_document: Optional[int]
    r"""Maximum number of chunks to retrieve per document. Use this to increase the number of documents the final chunks are retreived from. This feature is in beta and may change in the future."""

    def _to_retrieve_params(self, query: str) -> RetrieveParams:
        kwargs = {}
        if self.top_k is not None:
            kwargs["top_k"] = self.top_k
        if self.filter is not None:
            kwargs["filter"] = self.filter
        if self.rerank is not None:
            kwargs["rerank"] = self.rerank
        if self.max_chunks_per_document is not None:
            kwargs["max_chunks_per_document"] = self.max_chunks_per_document
        return RetrieveParams(**(kwargs | {"query": query}))

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        client = Ragie(self.api_key)
        retrieval = client.retrievals.retrieve(request=self._to_retrieve_params(query))
        return _to_documents(retrieval)

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        client = Ragie(self.api_key)
        retrieval = await client.retrievals.retrieve_async(request={"query": query})
        return _to_documents(retrieval)
