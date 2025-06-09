import asyncio
import operator
import os
from typing import Annotated, Literal

import fitz
from dotenv import load_dotenv
from langchain.chains.combine_documents.reduce import split_list_of_docs
from langchain.schema import Document, HumanMessage, SystemMessage
from langchain_community.callbacks.infino_callback import get_num_tokens
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from toil.core.prompt import SummaryPrompt
from toil.tools.utils import add_graph_components

load_dotenv()

gemini_llm = ChatGoogleGenerativeAI(
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    model=os.getenv("GOOGLE_MODEL_NAME"),
    temperature=0.1,
    max_tokens=8192,
    max_retries=2,
).with_config(tags=["summarize"])


# TODO: 기존 앨런과 같은 세팅을 위해 임시로 삽입한 함수이기 때문에 추후 제거되어야 함.
async def acollapse_docs(
    docs: list[Document],
    combine_document_func,
    user_query: str,
) -> Document:
    """Execute a collapse function on a set of documents and merge their metadatas.

    Args:
        docs: A list of Documents to combine.
        combine_document_func: A function that takes in a list of Documents and
            optionally addition keyword parameters and combines them into a single
            string.
        **kwargs: Arbitrary additional keyword params to pass to the
            combine_document_func.

    Returns:
        A single Document with the output of combine_document_func for the page content
            and the combined metadata's of all the input documents. All metadata values
            are strings, and where there are overlapping keys across documents the
            values are joined by ", ".
    """
    result = await combine_document_func({"context": docs, "user_query": user_query})
    combined_metadata = {k: str(v) for k, v in docs[0].metadata.items()}
    for doc in docs[1:]:
        for k, v in doc.metadata.items():
            if k in combined_metadata:
                combined_metadata[k] += f", {v}"
            else:
                combined_metadata[k] = str(v)
    return Document(page_content=result, metadata=combined_metadata)


class SummaryState(BaseModel):
    """개별 요약을 위한 state"""

    content: str | list = Field(default_factory=str)
    user_query: str = Field(default_factory=str)


class MapReduceSummarizationState(BaseModel):
    """
    contents (List[str]): 입력 문서 내용의 리스트.
    summaries (Annotated[list, operator.add]): 개별 노드들에서 생성한 모든 요약들을 하나의 리스트로 결합. (map_summaries에서 시작된 여러 generate_summary들)
    collapsed_summaries (List[Document]): 요약된 문서들의 리스트.
    final_summary (str): 최종 요약.
    """

    contents: list[str] = Field(default_factory=list)
    summaries: Annotated[list, operator.add] = Field(default_factory=list)
    collapsed_summaries: list[Document] = Field(default_factory=list)
    final_summary: str = Field(default_factory=str)
    user_query: str = Field(default_factory=str)


class MapReduceSummarizationSubgraph:
    llm: BaseLanguageModel
    graph: CompiledStateGraph
    text_splitter: TextSplitter

    def __init__(self, llm, graph, text_splitter, chunk_size):
        self.llm = llm
        self.graph = graph
        self.text_splitter = text_splitter
        self.chunk_size = chunk_size

    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, chunk_size: int = 4096):
        map_prompt = SummaryPrompt().get_prompt_template()
        reduce_prompt = SummaryPrompt().get_prompt_template()

        map_chain = map_prompt | llm | StrOutputParser()
        reduce_chain = reduce_prompt | llm | StrOutputParser()

        graph = cls._create_graph(llm, map_chain, reduce_chain, chunk_size)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=0,
            length_function=lambda x: get_num_tokens(
                x, "gpt-4o-mini"
            ),  # gemini 토큰 수 측정함수는 시간이 오래 소요되어 gpt-4o-mini 토큰 수 사용
        )

        return cls(
            llm=llm, graph=graph, text_splitter=text_splitter, chunk_size=chunk_size
        )

    @staticmethod
    def _create_graph(llm, map_chain, reduce_chain, chunk_size) -> CompiledStateGraph:
        async def generate_summary(state: SummaryState):
            """주어진 문서를 요약하는 비동기 함수."""
            response = await map_chain.ainvoke(
                {"context": state.content, "user_query": state.user_query}
            )
            return {"summaries": [response]}

        def map_summaries(state: MapReduceSummarizationState):
            """문서들에 대해 map할 로직을 정의하는 함수."""
            return [
                Send(
                    "generate_summary",
                    SummaryState(content=content, user_query=state.user_query),
                )
                for content in state.contents
            ]

        def length_function(documents: list[Document]) -> int:
            """입력된 내용의 토큰 수를 가져오는 함수. 이걸로 충분히 reduce되었는지 봄"""
            return sum(llm.get_num_tokens(doc.page_content) for doc in documents)

        def collect_summaries(state: MapReduceSummarizationState):
            """요약을 모아서 collapsed_summaries에 저장하는 함수."""
            return {
                "collapsed_summaries": [
                    Document(summary) for summary in state.summaries
                ]
            }

        async def generate_final_summary(state: MapReduceSummarizationState):
            """collapsed_summaries에 저장된 요약들을 읽어 최종 요약을 생성하는 비동기 함수."""
            response = await reduce_chain.ainvoke(
                {"context": state.collapsed_summaries, "user_query": state.user_query}
            )
            return {"final_summary": response}

        async def collapse_summaries(state: MapReduceSummarizationState):
            """여기는 collapsed_summaries에 저장된 요약들이 제한 길이를 넘을 때 오는 node.

            collapsed_summaries에 저장된 요약들을 읽어 토큰 수에 따라 분할하고, 각 분할된 리스트를 비동기적으로 병합하여 최종 요약 리스트를 생성하는 함수.
            """
            doc_lists = split_list_of_docs(
                state.collapsed_summaries, length_function, chunk_size
            )
            results = []
            for doc_list in doc_lists:
                results.append(
                    await acollapse_docs(
                        doc_list, reduce_chain.ainvoke, user_query=state.user_query
                    )
                )

            return {"collapsed_summaries": results}

        def should_collapse(
            state: MapReduceSummarizationState,
        ) -> Literal["collapse_summaries", "generate_final_summary"]:
            num_tokens = length_function(state.collapsed_summaries)
            if num_tokens > chunk_size:
                return "collapse_summaries"
            else:
                return "generate_final_summary"

        graph_builder = StateGraph(MapReduceSummarizationState)
        nodes = [
            ("generate_summary", generate_summary),
            ("collect_summaries", collect_summaries),
            ("generate_final_summary", generate_final_summary),
            ("collapse_summaries", collapse_summaries),
        ]
        edges = [
            ("generate_summary", "collect_summaries"),
            ("generate_final_summary", END),
        ]
        edges_with_conditions = [
            (START, map_summaries, ["generate_summary"]),
            ("collect_summaries", should_collapse),
            ("collapse_summaries", should_collapse),
        ]

        graph_builder = add_graph_components(
            graph_builder, nodes, edges, edges_with_conditions
        )
        return graph_builder.compile(checkpointer=False)

    async def arun(self, document: Document, user_query: str = "") -> Document:
        try:
            if document.metadata.get("type") == "pdf":
                messages = SummaryPrompt().format_messages(
                    user_query=user_query,
                    file_type="application/pdf",
                    media_file=document.page_content,
                )
                result = await gemini_llm.ainvoke(messages)
                document = Document(
                    page_content=result.content, metadata=document.metadata
                )

            if self.llm.get_num_tokens(document.page_content) < self.chunk_size:
                return document

            splitted = self.text_splitter.create_documents([document.page_content])
            summary = await self.graph.ainvoke(
                {
                    "contents": [doc.page_content for doc in splitted],
                    "user_query": user_query,
                }
            )

            return Document(
                page_content=summary["final_summary"], metadata=document.metadata
            )
        except Exception as e:
            print(f"Error: {e}")
            url = document.metadata.get("source", "unknown")
            return Document(
                page_content=f"Error occurred during document summarization. Providing snippet only: {document.page_content[:self.chunk_size]}",
                metadata={
                    "error": type(e).__name__,
                    "error_content": str(e),
                    "source": url,
                },
            )

    async def abatch(
        self, documents: list[Document], user_query: str = ""
    ) -> list[dict]:  # list[Document]
        async def summarize_doc(doc: Document, user_query: str = ""):
            if "error" not in doc.metadata:
                summary = await self.arun(doc, user_query)
                return summary.model_dump()
            else:
                return doc.model_dump()

        results = await asyncio.gather(
            *[summarize_doc(doc, user_query) for doc in documents]
        )
        return results
