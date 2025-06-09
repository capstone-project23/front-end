import base64
import os
import re
from typing import Annotated, Optional, Type

import fitz
import httpx
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun
from langchain.schema import BaseMessage, Document
from langchain_core.callbacks.manager import adispatch_custom_event
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field
from youtube_transcript_api import YouTubeTranscriptApi

from toil.tools.base import AsyncTool
from toil.tools.mixins import ContentTypeError, HTMLToMarkdownMixin, HTTPXMixin
from toil.tools.subgraphs import MapReduceSummarizationSubgraph
from toil.tools.utils import noop

LANGUAGE_CODES = ["ko", "en"]

youtube_url_pattern = re.compile(
    r"""^(?:https?:\/\/)?
    (?:www\.|m\.)?
    (?:youtube(?:-nocookie)?\.com|youtu\.be)
    \/
    (?:
        (?:watch(?:\?(?:.*&)?v=|\/))
      | (?:v|embed|e|shorts|live)\/
    )?
    ([A-Za-z0-9_-]{11})
    (?:[?&][^#]*)?
    (?:\#.*)?
    $""",
    re.VERBOSE,
)


class HTTPURLArgs(BaseModel):
    urls: list[str] = Field(
        ...,
        description="URL of the pages to visit (maximum 5)",
        min_items=1,
        max_items=5,
    )
    messages: Annotated[list[BaseMessage], InjectedState("messages")] = []
    verbose: bool = Field(
        default_factory=bool,
    )


class HTTPRefArgs(BaseModel):
    refs: list[int] = Field(
        ...,
        description="Reference numbers to visit (maximum 5)",
    )
    messages: Annotated[list[BaseMessage], InjectedState("messages")] = []
    references: Annotated[list[dict], InjectedState("references")] = {}
    verbose: bool = Field(
        default_factory=bool,
    )


class URLPattern:
    DOMAIN = re.compile(
        r"(http(s)?:\/\/)?(www\.)?([-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6})\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)"
    )

    @classmethod
    def extract_domain(cls, url: str) -> Optional[str]:
        match = cls.DOMAIN.match(url)
        return match.group(4) if match else None


class URLProcessor:
    def __init__(self, patterns: dict[str, str]):
        self.patterns = patterns

    def process_url(self, url: str) -> str:
        for pattern, replacement in self.patterns.items():
            match = re.search(pattern, url)
            if match:
                return replacement.format(*match.groups())
        return url


class ContentFetcher(HTTPXMixin):
    def __init__(self):
        self.url_processor = URLProcessor(
            {
                r"^https?:\/\/(?:m\.)?blog\.naver\.com\/([^\/?#]+)\/([0-9]+)\/?": "https://m.blog.naver.com/PostView.naver?blogId={0}&logNo={1}"
            }
        )

    async def fetch_content(self, url: str, headers: dict) -> Document:
        try:
            url = self.url_processor.process_url(url)
            if url.endswith(".pdf"):
                return await self._fetch_pdf(url)
            elif youtube_url_pattern.match(url):
                # youtube 영상 url 종류
                # 도메인: youtube.com, youtu.be, youtube-nocookie.com
                # 경로:
                #   - /watch : URL 내에 쿼리 파라미터로 v=가 있거나, /watch/ 뒤에 직접 영상ID가 오는 경우
                #   - /v/, /embed/, /e/, /shorts/, /live/ : 바로 뒤에 직접 영상ID가 오는 경우
                # 영상 ID: YouTube 영상 ID는 [A-Za-z0-9_-]{11} 형태로 11자리.
                return await self._fetch_youtube(url)
            return await self._fetch_html(url, headers)

        except ContentTypeError as e:
            return Document(
                page_content="올바른 타입의 컨텐츠가 아닙니다.",
                metadata={
                    "error": type(e).__name__,
                    "error_content": str(e),
                    "source": url,
                },
            )
        except Exception as e:
            return Document(
                page_content="콘텐츠에 접근할 수 없습니다.",
                metadata={
                    "error": type(e).__name__,
                    "error_content": str(e),
                    "source": url,
                },
            )

    async def _fetch_pdf(self, url: str) -> Document:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            pdf = base64.b64encode(response.content).decode("utf-8")
            return Document(page_content=pdf, metadata={"source": url, "type": "pdf"})

    async def _fetch_html(self, url: str, headers: dict) -> Document:
        html_text = await self.aget(url, headers, content_type="text")
        return Document(
            page_content=html_text, metadata={"source": url, "type": "text"}
        )

    async def _fetch_youtube(self, url: str) -> Document:
        # get video id from url
        pattern = re.compile(
            r'(?:youtube(?:-nocookie)?\.com/(?:[^/]+/.+/|(?:v|e(?:mbed)?/?|.*[?&]v=)|(?:embed/|shorts/|live/))|youtu\.be/)([^"&?/ ]{11})'
        )
        match = pattern.search(url)
        video_id = match.group(1)

        # get script from video id
        transcript_info = YouTubeTranscriptApi.get_transcript(
            video_id, languages=LANGUAGE_CODES
        )
        transcript_info = sorted(transcript_info, key=lambda x: x["start"])
        full_scripts = [script["text"].strip() for script in transcript_info]
        return Document(
            page_content="\n".join(full_scripts),
            metadata={"source": url, "type": "youtube_summary"},
        )


class BaseHTTPTool(AsyncTool, HTMLToMarkdownMixin):
    summarize_graph: MapReduceSummarizationSubgraph
    content_fetcher: ContentFetcher = ContentFetcher()

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        total_limit_tokens: int = 4096,
    ) -> "URLSummarizeTool":
        summarize_graph = MapReduceSummarizationSubgraph.from_llm(
            llm=llm, chunk_size=total_limit_tokens
        )

        return cls(
            summarize_graph=summarize_graph,
        )

    async def _fetch_and_preprocess(self, url: str) -> Document:
        doc = await self.content_fetcher.fetch_content(url, {"Accept": "text/*"})
        if "error" in doc.metadata:
            return doc
        md_doc = await self.aclean_html(doc)
        md_doc.page_content = re.sub(
            r"\n+", " ", md_doc.page_content
        )  # [:2048] # TODO: 2048제거.
        return md_doc


class URLSummarizeTool(BaseHTTPTool):
    name: str = "summarize_url"
    description: str = "Summarize the overall content of the website."
    args_schema: Type[BaseModel] = HTTPURLArgs

    def verify_args(self, user_message: str, args: dict) -> bool:
        urls = args["urls"]
        for url in urls:
            domain = URLPattern.extract_domain(url)
            if domain is None:
                return False
            if domain.lower() not in user_message.lower():
                return False
        return True

    async def _arun(
        self,
        urls: list[str],
        messages: list[dict] = [],
        verbose: bool = False,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> list:
        dispatcher = adispatch_custom_event if verbose else noop

        urls = [url if url.startswith("http") else "https://" + url for url in urls]
        keywords = ", ".join([f"{url}" for url in urls[:4]]) + (
            f" 외 {len(urls) - 4}개를" if len(urls) > 4 else "을"
        )

        await dispatcher("event", {"speak": f"{keywords} 읽고 있어요."})
        results = []
        md_docs = []
        for url in urls:
            md_doc = await self._fetch_and_preprocess(url)
            md_docs.append(md_doc)

        filtered = [doc for doc in md_docs if "error" not in doc.metadata]

        if not filtered:
            return [
                {
                    "page_content": "접근할 수 있는 출처가 없습니다.",
                    "metadata": {"error": "Error"},
                }
            ]

        if len(filtered) < len(urls):
            await dispatcher(
                "event",
                {
                    "speak": "일부 컨텐츠에 접근하지 못했어요. 접근 가능한 웹 사이트의 요약을 시작합니다."
                },
            )
        else:
            await dispatcher("event", {"speak": "웹 사이트 요약을 시작합니다."})

        results = await self.summarize_graph.abatch(
            md_docs, user_query=messages[-2].content
        )

        for url, result in zip(urls, results):
            result["metadata"]["thumbnail"] = None
            try:
                result["metadata"]["title"] = await self.aget_title(
                    url=url, headers={"Accept": "text/*"}, content_type="text"
                )
            except Exception as e:
                result["metadata"]["title"] = None

        if len(md_docs) > 1:
            if len(filtered) < len(urls):
                await dispatcher(
                    "event", {"speak": "접근 가능한 사이트의 요약을 모두 완료했습니다."}
                )
            else:
                await dispatcher(
                    "event", {"speak": "요청하신 사이트의 요약을 완료했습니다."}
                )
        else:
            await dispatcher(
                "event", {"speak": "요청하신 사이트의 요약을 모두 완료했습니다."}
            )
        return results


class RefSummarizeTool(BaseHTTPTool):
    name: str = "summarize_references"
    description: str = "Summarize the full content of referenced sources."
    args_schema: Type[BaseModel] = HTTPRefArgs

    async def _arun(
        self,
        refs: list[int],
        messages: list[dict] = [],
        references: list[dict] = [],
        verbose: bool = False,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> list:
        dispatcher = adispatch_custom_event if verbose else noop

        referenced_idxs = [
            ref["number"]
            for ref in references
            if ref.get("number") in refs and ref.get("source") is not None
        ][:5]

        await dispatcher(
            "event", {"speak": "검색 결과 중 요청하신 출처를 더 자세히 읽고 있어요."}
        )

        if not referenced_idxs:
            return [
                {
                    "page_content": "접근할 수 있는 출처가 없습니다.",
                    "metadata": {"error": "Error"},
                }
            ]

        referenced_urls = [
            ref["source"] for ref in references if ref.get("number") in referenced_idxs
        ]

        unique_urls = {}
        for idx, url in zip(referenced_idxs, referenced_urls):
            if url not in unique_urls:
                unique_urls[url] = idx

        referenced_urls, referenced_idxs = map(list, zip(*unique_urls.items()))

        keywords = ", ".join(referenced_urls[:4]) + (
            f" 외 {len(referenced_urls) - 4}개" if len(referenced_urls) > 4 else ""
        )
        await dispatcher("event", {"speak": f"{keywords}의 요약을 시작합니다."})

        results = []
        md_docs = []
        for idx, url in zip(referenced_idxs, referenced_urls):
            md_doc = await self._fetch_and_preprocess(url)
            md_docs.append(md_doc)

        results = await self.summarize_graph.abatch(
            md_docs, user_query=messages[-2].content
        )

        # md_docs내의 문서는 referenced_idxs와 일치하고, abatch 내의 gather는 return 순서를 보장하므로.
        for idx, url, result in zip(referenced_idxs, referenced_urls, results):
            # dict이므로, call by reference로 작동하므로.
            result["metadata"]["source_no"] = idx
            result["metadata"]["thumbnail"] = None
            try:
                result["metadata"]["title"] = await self.aget_title(
                    url=url, headers={"Accept": "text/*"}, content_type="text"
                )
            except Exception as e:
                result["metadata"]["title"] = None

        if len(refs) > 1:
            await dispatcher(
                "event", {"speak": "출처에 대한 요약을 모두 완료했습니다."}
            )

        return results
