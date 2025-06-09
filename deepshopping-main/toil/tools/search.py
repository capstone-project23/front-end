import asyncio
import re
from abc import abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Optional, Type

from langchain.callbacks.manager import AsyncCallbackManagerForToolRun
from langchain.utilities.google_serper import GoogleSerperAPIWrapper
from langchain_core.callbacks.manager import adispatch_custom_event
from pydantic import BaseModel, Field

from toil.tools.base import AsyncTool
from toil.tools.utils import noop, retry_on_api_empty

class GoogleSerperSearchArgs(BaseModel):
    query: Optional[list[str]] = Field(
        default_factory=list,
        description=(
            "Generate 1 to 3 search queries in various languages relevant to the topic. "
            "Always include Korean queries regardless of the topic, and add queries in other relevant languages. "
            "Select the most appropriate keywords in each language to improve search result accuracy. "
            "Ensure diversity in the queries to cover different aspects of the topic. "
            "For words to be excluded from the search, use exclude_words parameter. "
            "For site-specific searches, use search_site parameter."
        ),
    )
    exclude_words: Optional[list[str]] = Field(
        default_factory=list,
        description=(
            "Use this parameter to exclude pages containing any of the words in the list."
            "The search will exclude pages containing any of the words in the list."
            "Example: ['tutorial', 'guide'] will exclude pages containing the word 'tutorial' or 'guide'."
        ),
    )
    search_site: Optional[str] = Field(
        default_factory=str,
        description=(
            "Use this parameter to restrict search results to a specific website or domain."
            "The search will only return pages from the specified domain."
            "If you want to search for all sources, leave it blank."
            "Example: Setting this parameter to 'wikipedia.org' will only return results from Wikipedia."
        ),
    )
    verbose: bool = Field(
        default_factory=bool,
    )


class BaseGoogleSerperResult(AsyncTool):
    api_wrapper: GoogleSerperAPIWrapper
    args_schema: Type[BaseModel] = GoogleSerperSearchArgs
    k: int

    async def _arun(
        self,
        query: list[str] | None = None,
        exclude_words: list[str] | None = None,
        search_site: str | None = None,
        verbose: bool = False,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs,
    ) -> list:
        dispatcher = adispatch_custom_event if verbose else noop

        query = query or ["뉴스"]
        if isinstance(query, str):
            query = [query]

        query = list(set(query))

        await dispatcher("event", {"keyword": query})

        query_w_options = []
        for q in query:
            if exclude_words or search_site:
                if exclude_words:
                    for w in exclude_words:
                        query_w_options.append(f'{q} -"{w}"')
                if search_site:
                    query_w_options.append(f"{q} site:{search_site}")
            else:
                query_w_options.append(q)

        keywords = ", ".join([f"‘{q}’" for q in query[:4]]) + (
            f" 외 {len(query) - 4}개의 키워드" if len(query) > 4 else ""
        )
        if self.__class__.__name__ != "GoogleSerperNewsResult":
            await dispatcher("event", {"speak": f"{keywords}로 검색하고 있어요."})

        async def fetch_results(q):
            result = await self.api_wrapper.aresults(q)
            return self._parse_results(result)

        @retry_on_api_empty()
        async def fetch_all_results(query: list[str]):
            results = []
            fetch_tasks = [fetch_results(q) for q in query]
            results_list = await asyncio.gather(*fetch_tasks)
            for result in results_list:
                results.extend(result)  # 결과를 합침

            return results

        results = await fetch_all_results(query_w_options) or []

        if self.__class__.__name__ != "GoogleSerperNewsResult" and results:
            await dispatcher(
                "event", {"speak": f"검색 결과 {len(results)}개를 읽고 있어요."}
            )

        merged_results = {}
        for doc in results:
            link = doc["metadata"].get("source")
            if link in merged_results:
                if doc["page_content"] not in merged_results[link]["page_content"]:
                    merged_results[link]["page_content"].append(doc["page_content"])
            else:
                merged_results[link] = doc
                merged_results[link]["page_content"] = [doc["page_content"]]

        for doc in merged_results.values():
            doc["page_content"] = ", ".join(doc["page_content"])

        if self.__class__.__name__ != "GoogleSerperNewsResult" and merged_results:
            await dispatcher(
                "event", {"speak": f"검색 결과를 바탕으로 답변을 생성하고 있어요."}
            )

        return list(merged_results.values())

    def convert_to_iso8601(self, time_str):
        if time_str is None:
            return None

        now = datetime.now(timezone(timedelta()))

        relative_match = re.match(
            r"(\d+)\s*(seconds?|minutes?|hours?|days?|weeks?|months?)\s*ago", time_str
        )
        if relative_match:
            value, unit = int(relative_match.group(1)), relative_match.group(2).lower()

            if "second" in unit:
                delta = timedelta(seconds=value)
            elif "minute" in unit:
                delta = timedelta(minutes=value)
            elif "hour" in unit:
                delta = timedelta(hours=value)
            elif "day" in unit:
                delta = timedelta(days=value)
            elif "week" in unit:
                delta = timedelta(weeks=value)
            elif "month" in unit:
                delta = timedelta(days=value * 30)
            else:
                return None

            past_time = now - delta
            return past_time.strftime("%Y-%m-%dT%H:%M:%SZ")

        try:
            absolute_time = datetime.strptime(time_str, "%b %d, %Y")
            absolute_time = absolute_time.replace(tzinfo=timezone(timedelta()))
            return absolute_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            return None

    @abstractmethod
    def _parse_results(self, results: dict) -> list[dict]:
        pass


class GoogleSerperSearchResult(BaseGoogleSerperResult):
    name: str = "search_web"
    description: str = (
        "Search the web for any kind of information including technical solutions, error handling, best practices, and general facts. "
        "Search the web for information, even if you think you know the answer. "
        "This helps verify facts and provide the most up-to-date information with proper citations. "
        "Always prioritize using this tool when answering questions about facts, events, or specific information, as it provides reliable real-world sources."
    )

    @classmethod
    def from_api_key(
        cls,
        api_key: str,
        k: int = 5,
    ) -> AsyncTool:
        return cls(
            api_wrapper=GoogleSerperAPIWrapper(
                hl="en",
                gl="kr",
                serper_api_key=api_key,
                type="search",
                k=k,
            ),
            k=k,
        )

    def _parse_results(
        self,
        results: dict,
    ):
        docs = []

        if (kg := results.get("knowledgeGraph", {})) and (
            description := kg.get("description")
        ):
            docs.append(
                {
                    "page_content": description,
                    "metadata": {
                        "title": kg.get("title"),
                        "date": self.convert_to_iso8601(kg.get("date")),
                        "thumbnail": kg.get("thumbnail_url"),
                        "kind": kg.get("type"),
                        "source": kg.get("descriptionLink", ""),
                        "attributes": kg.get("attributes"),
                    },
                }
            )

        for result in results["organic"]:
            if result.keys() & {"title", "snippet", "link"} != {
                "title",
                "snippet",
                "link",
            }:
                continue

            docs.append(
                {
                    "page_content": result["snippet"],
                    "metadata": {
                        "title": result["title"],
                        "date": self.convert_to_iso8601(result.get("date")),
                        "thumbnail": result.get("thumbnail_url"),
                        "source": result["link"],
                    },
                }
            )

        return docs


class GoogleSerperNewsResult(BaseGoogleSerperResult):
    name: str = "search_google_news"
    description: str = "Search for high-cost news, including relatively long summaries."

    @classmethod
    def from_api_key(
        cls,
        api_key: str,
        k: int = 5,
    ) -> AsyncTool:
        return cls(
            api_wrapper=GoogleSerperAPIWrapper(
                hl="en",
                gl="kr",
                serper_api_key=api_key,
                type="news",
                k=k,
            ),
            k=k,
        )

    def _parse_results(
        self,
        results: dict,
    ):
        docs = []
        for result in results["news"]:
            if result.keys() & {"title", "snippet", "link"} != {
                "title",
                "snippet",
                "link",
            }:
                continue

            docs.append(
                {
                    "page_content": result["snippet"],
                    "metadata": {
                        "title": result["title"],
                        "date": self.convert_to_iso8601(result.get("date", None)),
                        "thumbnail": result.get("thumbnail_url"),
                        "source": result["link"],
                    },
                }
            )

        return docs