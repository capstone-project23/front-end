import asyncio
import json
import re
import base64
import httpx
from typing import Annotated, Any, Optional
from collections import defaultdict
from dotenv import load_dotenv
from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableBinding, RunnableConfig
from langgraph.graph.message import add_messages
from langgraph.utils.runnable import RunnableCallable
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_core.prompts import BasePromptTemplate

from toil.core.prompt import (
    PreliminaryInvestigationPrompt,
    ReferenceFilterPrompt,
    ReportCompilationPrompt,
    ReportSectionsGenerationPrompt,
    ResearchPlanningPrompt,
    ResearchUnitSynthesisPrompt,
    ImageEvaluatePrompt,
)
from toil.tools.base import AsyncTool
from toil.core.prompt import BasePrompt
from toil.model_config import get_maximum_tokens_from_llm

load_dotenv()

MAX_TASK_LIMIT = 10

class AsyncRunnableCallable(RunnableCallable):
    def _func(
        self,
        *args,
        **kwargs,
    ):
        raise NotImplementedError
    
# ===== State Models using Pydantic BaseModel =====
class ResearchUnitState(BaseModel):
    topic: str
    task_description: str
    research_results: str = ""
    messages: list[BaseMessage] = Field(default_factory=list)
    references: list[dict] = Field(default_factory=list)


def add_tasks(left: list[ResearchUnitState], right: list[ResearchUnitState]):
    # operator.add를 사용하지 않는 이유는 subgraph의 결과물이 바깥 main graph로 나오는 시점에서 각 값들이 duplicate되기 때문임.
    unique_dict = {}
    for task in left + right:
        unique_dict[task.task_description] = task

    return list(unique_dict.values())


# Combined graph state with message processing support
class DeepSearchState(BaseModel):
    topic: str = ""
    base_knowledge: list[dict] = Field(default_factory=list)
    current_task: ResearchUnitState | None = None
    incomplete_tasks: list[ResearchUnitState] = Field(default_factory=list)
    completed_tasks: Annotated[list[ResearchUnitState], add_tasks] = Field(
        default_factory=list
    )
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    references: list[dict] = Field(default_factory=list)
    sections: list[str] | None = None
    final_report: str = ""
    user_info: dict = Field(default_factory=dict)
    product_recommendations: list[dict] = Field(default_factory=list)

# Output Schemas
class Plan(BaseModel):
    """Plan to follow in future"""

    steps: list[str] = Field(
        description="different steps to follow, should be in sorted order"
    )


class UselessReference(BaseModel):
    reason: str = Field(description="A reason for excluding the reference")
    index: int = Field(description="Index of references to exclude")


class ReferenceFilterResult(BaseModel):
    reasons: list[str] = Field(description="Reasons for excluding each reference")
    excluded_indices: list[int] = Field(description="Indices of references to exclude")


class ReportSections(BaseModel):
    sections: list[str] = Field(
        description=(
            "A hierarchical list of report section titles. "
            "Each item should clearly indicate its hierarchy with numbering, "
            "e.g., '1. Section Title', '2. Section Title', etc. "
            "The structure must logically reflect the provided Topic, Tasks, and References, "
            "and serve as a comprehensive outline for detailed report writing."
        )
    )

class ImageScore(BaseModel):
    gender_appropriate: bool = Field(
        description="성별에 맞는 옷인가?"
    )
    garment_completeness: bool = Field(
        description="의류 완전성 (옷의 전체 형태가 보이는가?)"
    )
    single_garment: bool = Field(
        description="단일 의류 여부 (옷이 한 개만 있는가?)"
    )

# ===== Node Classes =====
class DeepSearchQueryAnalysis(AsyncRunnableCallable):
    def __init__(
        self,
        llm: BaseLanguageModel | RunnableBinding,
        summarize_llm: BaseLanguageModel | RunnableBinding,
        prompt: BasePrompt,
        tools: list[AsyncTool],
        *,
        name: str = "deep_search_query_analysis",
        tags: Optional[list[str]] = None,
        answer_llm_tags: list[str] = ["answer"],
    ) -> None:
        super().__init__(self._func, self._afunc, name=name, tags=tags, trace=False)
        self.prompt = prompt
        self.llm = llm.with_config(tags=answer_llm_tags)
        self.tools = tools
        self.llm_with_tools = llm.bind_tools(tools).with_config(tags=answer_llm_tags)
        self.summarize_llm = summarize_llm

    async def _afunc(self, state: DeepSearchState) -> dict[str, Any]:
        current_task = state.current_task

        inputs = await self._aprep_inputs(
            messages=current_task.messages,
            topic=state.topic,
            task_description=current_task.task_description,
            references=[
                ref
                for previous_task in state.completed_tasks
                for ref in previous_task.references
            ],
        )

        response = await self.llm_with_tools.ainvoke(inputs)
        response = await self._merge_tool_calls(response)

        return {
            "current_task": ResearchUnitState(
                topic=current_task.topic,
                task_description=current_task.task_description,
                messages=current_task.messages + [response],
                references=current_task.references,
            )
        }

    async def _aprep_inputs(
        self,
        messages: list[BaseMessage],
        topic: str,
        task_description: str,
        references: list[dict],
    ):
        # 토큰 제한에 맞게 메시지를 줄이기
        trimmed_messages = await self._atrim_messages(
            messages=messages,
            token_limit=get_maximum_tokens_from_llm(self.llm_with_tools),
            token_counter=self.llm_with_tools.get_num_tokens_from_messages,
        )

        # 프롬프트 형식에 맞게 메시지 준비
        formatted_prompt = self.prompt.format_messages(
            messages=trimmed_messages,
            topic=topic,
            task_description=task_description,
            references=references,
            user_info={},  # 빈 user_info 추가
        )

        return formatted_prompt

    async def _atrim_messages(
        self,
        messages: list[BaseMessage],
        token_limit: int,
        token_counter: callable,
    ):
        token_used = 0
        trim_index = 0
        for i in reversed(range(len(messages))):
            message_token_size = token_counter([messages[i]])

            if token_used + message_token_size > token_limit:
                trim_index = i + 1
                break

            token_used += message_token_size

        return messages[trim_index:] or []

    async def _merge_tool_calls(self, response):
        # Tool 호출을 병합하는 로직
        response = response.copy()
        try:
            if response.additional_kwargs.get("tool_calls"):
                # 1. additional_kwargs 정보 병합
                tool_id_index_map = {
                    call["id"]: call["index"]
                    for call in response.additional_kwargs.get("tool_calls")
                }

                calls_by_tool = defaultdict(list)
                for call in response.additional_kwargs.get("tool_calls"):
                    calls_by_tool[call["function"]["name"]].append(call)

                merged_calls = []
                for tool_name, calls in calls_by_tool.items():
                    base_call = calls[0]
                    base_args = json.loads(base_call["function"]["arguments"])

                    for call in calls[1:]:
                        for key, value in json.loads(
                            call["function"]["arguments"]
                        ).items():
                            if isinstance(value, list):
                                base_args.setdefault(key, []).extend(value)
                                base_args[key] = sorted(set(base_args[key]))
                            else:
                                # 비-리스트 값은 직접 업데이트
                                base_args[key] = value

                    base_call["function"]["arguments"] = json.dumps(
                        base_args, ensure_ascii=False
                    )
                    merged_calls.append(base_call)

                response.additional_kwargs["tool_calls"] = merged_calls

                # 2. tool call 정보 병합
                # tool_calls 정보와 additional_kwargs의 tool_calls 정보가 일치하지 않을 수 있음
                if hasattr(response, "tool_calls") and response.tool_calls:
                    response.tool_calls.sort(key=lambda call: tool_id_index_map[call["id"]])

                    calls_by_tool = defaultdict(list)
                    for call in response.tool_calls:
                        calls_by_tool[call["name"]].append(call)

                    merged_calls = []
                    for tool_name, calls in calls_by_tool.items():
                        base_call = calls[0]
                        base_args = base_call["args"]

                        for call in calls[1:]:
                            for key, value in call["args"].items():
                                if isinstance(value, list):
                                    base_args.setdefault(key, []).extend(value)
                                    base_args[key] = sorted(set(base_args[key]))
                                else:
                                    # 비-리스트 값은 직접 업데이트
                                    base_args[key] = value

                        base_call["args"] = base_args
                        merged_calls.append(base_call)
                    response.tool_calls = merged_calls
            
            return response
        except Exception as e:
            print(f"Error: {e}")
            return response


class DeepSearchToolCalling(AsyncRunnableCallable):
    def __init__(
        self,
        tools: list[AsyncTool],
        filter_llm: Optional[BaseLanguageModel] = None,
        *,
        name: str = "deep_search_tool_calling",
        tags: Optional[list[str]] = None,
    ) -> None:
        super().__init__(self._func, self._afunc, name=name, tags=tags, trace=False)
        self.tools = tools
        self.filter_llm = filter_llm
        self.blocked_url_pattern = re.compile(r"youtube\.com|youtu\.be|tiktok\.com")

    async def _afunc(
        self,
        input: DeepSearchState,
        config: RunnableConfig = None,
        *,
        store: Any = None,
    ) -> Any:
        config = config or {}
        
        # config에 필요한 키가 없으면 추가
        if 'configurable' not in config:
            config['configurable'] = {}
        
        if input.current_task:
            input = input.current_task

        tool_calls, input_type = self._parse_input(input, store)

        # TODO: DeepSearch 중 사용하는 모든 Tool speak 비활성화. 추후 정책에 따라 변경할 것.
        for call in tool_calls:
            call["args"]["verbose"] = False

        # 이벤트 디스패치 부분 수정 - config 전달 또는 조건부 실행
        if config and "callbacks" in config:
            try:
                await adispatch_custom_event(
                    "event",
                    {
                        "node": self.name,
                        "task": input.task_description,
                        "used_tools": [call["name"] for call in tool_calls],
                    },
                    config=config,  # config 전달
                )
            except Exception as e:
                print(f"이벤트 디스패치 오류 (무시됨): {e}")

        outputs = await asyncio.gather(
            *(self._arun_one(call, input_type, config) for call in tool_calls)
        )

        outputs = await self.postprocess_tool_results(input, outputs)

        return outputs

    async def postprocess_tool_results(
        self, state: ResearchUnitState, tool_call_results: list[ToolMessage]
    ):
        for tool_message in tool_call_results:
            name = tool_message.name
            observation = tool_message.content

            if not isinstance(observation, list):
                try:
                    observation = json.loads(observation)
                except Exception as e:
                    print(f"Error: {e}")
                    return {
                        "messages": ToolMessage(
                            content="An error occurred while post-processing the tool result.",
                            tool_call_id=tool_message.tool_call_id,
                        )
                    }

            if name in ["search_web"]:
                text_observation, _, _ = self._split_results_by_type(observation)
                formatted_tool_message, references = self._format_observation(
                    text_observation, state.references
                )
            elif name in [
                "search_google_news",
                "search_news",
                "search_weather",
                "summarize_url",
                "summarize_references",
            ]:
                formatted_tool_message, references = self._format_observation(
                    observation, state.references
                )
            else:
                continue

            tool_message.content = formatted_tool_message
            state.references = references

        return {
            "current_task": ResearchUnitState(
                topic=state.topic,
                task_description=state.task_description,
                messages=state.messages + tool_call_results,
                references=state.references,
            )
        }
        
    def _parse_input(self, input, store):
        # 입력에서 도구 호출 정보 추출
        if isinstance(input, ResearchUnitState):
            messages = input.messages
        else:
            messages = input
            
        tool_calls = []
        
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, "tool_calls"):
                for tool_call in last_message.tool_calls:
                    tool_calls.append({
                        "name": tool_call["name"],
                        "args": tool_call["args"],
                        "id": tool_call.get("id", ""),
                    })
                    
        return tool_calls, input
        
    async def _arun_one(self, call, input_type, config):
        # 단일 도구 호출 실행
        tool_name = call["name"]
        tool_args = call["args"]
        tool_id = call.get("id", "")
        
        # 도구 찾기
        tool = next((t for t in self.tools if t.name == tool_name), None)
        
        if not tool:
            return ToolMessage(
                content=f"Tool '{tool_name}' not found",
                tool_call_id=tool_id,
                name=tool_name,
            )
            
        try:
            # 도구 실행
            result = await tool.ainvoke(tool_args, config)
            return ToolMessage(
                content=result,
                tool_call_id=tool_id,
                name=tool_name,
            )
        except Exception as e:
            print(f"Error running tool {tool_name}: {e}")
            return ToolMessage(
                content=f"Error: {str(e)}",
                tool_call_id=tool_id,
                name=tool_name,
            )
        
    def _split_results_by_type(self, observation: list[dict]):
        text_results = []
        image_results = []
        video_results = []

        for item in observation:
            if item["metadata"].get("type") == "image":
                image_results.append(item)
            elif item["metadata"].get("type") == "video":
                video_results.append(item)
            else:
                text_results.append(item)

        return text_results, image_results, video_results
        
    def _format_observation(self, observation: list[dict], references: list[dict]):
        new_observation: list[dict] = []

        for it in observation:
            obj = {}
            reference = {}
            number = len(references)

            if "source" in it["metadata"]:
                if (
                    self.blocked_url_pattern.search(it["metadata"].get("source", ""))
                    and it["metadata"].get("type", "text") != "youtube_summary"
                ):
                    continue

                obj["number"] = number + 1
                reference["number"] = number + 1

                # reference key에 대한 처리
                reference_key = [
                    "source",
                    "title",
                    "thumbnail",
                    "date",
                    "attributes",
                    "kind",
                ]
                reference.update(
                    {
                        key: it["metadata"][key]
                        for key in reference_key
                        if key in it["metadata"]
                    }
                )
                reference["content"] = it["page_content"]

                references.append(reference)
            elif "source_no" in it["metadata"]:
                obj["number"] = it["metadata"]["source_no"]

            # 페이지 콘텐츠의 타입에 따라 JSON 파싱 또는 그대로 저장
            if it["metadata"].get("type", "text") == "json":
                obj["content"] = json.loads(it["page_content"].replace(r"\r", ""))
            else:
                obj["content"] = it["page_content"]

            other_keys = [
                key
                for key in it["metadata"]
                if key not in ["content", "source", "source_no", "thumbnail"]
            ]

            for key in other_keys:
                obj[key] = it["metadata"][key]

            new_observation.append(obj)

        formatted_tool_message = json.dumps(
            new_observation, indent=2, ensure_ascii=False
        )

        return formatted_tool_message, references


class PreliminaryInvestigation(AsyncRunnableCallable):
    def __init__(
        self,
        llm: BaseLanguageModel | RunnableBinding,
        search_tool: AsyncTool,
        *,
        name: str = "preliminary_investigation",
        tags: Optional[list[str]] = None,
    ) -> None:
        super().__init__(self._func, self._afunc, name=name, tags=tags, trace=False)
        self.llm = llm.bind_tools(tools=[search_tool])
        self.tools = [search_tool]
        self.prompt = PreliminaryInvestigationPrompt()

    async def _afunc(self, state: DeepSearchState) -> dict[str, Any]:
        # print("PreliminaryInvestigation: ", state)
        user_input = state.messages[-1].content
        user_info = getattr(state, "user_info", {})  # state에서 user_info 가져오기

        response = await self.llm.ainvoke(
            self.prompt.format_messages(topic=user_input, user_info=user_info)
        )
        
        # 올바른 config 객체 구성
        config = {
            "configurable": {},  # 필수 키
            "callbacks": None,   # 필요한 경우 콜백 추가
        }
        
        tool_results = await DeepSearchToolCalling(
            tools=self.tools, filter_llm=self.llm
        ).ainvoke(
            DeepSearchState(
                current_task=ResearchUnitState(
                    topic=user_input,
                    task_description="Preliminary Investigation",
                    messages=[response],
                ),
                user_info=user_info,  # user_info 추가
            ),
            config=config,
        )

        return {
            "topic": user_input,
            "base_knowledge": tool_results["current_task"].references,
            "user_info": user_info,  # 반환값에 user_info 포함
        }


class ResearchPlanning(AsyncRunnableCallable):
    def __init__(
        self,
        llm: BaseLanguageModel | RunnableBinding,
        *,
        name: str = "research_planning",
        tags: Optional[list[str]] = None,
    ) -> None:
        super().__init__(self._func, self._afunc, name=name, tags=tags, trace=False)
        self.llm = llm.with_structured_output(Plan).with_config(tags=["planner"])
        self.prompt = ResearchPlanningPrompt()

    async def _afunc(self, state: DeepSearchState) -> dict[str, Any]:
        if len(state.completed_tasks) >= MAX_TASK_LIMIT - 1:
            return {
                "current_task": None,
                "completed_tasks": [state.current_task],
            }

        if not state.incomplete_tasks:
            response: Plan = await self.llm.ainvoke(
                self.prompt.format_messages(
                    topic=state.topic,
                    base_knowledge=state.base_knowledge,
                    completed_tasks=state.completed_tasks,
                    incomplete_tasks=[],  # 빈 리스트 전달
                )
            )

            # TODO: LLM이 실제로 추가 계획이 필요없다고 판단되면, 생성을 안하는지 체크하기 위한 코드.
            if not response.steps:
                print("계획 생성 안함!!!")

            new_tasks = await self._create_tasks(
                topic=state.topic, task_list=response.steps
            )

            return {
                "current_task": new_tasks[0] if new_tasks else None,
                "incomplete_tasks": new_tasks[1:],
                "completed_tasks": [state.current_task] if state.current_task else [],
            }
        else:
            return {
                "current_task": state.incomplete_tasks[0],
                "incomplete_tasks": state.incomplete_tasks[1:],
                "completed_tasks": [state.current_task] if state.current_task else [],
            }

    async def _create_tasks(
        self, topic: str, task_list: list[str]
    ) -> list[ResearchUnitState]:
        def _create_init_message(topic: str, task_description: str) -> BaseMessage:
            return HumanMessage(
                content=f"Let's research '{task_description}' about {topic}."
            )  # TODO: 더 적절하게 content 바꾸기.

        tasks = []
        for task_item in task_list:
            tasks.append(
                ResearchUnitState(
                    topic=topic,
                    task_description=task_item,
                    messages=[
                        _create_init_message(topic=topic, task_description=task_item)
                    ],
                )
            )

        return tasks


class ResearchUnitSynthesis(AsyncRunnableCallable):
    def __init__(
        self,
        llm: BaseLanguageModel | RunnableBinding,
        *,
        name: str = "research_unit_synthesis",
        tags: Optional[list[str]] = None,
    ) -> None:
        super().__init__(self._func, self._afunc, name=name, tags=tags, trace=False)
        self.llm = llm
        self.prompt = ResearchUnitSynthesisPrompt()  # TODO: 이름 바꾸기.

    async def _afunc(self, state: DeepSearchState) -> dict[str, Any]:
        if not state.current_task:
            return {}

        summary_prompt_messages = self.prompt.format_messages(
            topic=state.topic,
            task_description=state.current_task.task_description,
            messages=state.current_task.messages,
            references=state.current_task.references,
            user_info=state.user_info,  # user_info 추가
        )  # TODO: 나중에 aprep_inputs로 ㄱㄱ.

        summary_response = await self.llm.ainvoke(summary_prompt_messages)

        await adispatch_custom_event(
            "event",
            {
                "node": self.name,
                "task": state.current_task.task_description,
                "subconclusion": summary_response.content,
            },
        )

        return {
            "current_task": ResearchUnitState(
                topic=state.current_task.topic,
                task_description=state.current_task.task_description,
                research_results=summary_response.content,
                messages=state.current_task.messages,
                references=state.current_task.references,
            )
        }


class ReferenceFilter(AsyncRunnableCallable):
    def __init__(
        self,
        llm: BaseLanguageModel | RunnableBinding,
        *,
        name: str = "reference_filter",
        tags: Optional[list[str]] = None,
    ) -> None:
        super().__init__(self._func, self._afunc, name=name, tags=tags, trace=False)
        self.llm = llm.with_structured_output(ReferenceFilterResult)
        self.prompt = ReferenceFilterPrompt()

    async def _afunc(self, state: DeepSearchState) -> dict[str, Any]:
        if not state.current_task or not state.current_task.references:
            return {}

        # 레퍼런스 필터링을 위한 프롬프트 생성
        filter_prompt_messages = self.prompt.format_messages(
            topic=state.topic,
            task_description=state.current_task.task_description,
            references=state.current_task.references,
            user_info=state.user_info,  # user_info 추가
        )

        # LLM을 사용하여 필터링할 레퍼런스 식별
        filter_result: ReferenceFilterResult = await self.llm.ainvoke(
            filter_prompt_messages
        )

        # 필터링 결과 로깅
        if filter_result is not None and filter_result.excluded_indices:
            print(f"Filtered out {len(filter_result.excluded_indices)} references:")
            for idx, reason in zip(
                filter_result.excluded_indices, filter_result.reasons
            ):
                print(f"  - Number {idx}: {reason}")

            # number 필드를 기준으로 필터링
            filtered_references = [
                ref
                for ref in state.current_task.references
                if ref.get("number") not in filter_result.excluded_indices
            ]
        else:
            filtered_references = state.current_task.references

        # number 필드를 기준으로 정렬
        filtered_references.sort(key=lambda x: x.get("number", float("inf")))

        # 1부터 n까지 number 다시 부여
        for i, ref in enumerate(filtered_references, 1):
            ref["number"] = i

        print(f"References after filtering and renumbering: {len(filtered_references)}")

        return {
            "current_task": ResearchUnitState(
                topic=state.current_task.topic,
                task_description=state.current_task.task_description,
                research_results="조사 완료.",
                messages=state.current_task.messages,
                references=filtered_references,
            )
        }


class ReportSectionsGeneration(AsyncRunnableCallable):
    def __init__(
        self,
        llm: BaseLanguageModel | RunnableBinding,
        *,
        name: str = "report_sections_generation",
        tags: Optional[list[str]] = None,
    ) -> None:
        super().__init__(self._func, self._afunc, name=name, tags=tags, trace=False)
        self.llm = llm.with_structured_output(ReportSections)
        self.prompt = ReportSectionsGenerationPrompt()

    async def _afunc(self, state: DeepSearchState) -> dict[str, Any]:

        inputs = self.prompt.format_messages(
            topic=state.topic,
            tasks=[task.task_description for task in state.completed_tasks],
            references=state.references,
            user_info=state.user_info,  # user_info 추가
        )
        sections = await self.llm.ainvoke(inputs)
        return {
            "sections": sections.sections,
        }


class ReportCompilation(AsyncRunnableCallable):
    def __init__(
        self,
        llm: BaseLanguageModel | RunnableBinding,
        *,
        name: str = "report_compilation",
        tags: Optional[list[str]] = None,
    ) -> None:
        super().__init__(self._func, self._afunc, name=name, tags=tags, trace=False)
        self.llm = llm.with_config(tags=["answer"])
        self.prompt = ReportCompilationPrompt()  # TODO: 이름바꾸기.

    async def _afunc(self, state: DeepSearchState) -> dict[str, Any]:
        inputs = await self._aprep_inputs(
            topic=state.topic, sections=state.sections, tasks=state.completed_tasks, user_info=state.user_info
        )
        response = await self.llm.ainvoke(inputs)

        return {
            "final_report": response.content,
            "messages": [response],
            "references": [
                ref for task in state.completed_tasks for ref in task.references
            ],
        }

    async def _aprep_inputs(
        self, topic: str, sections: list[str], tasks: list[ResearchUnitState], user_info: dict
    ):
        # 최종 보고서에서의 참조를 위해 index 정렬.
        current_index = 1

        for task in tasks:
            updated_references = []
            for item in task.references:
                updated_item = item.copy()
                updated_item["number"] = current_index
                current_index += 1
                updated_references.append(updated_item)
            task.references = updated_references

        # user_info를 tasks[0]에서 가져오거나 빈 딕셔너리 사용
        user_info = getattr(tasks[0], "user_info", {}) if tasks else {}
        
        return self.prompt.format_messages(
            topic=topic, 
            sections=sections, 
            tasks=tasks,
            user_info=user_info,  # user_info 추가
        )
        
class ProductRecommendationExtractor(AsyncRunnableCallable):
    def __init__(
        self,
        llm: BaseLanguageModel | RunnableBinding,
        search_product_tool: AsyncTool,
        *,
        name: str = "product_recommendation_extractor",
        tags: Optional[list[str]] = None,
    ) -> None:
        super().__init__(self._func, self._afunc, name=name, tags=tags, trace=False)
        self.llm = llm
        self.search_product_tool = search_product_tool

    async def _afunc(self, state: DeepSearchState) -> dict[str, Any]:
        # 보고서에서 상품 추천 목록 추출
        product_names = await self._extract_product_names(state.final_report, state.user_info)
        
        # 추출된 각 상품에 대한 정보 검색
        product_results = await self._search_products(product_names,state.user_info)
        
        return {
            "product_recommendations": product_results
        }
        
    async def _extract_product_names(self, report: str, user_info: dict) -> list[str]:
        # LLM을 사용하여 보고서에서 상품명 추출
        prompt = f"""
다음 보고서에서 사용자에게 추천된 상품들의 이름을 정확히 추출해주세요.
보고서 내용과 유저 정보를 보고 사용자에게 가장 잘 맞는 상의 상품을 3개 이상 추천해주세요.
이름이 정확하지 않을 경우 맞는 상품 쿼리라도 만드세요.

목록 형태로 상품 이름만 반환해주세요:

보고서 내용:
{report}

유저 정보:
{user_info}
        """
        
        response = await self.llm.ainvoke(prompt)
        product_names = [name.strip() for name in response.content.strip().split('\n') if name.strip()]
        if not product_names:
            return []
        return product_names
    
    async def _search_products(self, product_names: list[str], user_info: dict = None) -> list[dict]:
        results = []
        for product_name in product_names:
            try:
                search_query = product_name
                
                # 성별 정보가 있으면 검색 쿼리에 추가
                if user_info and "gender" in user_info:
                    gender_info = user_info.get("gender", "")
                    search_query = f"{gender_info} {product_name}"
                
                search_result = await self.search_product_tool.ainvoke(search_query)
                product_data = json.loads(search_result)
                
                if product_data.get("filtered_products"):
                    # 상위 3개 제품만 선택
                    top_products = product_data["filtered_products"][:1]
                    results.append({
                        "query": product_name,
                        "products": top_products
                    })
            except Exception as e:
                print(f"상품 '{product_name}' 검색 중 오류 발생: {e}")
                
        return results
    
class ImageEvaluator:
    llm: BaseLanguageModel
    image_evaluate_prompt: BasePromptTemplate
    
    def __init__(self, llm: BaseLanguageModel | RunnableBinding):
        self.llm = llm
        self.image_evaluate_prompt = ImageEvaluatePrompt()
    
    async def evaluate_image_from_url(self, image_url: str, user_info: dict = None):
        async with httpx.AsyncClient() as client:
            response = await client.get(image_url)
            response.raise_for_status()
            image_data = base64.b64encode(response.content).decode("utf-8")
        
        structured_llm = self.llm.with_structured_output(ImageScore)
        messages = self.image_evaluate_prompt.format_messages(
            image_data=image_data, 
            user_info=user_info
        )
        
        llm_response = await structured_llm.ainvoke(messages)
        return llm_response

    async def evaluate_image_from_data(self, image_data: str, user_info: dict = None): 
        structured_llm = self.llm.with_structured_output(ImageScore)
        messages = self.image_evaluate_prompt.format_messages(
            image_data=image_data, 
            user_info=user_info
        )
        response = await structured_llm.ainvoke(messages)
        return response

class ProductImageFilter(AsyncRunnableCallable):
    def __init__(
        self,
        llm: BaseLanguageModel | RunnableBinding,
        *,
        name: str = "product_image_filter",
        tags: Optional[list[str]] = None,
    ) -> None:
        super().__init__(self._func, self._afunc, name=name, tags=tags, trace=False)
        self.image_evaluator = ImageEvaluator(llm)

    async def _afunc(self, state: DeepSearchState) -> dict[str, Any]:
        if not state.product_recommendations:
            return {}
        
        filtered_recommendations = []
        user_info = state.user_info  # 사용자 정보 가져오기
        
        for recommendation in state.product_recommendations:
            query = recommendation.get("query", "")
            products = recommendation.get("products", [])
            
            filtered_products = []
            
            for product in products:
                image_url = product.get("image", "")
                
                if not image_url:
                    print(f"상품 '{product.get('title', 'Unknown')}': 이미지 URL이 없어서 제외")
                    continue
                
                try:
                    # 이미지 품질 평가 (user_info 전달)
                    image_score = await self.image_evaluator.evaluate_image_from_url(
                        image_url, user_info
                    )
                    
                    # 3가지 조건을 모두 만족하는 경우만 유지
                    if (image_score.gender_appropriate and 
                        image_score.garment_completeness and 
                        image_score.single_garment):
                        
                        filtered_products.append(product)
                        print(f"상품 '{product.get('title', 'Unknown')}': 이미지 조건 통과")
                    else:
                        print(f"상품 '{product.get('title', 'Unknown')}': 이미지 조건 불통과로 제외")
                        print(f"이미지 조건: 성별적합={image_score.gender_appropriate}, "
                              f"완전성={image_score.garment_completeness}, "
                              f"단일성={image_score.single_garment}")
                        print(f"상품 url: {image_url}")
                        
                except Exception as e:
                    print(f"상품 '{product.get('title', 'Unknown')}' 이미지 평가 중 오류로 제외: {e}")
                    continue
            
            # 필터링된 상품이 있는 경우만 추가
            if filtered_products:
                filtered_recommendations.append({
                    "query": query,
                    "products": filtered_products
                })
            else:
                print(f"쿼리 '{query}': 모든 상품이 이미지 조건 불통과로 제외됨")
        
        print(f"이미지 필터링 완료: {len(filtered_recommendations)}개 카테고리, "
              f"총 {sum(len(rec['products']) for rec in filtered_recommendations)}개 상품 남음")
        
        return {
            "product_recommendations": filtered_recommendations
        }