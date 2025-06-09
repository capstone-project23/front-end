from typing import Any
 
from langchain.schema.messages import HumanMessage
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages.base import messages_to_dict
from langchain_core.messages.utils import messages_from_dict
from langchain_core.runnables import RunnableBinding
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel
from langgraph.checkpoint.memory import MemorySaver
 
from toil.core.node import (
    DeepSearchQueryAnalysis,
    DeepSearchState,
    DeepSearchToolCalling,
    PreliminaryInvestigation,
    ReferenceFilter,
    ReportCompilation,
    ReportSectionsGeneration,
    ResearchPlanning,
    ProductRecommendationExtractor,
    ProductImageFilter,
)
from toil.core.prompt import ResearchUnitPrompt
from toil.tools.base import AsyncTool
from toil.tools.shopping import SearchProductTool
from toil.tools.utils import add_graph_components
 
default_config = {
    "configurable": {"thread_id": "1"},
    "recursion_limit": 100,
}
 
class DeepSearchAgent(BaseModel):
    llm: BaseLanguageModel | RunnableBinding
    graph: CompiledStateGraph
    config: dict = default_config
 
    class Config:
        arbitrary_types_allowed = True
 
    @classmethod
    def create(
        cls,
        llm: BaseLanguageModel | RunnableBinding,
        summarize_llm: BaseLanguageModel | RunnableBinding,
        filter_llm: BaseLanguageModel | RunnableBinding,
        search_tool: AsyncTool,
        tools: list[AsyncTool],
        init_data: dict[str, Any] | None = None,
    ):
        search_product_tool = SearchProductTool.from_llm(llm)
        
        # Build Graph
        graph = cls._create_graph(
            llm=llm,
            summarize_llm=summarize_llm,
            filter_llm=filter_llm,
            search_tool=search_tool,
            search_product_tool=search_product_tool,
            tools=tools,
            init_data=init_data,
            config=default_config,
        )
 
        return DeepSearchAgent(
            llm=llm,
            graph=graph,
        )
 
    @staticmethod
    def _create_graph(
        llm: BaseLanguageModel | RunnableBinding,
        summarize_llm: BaseLanguageModel | RunnableBinding,
        filter_llm: BaseLanguageModel | RunnableBinding,
        search_tool: AsyncTool,
        search_product_tool: AsyncTool,
        tools: list[AsyncTool],
        init_data: dict[str, Any] | None,   
        config: dict,
    ) -> CompiledStateGraph:
        research_unit_execution_nodes = [
            (
                "query_analysis",
                DeepSearchQueryAnalysis(
                    llm=llm,
                    summarize_llm=summarize_llm,
                    prompt=ResearchUnitPrompt(),  # TODO: 이름 node명과 맞게 바꾸기.
                    tools=tools,
                    answer_llm_tags=["research_unit"],
                ),
            ),
            (
                "tool_calling",
                DeepSearchToolCalling(
                    tools=tools,
                    filter_llm=filter_llm,
                ),
            ),
        ]
 
        research_unit_execution_edges = [
            (START, "query_analysis"),
            ("tool_calling", "query_analysis"),
        ]
 
        research_unit_execution_edges_with_conditions = [
            (
                "query_analysis",
                lambda state: (
                    "tool_calling"
                    if hasattr(state.current_task.messages[-1], "tool_calls")
                    and state.current_task.messages[-1].tool_calls
                    else END
                ),
            )
        ]
 
        research_unit_execution_graph = add_graph_components(
            StateGraph(DeepSearchState),
            research_unit_execution_nodes,
            research_unit_execution_edges,
            research_unit_execution_edges_with_conditions,
        )
 
        # Define main graph nodes with improved naming
        main_nodes = [
            ("preliminary_investigation", PreliminaryInvestigation(llm, search_tool)),
            ("research_planning", ResearchPlanning(llm)),
            ("research_unit_execution", research_unit_execution_graph.compile()),
            ("reference_filter", ReferenceFilter(llm=llm)),
            # ("research_unit_synthesis", ResearchUnitSynthesis(llm=llm)),
            ("report_compilation", ReportCompilation(llm)),
            ("report_sections_generation", ReportSectionsGeneration(llm)),
            ("product_recommendation_extractor", ProductRecommendationExtractor(llm=llm, search_product_tool=search_product_tool)),
            ("product_image_filter", ProductImageFilter(llm=llm)),
        ]
 
        # Define main graph edges
        main_edges = [
            (START, "preliminary_investigation"),
            ("preliminary_investigation", "research_planning"),
            ("research_unit_execution", "reference_filter"),
            ("reference_filter", "research_planning"),
            ("report_sections_generation", "report_compilation"),
            ("report_compilation", "product_recommendation_extractor"),  
            ("product_recommendation_extractor", "product_image_filter"),
            ("product_image_filter", END),
        ]
 
        # Define conditional edges
        main_edges_with_conditions = [
            (
                "research_planning",
                lambda state: (
                    "research_unit_execution"
                    if state.current_task
                    else "report_sections_generation"
                ),
            ),
        ]
 
        main_graph = add_graph_components(
            StateGraph(DeepSearchState),
            main_nodes,
            main_edges,
            main_edges_with_conditions,
        )
 
        # Compile the final graph
        compiled_graph = main_graph.compile(
            checkpointer=MemorySaver()
        )  # TODO: TrimmedMemorySaver 적용 가능한지 테스트 필요.
 
        # Update state if init_data is provided
        if init_data:
            # TODO: completed task같은건 다시 load할 필요없이, messages만 유지하면 되지 않을까?
            init_data["messages"] = messages_from_dict(init_data["messages"])
            compiled_graph.update_state(config=config, values=init_data)
 
        return compiled_graph
        
    async def astream_events(
        self, user_input: str, stream_mode: str = "values", version="v2"
    ):
        return self.graph.astream_events(
            input={"messages": [HumanMessage(content=user_input)]},
            config=self.config,
            stream_mode=stream_mode,
            version=version,
        )
        
    def dump(self):
        state = self.graph.get_state(config=self.config).values
        state["messages"] = messages_to_dict(state["messages"])
        return state
        
    def format_reference(self, answer: str, reference_format_string: str) -> tuple[str, list]:
        """
        answer: completed tokens
        reference_format_string: reference replace format string
            example: "[{number}]({link})"
        """
        import re
        
        metadata = []
        state = self.graph.get_state(config=self.config).values
        if references := re.findall(r"\[\^(\d+)\^\]", answer):
            references = {
                i: ref
                for i in sorted(set(map(int, references)))
                for ref in state.get("references", [])
                if ref.get("number") == i and ref.get("source") is not None
            }
 
            for key, value in references.items():
                answer = re.sub(
                    rf"\[\^{key}\^\]",
                    reference_format_string.format(number=key, link=value["source"]),
                    answer,
                )
            answer = re.sub(r"\[\^(\d+)\^\]", "", answer)
            metadata = list(references.values())
 
        return answer, metadata
 
    def get_product_recommendations(self):
        """최종 상태에서 추천 상품 정보를 반환합니다."""
        state = self.graph.get_state(config=self.config).values
        return state.get("product_recommendations", [])
    
    def get_formatted_recommendations(self):
        """추천 상품 정보를 포맷팅된 문자열로 반환합니다."""
        recommendations = self.get_product_recommendations()
        result = []
        
        for item in recommendations:
            query = item.get("query", "")
            products = item.get("products", [])
            
            result.append(f"## 추천 상품: {query}")
            
            for product in products:
                result.append(f"- 상품명: {product.get('title', '정보 없음')}")
                result.append(f"  가격: {product.get('lprice', '정보 없음')}원")
                result.append(f"  쇼핑몰: {product.get('mall', '정보 없음')}")
                result.append(f"  링크: {product.get('link', '정보 없음')}")
                result.append(f"  이미지: {product.get('image', '정보 없음')}")
                result.append("")
            
            result.append("")
        
        return "\n".join(result)