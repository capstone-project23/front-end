from typing import Any

from toil.core.agent import DeepSearchAgent

from toil.model_config import LLM_REGISTRY, ValidLLMType, validate_llm_config
from toil.tools import (
    GoogleSerperNewsResult,
    GoogleSerperSearchResult,
    RefSummarizeTool,
    URLSummarizeTool,
)

def _create_llm(llm_type: ValidLLMType, llm_config: dict):
    """LLM 인스턴스 생성

    Args:
        llm_type: LLM 타입
        llm_config: LLM 설정 딕셔너리

    Returns:
        LLM 인스턴스

    Examples:
        >>> config = {
        ...     "api_base": "https://example.openai.azure.com",
        ...     "api_key": "your-api-key",
        ...     "api_version": "2023-05-15",
        ...     "model_name": "gpt-4o",
        ...     "deployment_name": "gpt4o",
        ...     "temperature": 0.7
        ... }
        >>> llm = _create_llm("azure-openai-4o", config)
    """
    if llm_type not in LLM_REGISTRY:
        raise ValueError(f"Unsupported LLM type: {llm_type}")

    # 설정 유효성 검사 및 기본값 적용
    validated_config = validate_llm_config(llm_type, llm_config)

    return LLM_REGISTRY[llm_type](**validated_config)

def get_deepsearch(
    main_llm_type: ValidLLMType,
    main_llm_config: dict,
    sub_llm_type: ValidLLMType,
    sub_llm_config: dict,
    serper_api_key: str,
    init_data: dict[str, Any] | None = None,
):

    answer_llm = _create_llm(
        llm_type=main_llm_type,
        llm_config={
            **main_llm_config,
            "temperature": 1,  # Gemini 2.0 Flash Table generation bug 방지를 위한 temperature.
            "max_tokens": 8192,
        },
    )

    summarize_llm = _create_llm(
        llm_type=sub_llm_type,
        llm_config={
            **sub_llm_config,
            "temperature": 0.1,
            "max_tokens": 4096,
        },
    ).with_config(tags=["summarize"])

    filter_llm = _create_llm(
        llm_type=sub_llm_type,
        llm_config={
            **sub_llm_config,
            "temperature": 0.1,
            "max_tokens": 4096,
            "model_kwargs": {"response_format": {"type": "json_object"}},
        },
    ).with_config(tags=["filter_contents"])

    search_tool = GoogleSerperSearchResult.from_api_key(
        api_key=serper_api_key,
        k=15,
    )

    google_news_tool = GoogleSerperNewsResult.from_api_key(
        api_key=serper_api_key,
        k=15,
    )

    url_summary_tool = URLSummarizeTool.from_llm(
        llm=summarize_llm,
        total_limit_tokens=4096,
    )

    ref_summary_tool = RefSummarizeTool.from_llm(
        llm=summarize_llm,
        total_limit_tokens=4096,
    )
    
    tools = [
        search_tool,
        google_news_tool,
        ref_summary_tool,
        url_summary_tool,
    ]

    return DeepSearchAgent.create(
        llm=answer_llm,
        summarize_llm=summarize_llm,
        filter_llm=filter_llm,
        search_tool=search_tool,
        tools=tools,
        init_data=init_data,
    )