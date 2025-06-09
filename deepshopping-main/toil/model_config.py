from dataclasses import MISSING, dataclass, field
from typing import Any, ClassVar, Dict, Literal, Type, get_type_hints

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI

model_token_mapping = {
    "gpt-4o": 100000,
    "gpt-4o-mini": 100000,
    "models/gemini-2.0-flash": 1000000,
    "models/gemini-2.5-flash-preview-04-17": 1000000,
}


def get_maximum_tokens_from_model_name(model_name):
    return model_token_mapping[model_name]


def get_maximum_tokens_from_llm(llm):
    model_name = getattr(llm, "model_name", None) or getattr(llm, "model", None)
    if not model_name:
        raise ValueError("model name attribute is required for LLM")

    return get_maximum_tokens_from_model_name(model_name)


# LLM 설정 스키마를 dataclass로 정의
@dataclass
class LLMConfigSchema:
    tags: list[str]

    descriptions: ClassVar[Dict[str, str]] = {
        "tags": "LLM의 Output을 구분하기 위해 적용할 태그. (예: ['answer', 'summarize'])",
    }

    @classmethod
    def get_required_fields(cls) -> Dict[str, Type]:
        return {
            name: f.type
            for name, f in cls.__dataclass_fields__.items()
            if f.default is MISSING and f.default_factory is MISSING
        }

    @classmethod
    def get_schema_dict(cls) -> Dict[str, Dict[str, Any]]:
        """schema 정보 반환."""
        schema = {}
        hints = get_type_hints(cls)

        for field, typ in hints.items():
            if field.startswith("_") or field == "descriptions":
                continue

            field_info = {
                "type": str(typ),
                "required": field in cls.get_required_fields(),
            }

            # 기본값이 있으면 추가
            if hasattr(cls, field) and getattr(cls, field) is not None:
                field_info["default"] = getattr(cls, field)

            # 설명이 있으면 추가
            if field in cls.descriptions:
                field_info["description"] = cls.descriptions[field]

            schema[field] = field_info

        return schema


@dataclass
class AzureOpenAIConfigSchema(LLMConfigSchema):
    azure_endpoint: str
    openai_api_key: str
    openai_api_version: str
    openai_api_type: str
    model_name: str
    deployment_name: str
    default_headers: dict = field(default_factory={})

    descriptions: ClassVar[Dict[str, str]] = {
        **LLMConfigSchema.descriptions,
        "default_headers": "LLM 토큰 계산을 위한 헤더. (예: {'Ocp-Apim-Subscription-Key': 'your-subscription-key'})",
    }


@dataclass
class DeepseekConfigSchema(LLMConfigSchema):
    base_url: str
    fireworks_api_key: str
    model: str

    descriptions: ClassVar[Dict[str, str]] = {
        **LLMConfigSchema.descriptions,
    }


@dataclass
class GeminiConfigSchema(LLMConfigSchema):
    google_api_key: str
    model: str

    descriptions: ClassVar[Dict[str, str]] = {
        **LLMConfigSchema.descriptions,
    }


ValidLLMType = Literal["azure-openai-4o", "azure-openai-4o-mini"]

# LLM 타입별 설정 스키마 매핑
LLM_CONFIG_SCHEMAS: Dict[ValidLLMType, Type[LLMConfigSchema]] = {
    "azure-openai-4o": AzureOpenAIConfigSchema,
    "azure-openai-4o-mini": AzureOpenAIConfigSchema,
    "gemini-2.0-flash": GeminiConfigSchema,
    "gemini-2.5-flash-preview-04-17": GeminiConfigSchema,
}

# LLM 타입별 클래스 매핑
LLM_REGISTRY = {
    "azure-openai-4o": AzureChatOpenAI,
    "azure-openai-4o-mini": AzureChatOpenAI,
    "gemini-2.0-flash": ChatGoogleGenerativeAI,
    "gemini-2.5-flash-preview-04-17": ChatGoogleGenerativeAI,
}

isToolCallingFeatureEnabled = {
    AzureChatOpenAI: True,
    ChatGoogleGenerativeAI: True,  # Gemini
}


def get_llm_config_schema(llm_type: ValidLLMType) -> Dict[str, Dict[str, Any]]:
    """특정 LLM 타입에 대한 설정 스키마 반환

    Args:
        llm_type: LLM 타입

    Returns:
        필드별 타입, 필수 여부, 기본값, 설명 등을 포함한 스키마 딕셔너리

    Examples:
        >>> schema = get_llm_config_schema("azure-openai-4o")
        >>> print(schema["api_base"])
        {'type': "<class 'str'>", 'required': True, 'description': 'Azure OpenAI API 엔드포인트 URL'}
    """
    if llm_type not in LLM_CONFIG_SCHEMAS:
        raise ValueError(f"Unsupported LLM type: {llm_type}")

    schema_class = LLM_CONFIG_SCHEMAS[llm_type]
    return schema_class.get_schema_dict()


def validate_llm_config(llm_type: ValidLLMType, llm_config: dict) -> Dict[str, Any]:
    """LLM 설정 유효성 검사 및 기본값 적용

    Args:
        llm_type: LLM 타입
        llm_config: LLM 설정 딕셔너리

    Returns:
        검증되고 기본값이 적용된 설정 딕셔너리

    Raises:
        ValueError: 필수 필드가 누락되었거나 타입이 잘못된 경우
    """
    if llm_type not in LLM_CONFIG_SCHEMAS:
        raise ValueError(f"Unsupported LLM type: {llm_type}")

    schema_class = LLM_CONFIG_SCHEMAS[llm_type]
    schema = schema_class.get_schema_dict()

    # 필수 필드 검사
    for field, info in schema.items():
        if info.get("required", False) and field not in llm_config:
            raise ValueError(
                f"Missing required field '{field}' for {llm_type} configuration"
            )

    # 기본값 적용 (예시 - 실제 구현 시 타입 체크 등을 추가할 수 있음)
    result = {}
    for field, info in schema.items():
        if field in llm_config:
            result[field] = llm_config[field]
        elif "default" in info and info["default"] is not None:
            result[field] = info["default"]

    # llm_config에는 있지만 schema에는 없는 추가 필드도 포함
    for field, value in llm_config.items():
        if field not in result:
            result[field] = value

    return result
