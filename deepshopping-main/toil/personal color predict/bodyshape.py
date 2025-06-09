import os
from langchain_google_genai import ChatGoogleGenerativeAI
import base64
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from IPython.display import Image, display

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
    temperature=0,
)

class BodyShapeInfo(BaseModel):
    body_type: str = Field(description="사람의 체형")

# 스트럭쳐 아웃풋 모델 생성
structured_llm = llm.with_structured_output(BodyShapeInfo)

IMAGE_ANALYSIS_PROMPT = """
당신은 전문 이미지 분석가입니다. 제공된 인물 사진을 아래 기준에 따라 분류하세요.
- {역삼각형, 직사각형, 둥근형, 삼각형, 모래시계형} 중 하나로 구분하고
- 기준은 다음과 같아. 
- 역삼각형(남성형): 어깨 > 가슴 > 허리 > 엉덩이
- 직사각형(직선형): 어깨 ≈ 가슴 ≈ 허리 ≈ 엉덩이
- 둥근형(원형): 허리 ≥ 어깨 ≈ 엉덩이
- 삼각형(배형): 엉덩이 > 허리 > 가슴 > 어깨
- 모래시계형(균형형): 가슴 ≈ 엉덩이 > 허리
"""

def predict_body_shape(image_file_path: str, show_image: bool = False) -> str:
    try:
        # 파일 존재 여부 확인
        if not os.path.exists(image_file_path):
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_file_path}")
        
        # 이미지를 base64로 인코딩
        with open(image_file_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
        
        # 선택적으로 이미지 표시
        if show_image:
            display(Image(filename=image_file_path, width=300, height=300))
        
        # 메시지 생성
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": IMAGE_ANALYSIS_PROMPT,
                },
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{encoded_image}"
                },
            ]
        )
        
        # AI 모델로 분석 실행
        result = structured_llm.invoke([message])
        
        return result.body_type
        
    except FileNotFoundError:
        raise
    except Exception as e:
        raise Exception(f"체형 분석 중 오류가 발생했습니다: {str(e)}")