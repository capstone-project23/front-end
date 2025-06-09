import json
from typing import Any, List, Optional, Type, Dict
 
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun
from langchain.schema import BaseMessage
from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import RunnableBinding
from pydantic import BaseModel, Field
 
from toil.tools.base import AsyncTool
from toil.tools.utils import noop
 
import json
import urllib.request
from typing import List, Dict, Optional, Type
from pydantic import BaseModel, Field
from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import RunnableBinding
 
class SearchProductTool(AsyncTool):
    name: str = "search_product"
    description: str = "특정 상품을 정확도순으로 검색하여 이미지 URL과 함께 결과를 보여줍니다."
    
    class PriceArgs(BaseModel):
        query: str = Field(..., description="검색할 상품명")
        verbose: bool = Field(default=False)
    
    args_schema: Type[BaseModel] = PriceArgs
    filter_llm: Any  # 필터링에 사용할 LLM
 
    client_id: str = "q1had9wOvT_H9pnCpt__"
    client_secret: str = "d20dBOzzBB"
    
    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel | RunnableBinding,
    ) -> "SearchProductTool":
        return cls(filter_llm=llm)
    
    def gen_search_url(self, search_text: str, start_num: int, disp_num: int, sort: str) -> str:
        base = 'https://openapi.naver.com/v1/search'
        node = '/shop.json'
        param_query = '?query=' + urllib.parse.quote(search_text)
        param_start = '&start=' + str(start_num)
        param_disp = '&display=' + str(disp_num)
        param_sort = '&sort=' + str(sort)
        return base + node + param_query + param_disp + param_start + param_sort
 
    def get_result_onpage(self, url: str) -> Dict:
        request = urllib.request.Request(url)
        request.add_header('X-Naver-Client-Id', self.client_id)
        request.add_header('X-Naver-Client-Secret', self.client_secret)
        response = urllib.request.urlopen(request)
        result = json.loads(response.read().decode('utf-8'))
        return result
 
    def delete_tag(self, input_str) -> str:
        input_str = input_str.replace('<b>', '')
        input_str = input_str.replace('</b>', '')
        input_str = input_str.replace('\xa0', '')
        return input_str
    
    def get_fields(self, json_data: Dict) -> List[Dict]:
        products = [
            {
                'title': self.delete_tag(each['title']),
                'link': each['link'],
                'image': each['image'],
                'lprice': int(each['lprice']),
                'mall': each['mallName']
            }
            for each in json_data['items']
        ]
        return products
    
    async def _arun(
        self,
        query: str,
        verbose: bool = False,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        dispatcher = adispatch_custom_event if verbose else noop
            
        await dispatcher("event", {"speak": f"'{query}'에 맞는 상품을 검색하고 있어요."})
        
        try:
            # 2개의 상품만 가져오기
            search_url = self.gen_search_url(query, 1, 1, 'sim')
            result = self.get_result_onpage(search_url)
            
            if 'items' not in result or not result['items']:
                return json.dumps({
                    "filtered_products": [],
                    "message": "상품을 찾을 수 없습니다."
                }, ensure_ascii=False)

            products = self.get_fields(result)
            
            result = {
                "filtered_products": products,
                "total_products": len(products)
            }

            await dispatcher("event", {"speak": f"총 {len(products)}개의 상품을 찾았습니다."})

            return json.dumps(result, ensure_ascii=False)
            
        except Exception as e:
            error_message = f"상품 검색 중 오류가 발생했습니다: {str(e)}"
            await dispatcher("event", {"speak": error_message})
            return json.dumps({"error": error_message}, ensure_ascii=False)