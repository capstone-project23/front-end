from datetime import datetime, timedelta, timezone
from typing import Any

from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.schema import BaseMessage, HumanMessage
from abc import ABC, abstractmethod

PRELIMINARY_INVESTIGATION_PROMPT = """
당신은 패션 전문가로서 사용자의 상의 관련 요청에 대해 메타지식을 수집하는 임무를 맡았습니다.
메타지식이란 상의 제품을 체계적으로 이해하고 분류하는 데 필요한 기본 프레임워크와 개념적 구조를 의미합니다.
이 예비 단계는 효과적인 상품 추천 방향을 설정하기 위한 필수적인 기반을 제공합니다.

당신의 목표는 다음과 같습니다:
1. 상의 제품을 이해하고 분류하는 데 사용되는 주요 기준, 분류 시스템 및 주요 카테고리를 파악하고 제시하는 것
2. 상품의 상세 정보(소재, 디자인, 가격대, 사이즈, 컬러 등)를 체계적으로 조사할 방법을 구성하는 것
3. 상품에 대한 사용자 리뷰와 평가를 수집하고 분석할 프레임워크를 설정하는 것

이는 단순한 사실 정보가 아니라 패션 분야 전문가들이 상의를 어떻게 구조화하고, 상품 정보와 리뷰를 어떻게 분석하는지에 대한 지식입니다.

상의 관련 메타지식 키워드를 파악하고 이를 사용하여 검색을 수행하세요. 이 메타지식은 체계적인 접근 방식을 가능하게 하며 효율적인 제품 추천 방향을 설정하는 데 중요합니다.

<사용자 정보>
{user_info}
</사용자 정보>

<예시>
사용자 질문: "일상적으로 입을 만한 셔츠를 추천해주세요."
→ 메타지식 키워드: 
- 셔츠 스타일 분류, 셔츠 소재 특성, 체형별 적합한 셔츠 핏, 피부톤별 셔츠 색상 선택 기준, 셔츠 가격대 구분, 캐주얼/포멀 셔츠 구분 기준
- 인기 셔츠 브랜드, 셔츠 품질 평가 기준, 내구성 및 관리 용이성
- 셔츠 온라인 리뷰 분석, 소비자 만족도 지표, 착용감 평가 기준

<사용자 질문>
{topic}
</사용자 질문>
"""

RESEARCH_PLANNING_PROMPT = """
당신은 패션 제품 추천 전문가로서 사용자 맞춤형 상의 추천 계획을 수립하는 역할을 맡았습니다.

<사용자 정보>
{user_info}
</사용자 정보>

<주제>
{topic}
</주제>

<기초 지식>
{base_knowledge}
</기초 지식>

<완료된 작업>
{completed_tasks}
</완료된 작업>

<미완료 작업>
{incomplete_tasks}
</미완료 작업>

<작업>
제공된 주제, 기초 지식, 완료된 작업 및 미완료 작업을 분석하여 연구 계획을 적응적으로 업데이트하세요.

각 반복에서:
1. 완료된 작업과 그 결과를 검토하여 이미 달성한 내용을 파악하세요.
2. 완료된 작업에서 얻은 통찰력을 바탕으로 현재 미완료 작업 목록이 여전히 관련성이 있고 적절한지 평가하세요.
3. 다음과 같이 업데이트된 연구 계획을 생성하세요:
   - 관련성 있는 미완료 작업 유지
   - 완료된 작업에서 나온 통찰력이나 질문을 기반으로 새로운 연구 단계 추가
   - 더 이상 필요하지 않거나 개선이 필요한 작업 제거 또는 수정
   - 필요한 경우 더 나은 논리적 진행을 위해 작업 재정렬

반드시 다음 카테고리의 작업들이 포함되도록 계획을 세우세요:
1. 상품 기본 정보 조사: 스타일, 디자인, 소재, 가격대, 사이즈, 컬러 등 상품의 기본적인 특성
2. 상품 상세 사양 조사: 제조사 정보, 상세 소재 구성, 관리 방법, 특수 기능 등
3. 사용자 리뷰 및 평가 분석: 실제 사용자들의 경험, 만족도, 불만 사항, 장단점 등
4. 전문가 의견 및 평가: 패션 전문가, 블로거, 매체 등의 평가와 의견
5. 가격 대비 가치 분석: 유사 제품과의 가격 비교, 가성비 분석 등

연구 계획 관리 지침:
1. 연구는 논리적 흐름을 유지해야 하며, 각 단계는 이전 결과를 기반으로 해야 합니다.
2. 각 작업은 한국어로 간단한 명령문으로 작성되어야 하며, 항상 "조사하세요."로 끝나야 합니다.
3. 완료된 작업에서 예상치 못한 발견이나 새로운 관심 영역이 드러나면 이러한 영역을 탐색하기 위한 적절한 새 작업을 추가하세요.
4. 완료된 작업에서 특정 계획된 연구 방향이 불필요하거나 중복됨을 보여주는 경우 해당 작업을 제거하세요.
5. 각 업데이트된 작업이 상세한 제품 추천 보고서 작성에 가치 있는 정보를 제공하는지 확인하세요.
6. 요약이나 보고 단계를 포함하지 마세요.
7. 대명사나 모호한 용어를 사용하지 마세요.
8. 여러 개별 제품에 대해 유사한 단계를 반복하지 마세요. 대신 정보를 효과적으로 일반화하고 그룹화하세요.

중요:
- 각 반복은 축적된 지식을 기반으로 더 세밀하고 목표가 명확한 연구 계획을 생성해야 합니다.
- 계획은 미리 정해진 구조를 엄격하게 따르기보다는 나타나는 통찰력에 역동적으로 반응해야 합니다.
- 모든 필요한 연구가 완료되고 보고서에 의미 있게 기여할 추가 작업이 없는 경우 빈 목록 []을 반환하세요.
</작업>
"""

RESEARCH_UNIT_PROMPT = """
당신은 패션 전문 AI 리서치 어시스턴트로서 사용자 맞춤형 상의 추천 프로젝트의 특정 측면을 조사하는 임무를 맡았습니다. 당신의 목표는 제공된 도구를 사용하여 할당된 작업에 대한 철저한 조사를 수행하고, 충분한 정보를 수집할 때까지 조사를 계속하는 것입니다.

다음은 연구 중인 주제입니다:
<주제>
{topic}
</주제>

<사용자 정보>
{user_info}
</사용자 정보>

이 주제 내에서 당신의 특정 작업은 다음과 같습니다:
<작업>
{task_description}
</작업>

상품 정보 조사 지침:
1. 상품의 기본 정보(브랜드, 모델명, 가격, 출시일 등)를 명확하게 식별하세요.
2. 상품의 상세 사양(소재, 디자인, 사이즈, 컬러 옵션 등)을 체계적으로 수집하세요.
3. 제조사가 강조하는 주요 특징과 장점을 파악하세요.

리뷰 및 평가 조사 지침:
1. 다양한 사용자 리뷰를 수집하고 공통적인 피드백 패턴을 식별하세요.
2. 리뷰를 긍정적인 측면과 부정적인 측면으로 분류하세요.
3. 전문가 리뷰와 일반 사용자 리뷰를 구분하고 다른 관점을 비교하세요.
4. 리뷰의 신뢰성과 일관성을 평가하세요.

모든 정보는 객관적이고 편향되지 않아야 하며, 사실에 기반해야 합니다. 개인적인 의견이나 추측은 피하고, 신뢰할 수 있는 출처에서 정보를 수집하세요.

제공된 도구를 사용하여 관련 정보를 검색하고, 정보가 충분히 수집될 때까지 조사를 계속하세요. 정보가 불완전하거나 불충분한 경우, 추가 검색 쿼리를 작성하여 더 깊이 조사하세요.
"""

RESEARCH_UNIT_SYNTHESIS_PROMPT = """
당신은 패션 전문 리서치 요약 전문가입니다.
특정 상의 추천 관련 주제에 대한 연구 과정에서 수집된 정보와 참고 문헌을 바탕으로 종합적이고 체계적인 요약을 작성해야 합니다.

<요약 지침>
1. 연구 주제와 특정 연구 작업을 명확히 이해하세요.
2. 수집된 정보(메시지 및 참고 문헌)를 철저히 분석하세요.
3. 중요한 사실, 주요 발견, 통계, 트렌드, 인용문 등을 추출하세요.
4. 요약은 다음 구조를 따라야 합니다:
    - 주요 발견: 연구 작업과 관련된 3-5가지 가장 중요한 발견
    - 상세 분석: 주제의 다양한 측면에 대한 체계적인 구성
    - 데이터 포인트: 관련 수치, 통계, 비율 등
    - 구체적 제품 정보: 브랜드명, 제품명, 가격, 특징 등 구체적인 제품 정보
    - 한계: 연구 과정에서 확인된 정보 제한이나 부족
5. 요약은 명확하고 간결해야 하며, 전문적인 어조를 유지해야 합니다.
6. 한국어로 작성하되, 필요한 경우 영어 기술 용어를 사용할 수 있습니다.
7. 인용이나 참조를 포함할 때는 소스 번호를 [^번호^] 형식으로 표시하세요.
8. 반드시 구체적인 제품 정보와 이름을 포함하세요. 이는 최종 보고서에 필수적입니다.

최종 요약은 나중에 종합적인 보고서 작성에 직접 사용될 수 있도록 충분히 상세하고 체계적이어야 합니다.
</요약 지침>

<사용자 정보>
{user_info}
</사용자 정보>

<연구 주제>
{topic}
</연구 주제>

<연구 작업>
{task_description}
</연구 작업>

<수집된 메시지>
{messages}
</수집된 메시지>

<참고 문헌>
{references}
</참고 문헌>

위의 정보를 바탕으로 체계적이고 종합적인 연구 요약을 작성해 주세요. 구체적인 제품 정보와 이름을 반드시 포함하세요.
"""

REPORT_COMPILATION_PROMPT = """
당신은 꼼꼼하고 전문적인 한국어 패션 리서치 작가입니다.
당신의 임무는 사용자 맞춤형 상의 추천에 관한 상세하고, 잘 구성되며, 포괄적인 최종 보고서를 한국어로 작성하는 것입니다.
보고서는 철저히 조사되고, 논리적으로 구성되며, 아래 대화에서 모든 핵심 포인트를 통합해야 합니다.

<사용자 정보>
{user_info}
</사용자 정보>

<주제>
{topic}
</주제>

다음은 보고서의 섹션입니다. 최종 보고서를 작성할 때 이 섹션들을 참조하세요. 보고서는 다음 섹션을 준수해야 합니다:
<섹션>
{sections}
</섹션>

다음은 주제에 관한 참조 자료입니다. 최종 보고서를 작성할 때 이 참조 자료를 참조하세요. 보고서는 다음 요구 사항을 준수해야 합니다:
<참조>
{references}
</참조>

**보고서 요구 사항:**
1. **길이**: 최소 5,000자 이상의 한국어로 작성하세요(또는 다른 최소 길이 지정, 예: 3,000자 또는 ~2-3페이지).
2. **구조**:
   - 서론 (사용자 프로필, 요구사항 분석, 목표 설정)
   - 본문 (추천 브랜드 및 제품 소개, 제품별 세부 분석, 스타일 제안)
   - 결론 (최종 추천, 구매 가이드)
3. **작성 스타일**:
   - 전체를 한국어로 작성하세요.
   - 명확하고 간결하며 논리적인 전문적 어조를 사용하세요.
   - 각 섹션을 적절한 제목과 부제목으로 구성하세요.
   - 필요한 경우 설명, 예시, 간략한 인용이나 참조(가상의 출처일 수 있음)를 포함하세요.
   - **절대로** 응답 끝에 각주를 포함하지 마세요. 번호와 링크 간의 관계는 자동으로 사용자에게 전달됩니다.
   - 출처를 인용할 때는 각 개별 참조에 대해 [^%d^] 형식을 사용하세요. 예를 들어, 첫 번째 참조에는 [^1^], 두 번째 참조에는 [^2^]를 사용하세요.
   - **중요**: 항상 각 출처를 개별적으로 인용하세요. 여러 참조를 하나의 인용으로 결합하지 마세요. 예를 들어, 여러 출처를 인용할 때는 [^1^][^2^]와 같이 사용하고, [^1^, ^2^]와 같이 사용하지 마세요.
   - 인용에 선택적이어야 합니다: 각 포인트에 대해 가장 관련성이 높고 고품질의 출처만 인용하세요. 하나의 문장이나 단락에 너무 많은 출처를 인용하지 마세요.
   - 여러 출처에서 유사한 정보를 얻을 경우, 가장 권위 있거나 포괄적인 출처만 선택하여 인용하세요.
4. **목적**: 리서치 대화에서 정보를 하나의 일관된 최종 보고서로 종합하세요.
5. **추가 지침**:
   - 리서치 대화에서 모든 중요한 개념과 분석을 통합하세요.
   - 보고서를 시각적으로나 구조적으로 더 풍부하게 만들기 위해 목록, 표 등을 사용하는 것을 고려하세요.
   - 주요 요점이나 추천으로 결론을 맺으세요.
   - 마크다운 표의 경우, GFM 형식이어야 하며 각 " |" 사이에 정확히 하나의 공백만 있어야 합니다. 이것을 잊지 마세요, 그렇지 않으면 처벌받을 것입니다.
   - 마크다운 표의 경우, 각 헤더는 반드시 | **헤더 제목** | 형식을 따라야 합니다.
   - **반드시 구체적인 제품 정보와 이름을 포함하세요.** 브랜드명, 제품명, 가격, 소재, 디자인 특징 등 구체적인 정보를 상세히 기술하세요.
   - 최소 5개 이상의 구체적인 제품을 추천하고, 각 제품에 대한 상세 정보를 제공하세요.
   - 제품 추천 시 사용자 정보를 고려하여 맞춤형 추천을 제공하세요.
"""

REFERENCE_FILTER_PROMPT = """
당신은 패션 상의 추천 리서치 참조 필터 전문가입니다.
주어진 연구 주제와 작업 설명을 바탕으로 최종 보고서에 불필요하거나 관련성이 낮은 참조 자료를 식별하세요.

<필터링 지침>
1. 주제와 작업에 대한 관련성, 정보 품질 및 중복성을 기준으로 각 참조 자료를 평가하세요.
2. 다음에 해당하는 참조 자료를 제외하는 것을 고려하세요:
   - 주제나 작업과 직접적인 관련이 없는 자료
   - 오래되었거나 부정확한 정보를 포함하는 자료
   - 다른 참조 자료에서 발견된 내용을 중복하는 자료
   - 정보의 깊이나 품질이 현저히 낮은 자료
3. 각 제외 결정에 대해 간결하고 명확한 이유를 제공하세요.
4. 제외할 참조 자료의 인덱스와 제외 이유를 반환하세요.
5. 유저 정보와 관련 없는 정보는 제외하세요.
</필터링 지침>

<연구 주제>
{topic}
</연구 주제>

<작업 설명>
{task_description}
</작업 설명>

<유저 정보>
{user_info}
</유저 정보>

이 정보를 바탕으로 최종 보고서에 불필요한 참조 자료를 식별하고 각 제외에 대한 이유를 한국어로 제공하세요.
"""

REPORT_SECTION_GENERATION_PROMPT = """
당신은 한국어 패션 보고서 구조 디자이너입니다.

다음 주제에 대한 검색 결과를 바탕으로 상세한 계층적 보고서 개요를 생성하세요:

<주제>
{topic}
</주제>

<사용자 정보>
{user_info}
</사용자 정보>

이 보고서는 다음 작업에 따라 안내된 웹 검색을 통해 수집된 정보를 요약하고 종합합니다:

<작업>
{tasks}
</작업>

이러한 검색에서 수집된 주요 정보, 데이터 및 참조는 다음과 같습니다:

<참조>
{references}
</참조>

검색 결과를 위해 특별히 맞춤화된 논리적 보고서 개요를 번호가 매겨진 계층적 섹션 목록으로 생성하세요.
"연구 목표," "방법론," 또는 "결론"과 같은 학술 연구 섹션을 포함하지 마세요.
하위 섹션을 포함하지 마세요.

계층 구조를 명확하게 나타내기 위해 다음 형식을 사용하세요:

- '1. 섹션 제목'
- '2. 섹션 제목'

개요는 검색 결과를 관련 카테고리, 주제, 트렌드, 비교, 실용적인 통찰력 또는 추천 사항으로 명확하게 제시, 구성 및 종합하는 데 중점을 두어야 하며, 제공된 주제, 작업 및 참조를 직접적으로 반영해야 합니다.

이 보고서의 최종 목적은 사용자 맞춤형 상의 추천이므로, 다음과 같은 섹션이 포함되어야 합니다:
1. 사용자 프로필 및 요구 분석
2. 추천 브랜드 개요
3. 제품별 상세 분석 (가격, 디자인, 소재 등)
4. 사용자 특성별 적합성 평가
5. 스타일링 제안
6. 최종 추천 및 구매 가이드

반드시 구체적인 제품 정보와 이름을 포함할 수 있는 섹션을 만드세요. 예를 들어 '3. 추천 제품 상세 분석' 또는 '4. 브랜드별 추천 제품' 등의 섹션을 포함하세요.
"""

SUMMARY_SYSTEM_PROMPT = """
Provide a query-related summary of the given text.

On summary content requirements:
- The summary should cover the key points and main ideas related to the query presented in the original text.
- The length of the summary should be appropriate for the length and complexity of the original text.
- As a result of the summary, Korean should be used mainly.
- Ensure the summary directly connects to the question being asked.
- Include only details that help answer the user's specific query.
- Maintain proper context while filtering out information unrelated to the question.

For the summary, you recommend following these rules:
1. Split content into sections with titles. Section titles act as signposts, telling readers whether to focus in or move on.
- Prefer titles with informative sentences over abstract nouns. For example, if you use a title like "Results", a reader will need to hop into the following text to learn what the results actually are. In contrast, if you use the title "Streaming reduced time to first token by 50%", it gives the reader the information immediately, without the burden of an extra hop.
- Include a table of contents. Tables of contents help readers find information faster, akin to how hash maps have faster lookups than linked lists. Tables of contents also have a second, oft overlooked benefit: they give readers clues about the doc, which helps them understand if it's worth reading.
- Keep paragraphs short. Shorter paragraphs are easier to skim. If you have an essential point, consider putting it in its own one-sentence paragraph to reduce the odds it's missed. Long paragraphs can bury information.
- Begin paragraphs and sections with short topic sentences that give a standalone preview. When people skim, they look disproportionately at the first word, first line, and first sentence of a section. Write these sentences in a way that don't depend on prior text. For example, consider the first sentence "Building on top of this, let's now talk about a faster way." This sentence will be meaningless to someone who hasn't read the prior paragraph. Instead, write it in a way that can understood standalone: e.g., "Vector databases can speed up embeddings search."
- Put topic words at the beginning of topic sentences. Readers skim most efficiently when they only need to read a word or two to know what a paragraph is about. Therefore, when writing topic sentences, prefer putting the topic at the beginning of the sentence rather than the end. For example, imagine you're writing a paragraph on vector databases in the middle of a long article on embeddings search. Instead of writing "Embeddings search can be sped up by vector databases" prefer "Vector databases speed up embeddings search." The second sentence is better for skimming, because it puts the paragraph topic at the beginning of the paragraph.
- Put the takeaways up front. Put the most important information at the tops of documents and sections. Don't write a Socratic big build up. Don't introduce your procedure before your results.
- Use bullets and tables. Bulleted lists and tables make docs easier to skim. Use them frequently.
- Bold important text. Don't be afraid to bold important text to help readers find it.

2. Write well
- Badly written text is taxing to read. Minimize the tax on readers by writing well.
- Keep sentences simple. Split long sentences into two. Cut adverbs. Cut unnecessary words and phrases. Use the imperative mood, if applicable. Do what writing books tell you.
- Write sentences that can be parsed unambiguously. For example, consider the sentence "Title sections with sentences." When a reader reads the word "Title", their brain doesn't yet know whether "Title" is going to be a noun or verb or adjective. It takes a bit of brainpower to keep track as they parse the rest of the sentence, and can cause a hitch if their brain mispredicted the meaning. Prefer sentences that can be parsed more easily (e.g., "Write section titles as sentences") even if longer. Similarly, avoid noun phrases like "Bicycle clearance exercise notice" which can take extra effort to parse.
- Avoid left-branching sentences. Linguistic trees show how words relate to each other in sentences. Left-branching trees require readers to hold more things in memory than right-branching sentences, akin to breadth-first search vs depth-first search. An example of a left-branching sentence is "You need flour, eggs, milk, butter and a dash of salt to make pancakes." In this sentence you don't find out what 'you need' connects to until you reach the end of the sentence. An easier-to-read right-branching version is "To make pancakes, you need flour, eggs, milk, butter, and a dash of salt." Watch out for sentences in which the reader must hold onto a word for a while, and see if you can rephrase them.
- Avoid demonstrative pronouns (e.g., "this"), especially across sentences. For example, instead of saying "Building on our discussion of the previous topic, now let's discuss function calling" try "Building on message formatting, now let's discuss function calling." The second sentence is easier to understand because it doesn't burden the reader with recalling the previous topic. Look for opportunities to cut demonstrative pronouns altogether: e.g., "Now let's discuss function calling."
- Be consistent. Human brains are amazing pattern matchers. Inconsistencies will annoy or distract readers. If we use Title Case everywhere, use Title Case. If we use terminal commas everywhere, use terminal commas. If all of the Cookbook notebooks are named with underscores and sentence case, use underscores and sentence case. Don't do anything that will cause a reader to go 'huh, that's weird.' Help them focus on the content, not its inconsistencies.
- Don't tell readers what they think or what to do. Avoid sentences like "Now you probably want to understand how to call a function" or "Next, you'll need to learn to call a function." Both examples presume a reader's state of mind, which may annoy them or burn our credibility. Use phrases that avoid presuming the reader's state. E.g., "To call a function, …"

3. Be broadly helpful
- People come to documentation with varying levels of knowledge, language proficiency, and patience. Even if we target experienced developers, we should try to write docs helpful to everyone.
- Write simply. Explain things more simply than you think you need to. Many readers might not speak English as a first language. Many readers might be really confused about technical terminology and have little excess brainpower to spend on parsing English sentences. Write simply. (But don't oversimplify.)
- Avoid abbreviations. Write things out. The cost to experts is low and the benefit to beginners is high. Instead of IF, write instruction following. Instead of RAG, write retrieval-augmented generation (or my preferred term: the search-ask procedure).
- Offer solutions to potential problems. Even if 95% of our readers know how to install a Python package or save environment variables, it can still be worth proactively explaining it. Including explanations is not costly to experts—they can skim right past them. But excluding explanations is costly to beginners—they might get stuck or even abandon us. Remember that even an expert JavaScript engineer or C++ engineer might be a beginner at Python. Err on explaining too much, rather than too little.
- Prefer terminology that is specific and accurate. Jargon is bad. Optimize the docs for people new to the field, instead of ourselves. For example, instead of writing "prompt", write "input." Or instead of writing "context limit" write "max token limit." The latter terms are more self-evident, and are probably better than the jargon developed in base model days.
- Keep code examples general and exportable. In code demonstrations, try to minimize dependencies. Don't make users install extra libraries. Don't make them have to refer back and forth between different pages or sections. Try to make examples simple and self-contained.
- Prioritize topics by value. Documentation that covers common problems—e.g., how to count tokens—is magnitudes more valuable than documentation that covers rare problems—e.g., how to optimize an emoji database. Prioritize accordingly.
- Don't teach bad habits. If API keys should not be stored in code, never share an example that stores an API key in code.
- Introduce topics with a broad opening. For example, if explaining how to program a good recommender, consider opening by briefly mentioning that recommendations are widespread across the web, from YouTube videos to Amazon items to Wikipedia. Grounding a narrow topic with a broad opening can help people feel more secure before jumping into uncertain territory. And if the text is well-written, those who already know it may still enjoy it.

4. Break these rules when you have a good reason
- Ultimately, do what you think is best. Documentation is an exercise in empathy. Put yourself in the reader's position, and do what you think will help them the most.
"""

IMAGE_EVALUATE_PROMPT = """
당신은 가상피팅용 의류 이미지 평가 전문가입니다.

제공된 이미지가 가상피팅에 적합한 의류 이미지인지 다음 3가지 기준으로 평가해주세요:

<사용자 정보>
{gender}
</사용자 정보>

1. **성별에 맞는 옷인가? (gender_appropriate)**: 
   - 사용자 성별에 맞는 옷인가?
   - 남성용/여성용/유니섹스 구분

2. **의류 완전성 (garment_completeness)**:
   - 옷의 전체 형태(앞면, 소매, 밑단 등)가 완전히 보이는가?
   - 잘리거나 가려진 부분이 없는가?
   - 앞면이 제대로 보이는가?

3. **단일 의류 여부 (single_garment)**:
   - 이미지에 옷이 정확히 한 개만 있는가?
   - 여러 벌의 옷이나 액세서리가 함께 있지 않은가?
   - ***상의 1개와 바지1개 조합까지는 괜찮음***

각 항목을 True 또는 False로 평가해주세요.
"""

SUMMARY_HUMAN_PROMPT = (
    "Here are some of the content and topic you can use.\n\n"
    "User query: \n"
    "{user_query}\n\n"
    "Content: \n"
    "{context}"
)

class BasePrompt(ABC):
    def __init__(self):
        self.prompt = None
        self.initialize_prompt()

    @abstractmethod
    def initialize_prompt(self) -> None:
        pass

    def format_messages(self, **kwargs: dict[str, Any]):
        pass

    def get_prompt_template(self):
        return self.prompt

class PreliminaryInvestigationPrompt(BasePrompt):
    def initialize_prompt(self) -> None:
        self.prompt = ChatPromptTemplate.from_messages(
            [
                HumanMessagePromptTemplate.from_template(
                    PRELIMINARY_INVESTIGATION_PROMPT
                ),
            ]
        )

    def format_messages(
        self,
        topic: str,
        user_info: dict = None,
    ) -> list:
        base_prompt = self.get_prompt_template()

        time_kst = datetime.now().astimezone(timezone(timedelta(hours=9)))
        time_prompt = HumanMessage(
            content=f"The current time and date is {time_kst.strftime('%c')}",
        )

        messages: list[BaseMessage] = base_prompt + time_prompt

        # user_info가 None인 경우 빈 딕셔너리로 초기화
        if user_info is None:
            user_info = {}

        return messages.format_messages(topic=topic, user_info=user_info)


class ResearchPlanningPrompt(BasePrompt):
    def initialize_prompt(self) -> None:
        self.prompt = ChatPromptTemplate.from_messages(
            [
                HumanMessagePromptTemplate.from_template(RESEARCH_PLANNING_PROMPT),
            ]
        )

    def format_messages(
        self,
        topic: str,
        base_knowledge: str,
        completed_tasks: list["ResearchUnitState"],
        incomplete_tasks: list["ResearchUnitState"],
        user_info: dict = None,
    ) -> list:
        base_prompt = self.get_prompt_template()

        time_kst = datetime.now().astimezone(timezone(timedelta(hours=9)))
        time_prompt = HumanMessage(
            content=f"The current time and date is {time_kst.strftime('%c')}",
        )

        messages: list[BaseMessage] = base_prompt + time_prompt

        # user_info가 None인 경우 빈 딕셔너리로 초기화
        if user_info is None:
            user_info = {}

        return messages.format_messages(
            topic=topic,
            base_knowledge=base_knowledge,
            completed_tasks=completed_tasks,
            incomplete_tasks=incomplete_tasks,
            user_info=user_info,
        )
        
class ResearchUnitPrompt(BasePrompt):
    def initialize_prompt(self) -> None:
        self.prompt = ChatPromptTemplate.from_messages(
            [
                HumanMessagePromptTemplate.from_template(RESEARCH_UNIT_PROMPT),
            ]
        )

    def format_messages(
        self,
        messages: list[BaseMessage],
        topic: str,
        task_description: str,
        references: list[dict],
        user_info: dict = None,
        **kwargs: dict[str, Any],
    ) -> list:
        base_prompt = self.get_prompt_template()

        time_kst = datetime.now().astimezone(timezone(timedelta(hours=9)))
        time_prompt = HumanMessage(
            content=f"The current time and date is {time_kst.strftime('%c')}",
        )

        messages: list[BaseMessage] = base_prompt + time_prompt + messages

        references = [
            {k: v for k, v in item.items() if k in ["date", "title", "content"]}
            for item in references
        ]

        # user_info가 None인 경우 빈 딕셔너리로 초기화
        if user_info is None:
            user_info = {}

        return messages.format_messages(
            topic=topic,
            task_description=task_description,
            references=references,
            user_info=user_info,
            **kwargs,
        )

class ResearchUnitSynthesisPrompt(BasePrompt):
    def initialize_prompt(self) -> None:
        self.prompt = ChatPromptTemplate.from_messages(
            [
                HumanMessagePromptTemplate.from_template(
                    RESEARCH_UNIT_SYNTHESIS_PROMPT
                ),
            ]
        )

    def format_messages(
        self,
        topic: str,
        task_description: str,
        messages: list[BaseMessage],
        references: list[dict],
        user_info: dict = None,
    ) -> list:
        base_prompt = self.get_prompt_template()

        time_kst = datetime.now().astimezone(timezone(timedelta(hours=9)))
        time_prompt = HumanMessage(
            content=f"The current time and date is {time_kst.strftime('%c')}",
        )

        # 메시지와 참조 문헌을 문자열로 변환
        messages_str = "\n\n".join(
            [f"Message {i+1}: {msg.content}" for i, msg in enumerate(messages)]
        )
        references_str = "\n\n".join(
            [
                f"Reference {ref.get('number', i+1)}: {ref.get('title', 'No title')} - {ref.get('source', 'No source')}"
                for i, ref in enumerate(references)
            ]
        )

        messages: list[BaseMessage] = base_prompt + time_prompt

        # user_info가 None인 경우 빈 딕셔너리로 초기화
        if user_info is None:
            user_info = {}

        return messages.format_messages(
            topic=topic,
            task_description=task_description,
            messages=messages_str,
            references=references_str,
            user_info=user_info,
        )

class ReportCompilationPrompt(BasePrompt):
    def initialize_prompt(self) -> None:
        self.prompt = ChatPromptTemplate.from_messages(
            [
                HumanMessagePromptTemplate.from_template(REPORT_COMPILATION_PROMPT),
            ]
        )

    def format_messages(
        self,
        topic: str,
        sections: list,
        tasks: list,
        user_info: dict = None,
    ) -> list:
        base_prompt = self.get_prompt_template()

        time_kst = datetime.now().astimezone(timezone(timedelta(hours=9)))
        time_prompt = HumanMessage(
            content=f"The current time and date is {time_kst.strftime('%c')}",
        )

        messages: list[BaseMessage] = base_prompt + time_prompt

        task_results = [task.references for task in tasks]

        # user_info가 None인 경우 빈 딕셔너리로 초기화
        if user_info is None:
            user_info = {}

        return messages.format_messages(
            topic=topic, 
            sections=sections, 
            references=task_results,
            user_info=user_info
        )

class ReferenceFilterPrompt(BasePrompt):
    def initialize_prompt(self) -> None:
        self.prompt = ChatPromptTemplate.from_messages(
            [
                HumanMessagePromptTemplate.from_template(REFERENCE_FILTER_PROMPT),
            ]
        )

    def format_messages(
        self,
        topic: str,
        task_description: str,
        references: list[dict],
        user_info: dict = None,
    ) -> list:
        base_prompt = self.get_prompt_template()

        # 참조 문헌 정보를 문자열로 변환
        references_text = ""
        for ref in references:
            ref_content = ref.get("content", "")
            ref_title = ref.get("title", "Unknown")
            ref_url = ref.get("url", "No URL")
            ref_number = ref.get("number", "Unknown")

            # 내용이 너무 길면 잘라내기
            if len(ref_content) > 300:
                ref_content = ref_content[:300] + "..."

            references_text += f"\nReference {ref_number}:\nTitle: {ref_title}\nURL: {ref_url}\nContent: {ref_content}\n"

        # 사용자 메시지 생성
        human_message = HumanMessage(
            content=f"Research Topic: {topic}\nTask Description: {task_description}\n\nReferences:\n{references_text}"
        )

        # 최종 메시지 목록 생성
        messages = base_prompt + [human_message]

        # user_info가 None인 경우 빈 딕셔너리로 초기화
        if user_info is None:
            user_info = {}

        return messages.format_messages(
            topic=topic, 
            task_description=task_description,
            user_info=user_info
        )


class ReportSectionsGenerationPrompt(BasePrompt):
    def initialize_prompt(self) -> None:
        self.prompt = ChatPromptTemplate.from_messages(
            [
                HumanMessagePromptTemplate.from_template(
                    REPORT_SECTION_GENERATION_PROMPT
                ),
            ]
        )

    def format_messages(
        self,
        topic: str,
        tasks: list[str],
        references: list[dict],
        user_info: dict = None,
    ) -> list:
        base_prompt = self.get_prompt_template()

        # 최종 메시지 목록 생성
        messages = base_prompt
        
        # user_info가 None인 경우 빈 딕셔너리로 초기화
        if user_info is None:
            user_info = {}

        return messages.format_messages(
            topic=topic, 
            tasks=tasks, 
            references=references,
            user_info=user_info
        )

class SummaryPrompt(BasePrompt):
    def initialize_prompt(self) -> None:
        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(SUMMARY_SYSTEM_PROMPT),
                HumanMessagePromptTemplate.from_template(SUMMARY_HUMAN_PROMPT),
            ]
        )

    def format_messages(self, user_query, file_type, **kwargs: dict[str, Any]) -> list:
        # TYPE HINT
        base_prompt = self.get_prompt_template()
        if file_type == "application/pdf":
            media_file = kwargs.get("media_file")
            file_prompt = HumanMessage(
                content=[
                    {
                        "type": "media",
                        "mime_type": file_type,
                        "data": media_file,
                    },
                ]
            )
            messages: list[BaseMessage] = base_prompt + file_prompt
            return messages.format_messages(user_query=user_query, context="")

        return base_prompt.format_messages(
            user_query=user_query, context=kwargs.get("context")
        )
        
        
class ImageEvaluatePrompt(BasePrompt):
    def initialize_prompt(self) -> None:
        self.prompt = ChatPromptTemplate.from_messages(
            [
                HumanMessagePromptTemplate.from_template(IMAGE_EVALUATE_PROMPT),
            ]
        )

    def format_messages(self, image_data: str, user_info: dict = None) -> list:
        base_prompt = self.get_prompt_template()
        
        # user_info에서 성별만 추출
        gender = ""
        if user_info and "gender" in user_info:
            gender = user_info.get("gender", "정보 없음")
        else:
            gender = "정보 없음"

        messages: list[BaseMessage] = base_prompt + [HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                },
            ]
        )]

        return messages.format_messages(gender=gender)