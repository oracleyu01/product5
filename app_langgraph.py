"""
스마트한 쇼핑 앱 - LangGraph 버전 (벡터 검색 포함)
"""

import streamlit as st
import pandas as pd
from supabase import create_client
from openai import OpenAI
import os
from dotenv import load_dotenv
from datetime import datetime
import time
import json
import re
import requests
from bs4 import BeautifulSoup
import numpy as np

# LangGraph 관련
from typing import TypedDict, Annotated, List, Union, Dict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
import operator

# 페이지 설정
st.set_page_config(
    page_title="스마트한 쇼핑 (LangGraph)",
    page_icon="🛒",
    layout="wide"
)

# 환경 변수 로드
load_dotenv()

# API 키 설정
SUPABASE_URL = os.getenv("SUPABASE_URL") or st.secrets.get("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY") or st.secrets.get("SUPABASE_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID") or st.secrets.get("NAVER_CLIENT_ID", "")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET") or st.secrets.get("NAVER_CLIENT_SECRET", "")

# LangSmith 설정 (선택적)
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY") or st.secrets.get("LANGSMITH_API_KEY", "")
if LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "smart-shopping-app"
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

# CSS 스타일
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .pros-section {
        background-color: #d4edda;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 5px solid #28a745;
    }
    .cons-section {
        background-color: #f8d7da;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 5px solid #dc3545;
    }
    .process-info {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        border-left: 3px solid #2196f3;
    }
</style>
""", unsafe_allow_html=True)

# 헤더
st.markdown("""
<div class="main-header">
    <h1>🛒 스마트한 쇼핑 (LangGraph Edition)</h1>
    <p style="font-size: 1.2rem; margin-top: 1rem;">
        LangGraph로 구현한 지능형 제품 리뷰 분석 시스템
    </p>
</div>
""", unsafe_allow_html=True)

# ========================
# LangGraph State 정의
# ========================

class SearchState(TypedDict):
    """검색 프로세스의 상태"""
    product_name: str
    search_method: str  # "database", "web_crawling", "similarity"
    results: dict
    pros: List[str]
    cons: List[str]
    sources: List[dict]
    messages: Annotated[List[Union[HumanMessage, AIMessage]], operator.add]
    error: str
    similar_product: str  # 유사도 검색으로 찾은 제품명

# ========================
# OpenAI 임베딩 클래스
# ========================

class OpenAIEmbeddings:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = "text-embedding-ada-002"
    
    def get_embedding(self, text: str):
        """텍스트의 임베딩 벡터 생성"""
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            return response.data[0].embedding
        except Exception as e:
            st.error(f"임베딩 생성 오류: {e}")
            return None
    
    def cosine_similarity(self, vec1, vec2):
        """코사인 유사도 계산"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

# ========================
# 크롤링 클래스
# ========================

class ProConsLaptopCrawler:
    def __init__(self, naver_client_id, naver_client_secret):
        self.naver_headers = {
            "X-Naver-Client-Id": naver_client_id,
            "X-Naver-Client-Secret": naver_client_secret
        }
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
    
    def remove_html_tags(self, text):
        text = BeautifulSoup(text, "html.parser").get_text()
        text = re.sub(r'<[^>]+>', '', text)
        return text.strip()
    
    def search_blog(self, query, display=10):
        url = "https://openapi.naver.com/v1/search/blog"
        params = {
            "query": query,
            "display": display,
            "sort": "sim"
        }
        
        try:
            response = requests.get(url, headers=self.naver_headers, params=params)
            if response.status_code == 200:
                result = response.json()
                for item in result.get('items', []):
                    item['title'] = self.remove_html_tags(item['title'])
                    item['description'] = self.remove_html_tags(item['description'])
                return result
        except Exception as e:
            return None
        return None
    
    def crawl_content(self, url):
        try:
            if "blog.naver.com" in url:
                parts = url.split('/')
                if len(parts) >= 5:
                    blog_id = parts[3]
                    post_no = parts[4].split('?')[0]
                    mobile_url = f"https://m.blog.naver.com/{blog_id}/{post_no}"
                    
                    response = requests.get(mobile_url, headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    })
                    
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        
                        content = ""
                        for selector in ['div.se-main-container', 'div#postViewArea', 'div.post_ct']:
                            elem = soup.select_one(selector)
                            if elem:
                                content = elem.get_text(separator='\n', strip=True)
                                break
                        
                        if not content:
                            content = soup.get_text(separator='\n', strip=True)
                        
                        content = re.sub(r'\s+', ' ', content)
                        content = content.replace('\u200b', '')
                        
                        return content if len(content) > 300 else None
        except:
            pass
        return None
    
    def extract_pros_cons_with_gpt(self, product_name, content):
        if not content or len(content) < 200:
            return None
        
        content_preview = content[:1500]
        
        prompt = f"""다음은 "{product_name}"에 대한 블로그 리뷰입니다.

[블로그 내용]
{content_preview}

위 내용에서 {product_name}의 장점과 단점을 추출해주세요.

장점:
- (구체적인 장점)

단점:
- (구체적인 단점)

정보가 부족하면 "정보 부족"이라고 답하세요."""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "제품 리뷰 분석 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            result = response.choices[0].message.content.strip()
            
            if result and "정보 부족" not in result:
                pros = []
                cons = []
                
                lines = result.split('\n')
                current_section = None
                
                for line in lines:
                    line = line.strip()
                    if '장점:' in line:
                        current_section = 'pros'
                    elif '단점:' in line:
                        current_section = 'cons'
                    elif line.startswith('-') and current_section:
                        point = line[1:].strip()
                        if point and len(point) > 5:
                            if current_section == 'pros':
                                pros.append(point)
                            else:
                                cons.append(point)
                
                if pros or cons:
                    return {'pros': pros[:5], 'cons': cons[:5]}
            
            return None
        except:
            return None

# ========================
# LangGraph 노드 함수들
# ========================

# 클라이언트 초기화
@st.cache_resource
def get_supabase_client():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

@st.cache_resource
def get_crawler():
    return ProConsLaptopCrawler(NAVER_CLIENT_ID, NAVER_CLIENT_SECRET)

@st.cache_resource
def get_embeddings_helper():
    return OpenAIEmbeddings()

def search_database(state: SearchState) -> SearchState:
    """데이터베이스에서 제품 검색 (벡터 검색 포함)"""
    product_name = state["product_name"]
    supabase = get_supabase_client()
    embeddings_helper = get_embeddings_helper()
    
    state["messages"].append(
        HumanMessage(content=f"📊 데이터베이스에서 '{product_name}' 검색 중...")
    )
    
    try:
        # 1. 정확한 매칭
        exact_match = supabase.table('laptop_pros_cons').select("*").eq('product_name', product_name).execute()
        if exact_match.data:
            state["search_method"] = "database"
            state["results"] = {"data": exact_match.data}
            state["messages"].append(
                AIMessage(content=f"✅ 데이터베이스에서 정확히 일치하는 제품 발견! ({len(exact_match.data)}개 항목)")
            )
            return state
        
        # 2. 부분 매칭
        partial_match = supabase.table('laptop_pros_cons').select("*").ilike('product_name', f'%{product_name}%').execute()
        if partial_match.data:
            state["search_method"] = "database"
            state["results"] = {"data": partial_match.data}
            state["messages"].append(
                AIMessage(content=f"📌 데이터베이스에서 유사한 제품 발견! ({len(partial_match.data)}개 항목)")
            )
            return state
        
        # 3. 벡터 유사도 검색
        state["messages"].append(
            AIMessage(content=f"🤖 AI 유사도 검색 시도 중...")
        )
        
        query_embedding = embeddings_helper.get_embedding(product_name)
        if query_embedding:
            all_products = supabase.table('laptop_pros_cons').select("*").execute()
            
            similar_products = []
            checked_products = set()
            
            for item in all_products.data:
                if item['product_name'] in checked_products:
                    continue
                
                if item.get('embedding'):
                    try:
                        item_embedding = json.loads(item['embedding']) if isinstance(item['embedding'], str) else item['embedding']
                        similarity = embeddings_helper.cosine_similarity(query_embedding, item_embedding)
                        
                        if similarity >= 0.8:  # 임계값을 0.8로 높임
                            similar_products.append({
                                'product_name': item['product_name'],
                                'similarity': similarity
                            })
                            checked_products.add(item['product_name'])
                    except:
                        pass
            
            if similar_products:
                similar_products.sort(key=lambda x: x['similarity'], reverse=True)
                best_match = similar_products[0]['product_name']
                best_similarity = similar_products[0]['similarity']
                
                # 실제로 유사한 제품인지 확인 (유사도가 0.8 이상이면서 검색어가 포함된 경우)
                if best_similarity >= 0.8:
                    result = supabase.table('laptop_pros_cons').select("*").eq('product_name', best_match).execute()
                    if result.data:
                        state["search_method"] = "similarity"
                        state["results"] = {"data": result.data}
                        state["similar_product"] = best_match
                        state["messages"].append(
                            AIMessage(content=f"🎯 AI가 유사한 제품 '{best_match}'를 찾았습니다! (유사도: {best_similarity:.2f})")
                        )
                        return state
        
        state["messages"].append(
            AIMessage(content="❌ 데이터베이스에서 제품을 찾을 수 없음. 웹 검색을 시작합니다...")
        )
        state["results"] = {"data": None}
        return state
        
    except Exception as e:
        state["error"] = str(e)
        state["messages"].append(
            AIMessage(content=f"⚠️ 데이터베이스 검색 오류: {str(e)}")
        )
        state["results"] = {"data": None}
        return state

def crawl_web(state: SearchState) -> SearchState:
    """웹에서 제품 정보 크롤링"""
    if state["results"].get("data"):  # 이미 DB에서 찾은 경우
        return state
    
    product_name = state["product_name"]
    state["search_method"] = "web_crawling"
    crawler = get_crawler()
    
    state["messages"].append(
        HumanMessage(content=f"🌐 웹에서 '{product_name}' 리뷰 수집 시작...")
    )
    
    all_pros = []
    all_cons = []
    sources = []
    
    # 검색 쿼리
    search_queries = [
        f"{product_name} 장단점 실사용",
        f"{product_name} 후기"
    ]
    
    for query in search_queries[:2]:
        state["messages"].append(
            AIMessage(content=f"🔍 검색어: '{query}'로 네이버 블로그 검색 중...")
        )
        
        result = crawler.search_blog(query, display=5)
        if not result or 'items' not in result:
            continue
        
        posts = result['items']
        
        for idx, post in enumerate(posts[:3]):
            state["messages"].append(
                AIMessage(content=f"📖 블로그 분석 중: {post['title'][:30]}...")
            )
            
            content = crawler.crawl_content(post['link'])
            if not content:
                continue
            
            pros_cons = crawler.extract_pros_cons_with_gpt(product_name, content)
            
            if pros_cons:
                all_pros.extend(pros_cons['pros'])
                all_cons.extend(pros_cons['cons'])
                sources.append({
                    'title': post['title'],
                    'link': post['link']
                })
                
                state["messages"].append(
                    AIMessage(content=f"✓ 장점 {len(pros_cons['pros'])}개, 단점 {len(pros_cons['cons'])}개 추출 완료")
                )
            
            time.sleep(0.5)
    
    # 중복 제거
    state["pros"] = list(dict.fromkeys(all_pros))[:10]
    state["cons"] = list(dict.fromkeys(all_cons))[:10]
    state["sources"] = sources[:5]
    
    if state["pros"] or state["cons"]:
        state["messages"].append(
            AIMessage(content=f"🎉 웹 크롤링 완료! 총 장점 {len(state['pros'])}개, 단점 {len(state['cons'])}개 수집")
        )
        
        # DB에 저장 (임베딩 포함)
        try:
            supabase = get_supabase_client()
            embeddings_helper = get_embeddings_helper()
            
            # 제품명의 임베딩 생성
            product_embedding = embeddings_helper.get_embedding(product_name)
            
            data = []
            for pro in state["pros"]:
                data.append({
                    'product_name': product_name,
                    'type': 'pro',
                    'content': pro,
                    'embedding': product_embedding  # 임베딩 추가
                })
            
            for con in state["cons"]:
                data.append({
                    'product_name': product_name,
                    'type': 'con',
                    'content': con,
                    'embedding': product_embedding  # 임베딩 추가
                })
            
            if data:
                supabase.table('laptop_pros_cons').insert(data).execute()
                state["messages"].append(
                    AIMessage(content="💾 데이터베이스에 저장 완료! (임베딩 포함)")
                )
        except Exception as e:
            state["messages"].append(
                AIMessage(content=f"⚠️ DB 저장 실패: {str(e)}")
            )
    else:
        state["messages"].append(
            AIMessage(content="😢 웹에서도 정보를 찾을 수 없습니다.")
        )
    
    return state

def process_results(state: SearchState) -> SearchState:
    """결과 처리 및 정리"""
    if state["search_method"] in ["database", "similarity"] and state["results"].get("data"):
        # DB 결과 처리
        data = state["results"]["data"]
        state["pros"] = [item['content'] for item in data if item['type'] == 'pro']
        state["cons"] = [item['content'] for item in data if item['type'] == 'con']
        state["sources"] = []  # DB에는 별도 소스 없음
        
        state["messages"].append(
            AIMessage(content=f"📋 결과 정리 완료: 장점 {len(state['pros'])}개, 단점 {len(state['cons'])}개")
        )
    
    return state

def should_search_web(state: SearchState) -> str:
    """웹 검색이 필요한지 판단"""
    if state["results"].get("data"):
        return "process"
    else:
        return "crawl"

# ========================
# LangGraph 워크플로우 생성
# ========================

@st.cache_resource
def create_search_workflow():
    workflow = StateGraph(SearchState)
    
    # 노드 추가
    workflow.add_node("search_db", search_database)
    workflow.add_node("crawl_web", crawl_web)
    workflow.add_node("process", process_results)
    
    # 엣지 설정
    workflow.set_entry_point("search_db")
    workflow.add_conditional_edges(
        "search_db",
        should_search_web,
        {
            "crawl": "crawl_web",
            "process": "process"
        }
    )
    workflow.add_edge("crawl_web", "process")
    workflow.add_edge("process", END)
    
    return workflow.compile()

# 워크플로우 인스턴스 생성
search_app = create_search_workflow()

# ========================
# Streamlit UI
# ========================

# 검색 섹션
col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    product_name = st.text_input(
        "🔍 제품명을 입력하세요",
        placeholder="예: 맥북 프로 M3, LG 그램 2024, 갤럭시북4 프로, 게이트맨 도어락"
    )
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        search_button = st.button("🔍 검색하기", use_container_width=True, type="primary")
    with col_btn2:
        show_process = st.checkbox("🔧 프로세스 보기", value=True)

# 검색 실행
if search_button and product_name:
    with st.spinner(f"'{product_name}' 검색 중..."):
        # LangGraph 실행
        initial_state = {
            "product_name": product_name,
            "search_method": "",
            "results": {},
            "pros": [],
            "cons": [],
            "sources": [],
            "messages": [],
            "error": "",
            "similar_product": ""
        }
        
        # 워크플로우 실행
        final_state = search_app.invoke(initial_state)
    
    # 프로세스 로그 표시
    if show_process and final_state["messages"]:
        with st.expander("🔧 검색 프로세스", expanded=True):
            for msg in final_state["messages"]:
                if isinstance(msg, HumanMessage):
                    st.write(f"👤 {msg.content}")
                else:
                    st.write(f"🤖 {msg.content}")
    
    # 결과 표시
    if final_state["pros"] or final_state["cons"]:
        # 검색 정보
        method_display = {
            "database": "데이터베이스 (정확히 일치)",
            "similarity": f"AI 유사도 검색 ('{final_state.get('similar_product', '')}')",
            "web_crawling": "웹 크롤링"
        }
        
        st.markdown(f"""
        <div class="process-info">
            <strong>검색 방법:</strong> {method_display.get(final_state["search_method"], "알 수 없음")} | 
            <strong>장점:</strong> {len(final_state["pros"])}개 | 
            <strong>단점:</strong> {len(final_state["cons"])}개
        </div>
        """, unsafe_allow_html=True)
        
        # 장단점 표시
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="pros-section">
                <h3>✅ 장점</h3>
            </div>
            """, unsafe_allow_html=True)
            
            for idx, pro in enumerate(final_state["pros"][:10], 1):
                st.write(f"{idx}. {pro}")
        
        with col2:
            st.markdown("""
            <div class="cons-section">
                <h3>❌ 단점</h3>
            </div>
            """, unsafe_allow_html=True)
            
            for idx, con in enumerate(final_state["cons"][:10], 1):
                st.write(f"{idx}. {con}")
        
        # 출처 (웹 크롤링인 경우)
        if final_state["sources"]:
            with st.expander("📚 출처 보기"):
                for idx, source in enumerate(final_state["sources"], 1):
                    st.write(f"{idx}. [{source['title']}]({source['link']})")
        
        # 통계
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("총 장점", f"{len(final_state['pros'])}개")
        with col2:
            st.metric("총 단점", f"{len(final_state['cons'])}개")
        with col3:
            st.metric("검색 방법", 
                     "DB" if final_state["search_method"] == "database" else 
                     "AI" if final_state["search_method"] == "similarity" else 
                     "웹")
    else:
        st.error(f"'{product_name}'에 대한 정보를 찾을 수 없습니다.")

# 하단 정보
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.info("💡 LangGraph로 구현된 체계적인 검색 프로세스")
with col2:
    st.info("🤖 OpenAI 임베딩을 사용한 지능형 검색")
with col3:
    st.info("💾 검색 결과 자동 저장 (임베딩 포함)")

current_date = datetime.now().strftime('%Y년 %m월 %d일')
st.markdown(f"""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>마지막 업데이트: {current_date}</p>
    <p>Powered by LangGraph & OpenAI</p>
</div>
""", unsafe_allow_html=True)
