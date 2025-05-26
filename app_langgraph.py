"""
스마트한 쇼핑 앱 - LangGraph 버전
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

# LangGraph 관련 임포트
from typing import TypedDict, Annotated, List, Union, Dict
from langgraph.graph import StateGraph, END
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
    results: Dict
    pros: List[str]
    cons: List[str]
    sources: List[Dict]
    messages: Annotated[List[str], operator.add]  # 프로세스 로그
    error: str
    db_search_complete: bool
    web_search_complete: bool
    save_complete: bool

# ========================
# 헬퍼 클래스들
# ========================

# 클라이언트 초기화
@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=OPENAI_API_KEY)

@st.cache_resource
def get_supabase_client():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

# OpenAI 임베딩 클래스
class OpenAIEmbeddings:
    def __init__(self):
        self.client = get_openai_client()
        self.model = "text-embedding-ada-002"
    
    def get_embedding(self, text: str):
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
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

# 크롤러 클래스
class NaverBlogCrawler:
    def __init__(self):
        self.headers = {
            "X-Naver-Client-Id": NAVER_CLIENT_ID,
            "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
        }
    
    def search_blog(self, query, display=10):
        url = "https://openapi.naver.com/v1/search/blog"
        params = {"query": query, "display": display, "sort": "sim"}
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            st.error(f"블로그 검색 오류: {e}")
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
                        
                        for selector in ['div.se-main-container', 'div#postViewArea', 'div.post_ct']:
                            elem = soup.select_one(selector)
                            if elem:
                                content = elem.get_text(separator='\n', strip=True)
                                content = re.sub(r'\s+', ' ', content)
                                return content if len(content) > 300 else None
        except:
            pass
        return None

# ========================
# LangGraph 노드 함수들
# ========================

def search_database_node(state: SearchState) -> SearchState:
    """데이터베이스 검색 노드"""
    product_name = state["product_name"]
    state["messages"].append(f"📊 데이터베이스에서 '{product_name}' 검색 중...")
    
    supabase = get_supabase_client()
    embeddings_helper = OpenAIEmbeddings()
    
    try:
        # 1. 정확한 매칭
        result = supabase.table('laptop_pros_cons').select("*").eq('product_name', product_name).execute()
        if result.data:
            state["search_method"] = "database"
            state["results"] = {"data": result.data}
            state["messages"].append(f"✅ 데이터베이스에서 정확히 일치하는 제품 발견!")
            state["db_search_complete"] = True
            return state
        
        # 2. 부분 매칭
        result = supabase.table('laptop_pros_cons').select("*").ilike('product_name', f'%{product_name}%').execute()
        if result.data:
            state["search_method"] = "database"
            state["results"] = {"data": result.data}
            state["messages"].append(f"📌 데이터베이스에서 유사한 제품 발견!")
            state["db_search_complete"] = True
            return state
        
        # 3. 유사도 검색
        state["messages"].append(f"🤖 AI 유사도 검색 시도 중...")
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
                        
                        if similarity >= 0.7:
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
                result = supabase.table('laptop_pros_cons').select("*").eq('product_name', best_match).execute()
                if result.data:
                    state["search_method"] = "similarity"
                    state["results"] = {"data": result.data}
                    state["messages"].append(f"🎯 AI가 유사한 제품 '{best_match}'를 찾았습니다!")
                    state["db_search_complete"] = True
                    return state
        
        state["messages"].append(f"❌ 데이터베이스에서 제품을 찾을 수 없음")
        state["results"] = {"data": None}
        state["db_search_complete"] = True
        
    except Exception as e:
        state["error"] = str(e)
        state["messages"].append(f"⚠️ 데이터베이스 검색 오류: {str(e)}")
        state["db_search_complete"] = True
    
    return state

def crawl_web_node(state: SearchState) -> SearchState:
    """웹 크롤링 노드"""
    if state["results"].get("data"):  # 이미 DB에서 찾은 경우
        return state
    
    product_name = state["product_name"]
    state["search_method"] = "web_crawling"
    state["messages"].append(f"🌐 웹에서 '{product_name}' 리뷰 수집 시작...")
    
    crawler = NaverBlogCrawler()
    openai_client = get_openai_client()
    
    all_pros = []
    all_cons = []
    sources = []
    
    search_queries = [
        f"{product_name} 장단점 실사용",
        f"{product_name} 후기"
    ]
    
    for query in search_queries[:2]:
        state["messages"].append(f"🔍 검색어: '{query}'")
        blog_result = crawler.search_blog(query, display=5)
        
        if not blog_result or 'items' not in blog_result:
            continue
        
        for post in blog_result['items'][:3]:
            title = BeautifulSoup(post['title'], "html.parser").get_text()
            state["messages"].append(f"📖 분석 중: {title[:30]}...")
            
            content = crawler.crawl_content(post['link'])
            
            if content:
                # GPT로 장단점 추출
                prompt = f"""다음은 "{product_name}"에 대한 블로그 리뷰입니다.

[블로그 내용]
{content[:1500]}

위 내용에서 {product_name}의 장점과 단점을 추출해주세요.

장점:
- (구체적인 장점)

단점:
- (구체적인 단점)

정보가 부족하면 "정보 부족"이라고 답하세요."""
                
                try:
                    response = openai_client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "노트북 리뷰 분석 전문가입니다."},
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
                            all_pros.extend(pros)
                            all_cons.extend(cons)
                            sources.append({
                                'title': title,
                                'link': post['link']
                            })
                            state["messages"].append(f"✓ 장점 {len(pros)}개, 단점 {len(cons)}개 추출")
                except:
                    pass
            
            time.sleep(0.5)
    
    # 중복 제거
    state["pros"] = list(dict.fromkeys(all_pros))[:10]
    state["cons"] = list(dict.fromkeys(all_cons))[:10]
    state["sources"] = sources[:5]
    
    state["messages"].append(f"🎉 웹 크롤링 완료! 총 장점 {len(state['pros'])}개, 단점 {len(state['cons'])}개 수집")
    state["web_search_complete"] = True
    
    return state

def process_results_node(state: SearchState) -> SearchState:
    """결과 처리 노드"""
    if state["search_method"] in ["database", "similarity"] and state["results"].get("data"):
        # DB 결과 처리
        data = state["results"]["data"]
        state["pros"] = [item['content'] for item in data if item['type'] == 'pro']
        state["cons"] = [item['content'] for item in data if item['type'] == 'con']
        state["sources"] = []
        
        state["messages"].append(f"📋 결과 정리: 장점 {len(state['pros'])}개, 단점 {len(state['cons'])}개")
    
    return state

def save_to_database_node(state: SearchState) -> SearchState:
    """데이터베이스 저장 노드"""
    if state["search_method"] != "web_crawling" or not (state["pros"] or state["cons"]):
        return state
    
    state["messages"].append(f"💾 데이터베이스에 저장 중...")
    
    supabase = get_supabase_client()
    embeddings_helper = OpenAIEmbeddings()
    
    try:
        # 임베딩 생성 (선택적)
        embedding = None
        # embedding = embeddings_helper.get_embedding(state["product_name"])
        
        data = []
        for pro in state["pros"]:
            data.append({
                'product_name': state["product_name"],
                'type': 'pro',
                'content': pro,
                # 'embedding': embedding
            })
        
        for con in state["cons"]:
            data.append({
                'product_name': state["product_name"],
                'type': 'con',
                'content': con,
                # 'embedding': embedding
            })
        
        if data:
            supabase.table('laptop_pros_cons').insert(data).execute()
            state["messages"].append(f"✅ 데이터베이스 저장 완료!")
            state["save_complete"] = True
    except Exception as e:
        state["messages"].append(f"⚠️ 저장 오류: {str(e)}")
    
    return state

# ========================
# LangGraph 워크플로우
# ========================

def should_search_web(state: SearchState) -> str:
    """웹 검색 필요 여부 결정"""
    if state["results"].get("data"):
        return "process"
    else:
        return "crawl"

def should_save_to_db(state: SearchState) -> str:
    """DB 저장 필요 여부 결정"""
    if state["search_method"] == "web_crawling" and (state["pros"] or state["cons"]):
        return "save"
    else:
        return "end"

@st.cache_resource
def create_search_workflow():
    """검색 워크플로우 생성"""
    workflow = StateGraph(SearchState)
    
    # 노드 추가
    workflow.add_node("search_db", search_database_node)
    workflow.add_node("crawl_web", crawl_web_node)
    workflow.add_node("process", process_results_node)
    workflow.add_node("save_db", save_to_database_node)
    
    # 엣지 설정
    workflow.set_entry_point("search_db")
    
    # DB 검색 후 분기
    workflow.add_conditional_edges(
        "search_db",
        should_search_web,
        {
            "crawl": "crawl_web",
            "process": "process"
        }
    )
    
    # 웹 크롤링 후 처리
    workflow.add_edge("crawl_web", "process")
    
    # 결과 처리 후 분기
    workflow.add_conditional_edges(
        "process",
        should_save_to_db,
        {
            "save": "save_db",
            "end": END
        }
    )
    
    # DB 저장 후 종료
    workflow.add_edge("save_db", END)
    
    return workflow.compile()

# ========================
# Streamlit UI
# ========================

# 워크플로우 초기화
search_app = create_search_workflow()

# 검색 UI
col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    product_name = st.text_input(
        "🔍 제품명을 입력하세요",
        placeholder="예: 맥북 프로 M3, LG 그램 2024, 갤럭시북4 프로"
    )
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        search_button = st.button("🔍 검색하기", use_container_width=True, type="primary")
    with col_btn2:
        show_process = st.checkbox("🔧 프로세스 보기", value=False)

# 검색 실행
if search_button and product_name:
    # 초기 상태
    initial_state = {
        "product_name": product_name,
        "search_method": "",
        "results": {},
        "pros": [],
        "cons": [],
        "sources": [],
        "messages": [],
        "error": "",
        "db_search_complete": False,
        "web_search_complete": False,
        "save_complete": False
    }
    
    # LangGraph 실행
    with st.spinner(f"'{product_name}' 검색 중..."):
        final_state = search_app.invoke(initial_state)
    
    # 프로세스 로그 표시
    if show_process and final_state["messages"]:
        with st.expander("🔧 검색 프로세스", expanded=True):
            for msg in final_state["messages"]:
                st.write(msg)
    
    # 결과 표시
    if final_state["pros"] or final_state["cons"]:
        # 검색 정보
        st.markdown(f"""
        <div class="process-info">
            <strong>검색 방법:</strong> {
                '데이터베이스 (정확히 일치)' if final_state["search_method"] == "database" else
                'AI 유사도 검색' if final_state["search_method"] == "similarity" else
                '웹 크롤링'
            } | 
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
            
            for idx, pro in enumerate(final_state["pros"], 1):
                st.write(f"{idx}. {pro}")
        
        with col2:
            st.markdown("""
            <div class="cons-section">
                <h3>❌ 단점</h3>
            </div>
            """, unsafe_allow_html=True)
            
            for idx, con in enumerate(final_state["cons"], 1):
                st.write(f"{idx}. {con}")
        
        # 출처 (웹 크롤링인 경우)
        if final_state["sources"]:
            with st.expander("📚 출처 보기"):
                for idx, source in enumerate(final_state["sources"], 1):
                    st.write(f"{idx}. [{source['title']}]({source['link']})")
        
        # 통계
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("총 장점", f"{len(final_state['pros'])}개")
        with col2:
            st.metric("총 단점", f"{len(final_state['cons'])}개")
        with col3:
            st.metric("검색 방법", 
                     "DB" if final_state["search_method"] == "database" else 
                     "AI" if final_state["search_method"] == "similarity" else 
                     "웹")
        with col4:
            st.metric("프로세스 단계", 
                     f"{len(final_state['messages'])}개")
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
    st.info("📊 자동 데이터베이스 저장 및 관리")

current_date = datetime.now().strftime('%Y년 %m월 %d일')
st.markdown(f"""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>마지막 업데이트: {current_date}</p>
    <p>Powered by LangGraph & OpenAI</p>
</div>
""", unsafe_allow_html=True)
