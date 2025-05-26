"""
스마트한 쇼핑 앱 - 전체 기능 (LangGraph 없음)
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

# 페이지 설정
st.set_page_config(
    page_title="스마트한 쇼핑",
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
    .search-info {
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
    <h1>🛒 스마트한 쇼핑</h1>
    <p style="font-size: 1.2rem; margin-top: 1rem;">
        블로그에서 수집한 실사용 후기를 바탕으로 제품의 장점과 단점을 한눈에 확인하세요
    </p>
</div>
""", unsafe_allow_html=True)

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

# 네이버 블로그 검색
def search_naver_blog(query, display=10):
    url = "https://openapi.naver.com/v1/search/blog"
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
    }
    params = {"query": query, "display": display, "sort": "sim"}
    
    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"블로그 검색 오류: {e}")
    return None

# HTML 태그 제거
def remove_html_tags(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'<[^>]+>', '', text)
    return text.strip()

# 블로그 내용 크롤링
def crawl_blog_content(url):
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
    except Exception as e:
        st.error(f"크롤링 오류: {e}")
    return None

# GPT로 장단점 추출
def extract_pros_cons(product_name, content):
    if not content or len(content) < 200:
        return None
    
    client = get_openai_client()
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
        response = client.chat.completions.create(
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
                return {'pros': pros[:5], 'cons': cons[:5]}
    except Exception as e:
        st.error(f"GPT 분석 오류: {e}")
    return None

# 데이터베이스 검색 (유사도 검색 포함)
def search_database(product_name):
    supabase = get_supabase_client()
    embeddings_helper = OpenAIEmbeddings()
    
    try:
        # 1. 정확한 매칭
        result = supabase.table('laptop_pros_cons').select("*").eq('product_name', product_name).execute()
        if result.data:
            return process_db_results(result.data), "exact", None
        
        # 2. 부분 매칭
        result = supabase.table('laptop_pros_cons').select("*").ilike('product_name', f'%{product_name}%').execute()
        if result.data:
            return process_db_results(result.data), "partial", None
        
        # 3. 유사도 검색 (임베딩이 있는 경우)
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
                    return process_db_results(result.data), "similarity", best_match
        
    except Exception as e:
        st.error(f"데이터베이스 검색 오류: {e}")
    
    return None, None, None

def process_db_results(data):
    pros = [item['content'] for item in data if item['type'] == 'pro']
    cons = [item['content'] for item in data if item['type'] == 'con']
    return {'pros': pros, 'cons': cons}

# 웹 크롤링 및 분석
def crawl_and_analyze(product_name):
    all_pros = []
    all_cons = []
    sources = []
    
    search_queries = [
        f"{product_name} 장단점 실사용",
        f"{product_name} 후기"
    ]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_posts = 6  # 각 쿼리당 3개씩
    current_post = 0
    
    for query in search_queries[:2]:
        status_text.text(f"'{query}' 검색 중...")
        blog_result = search_naver_blog(query, display=5)
        
        if not blog_result or 'items' not in blog_result:
            continue
        
        for post in blog_result['items'][:3]:
            current_post += 1
            progress_bar.progress(current_post / total_posts)
            
            post['title'] = remove_html_tags(post['title'])
            status_text.text(f"분석 중: {post['title'][:30]}...")
            
            content = crawl_blog_content(post['link'])
            
            if content:
                pros_cons = extract_pros_cons(product_name, content)
                if pros_cons:
                    all_pros.extend(pros_cons['pros'])
                    all_cons.extend(pros_cons['cons'])
                    sources.append({
                        'title': post['title'],
                        'link': post['link']
                    })
            
            time.sleep(0.5)
    
    progress_bar.empty()
    status_text.empty()
    
    # 중복 제거
    pros = list(dict.fromkeys(all_pros))[:10]
    cons = list(dict.fromkeys(all_cons))[:10]
    
    return {'pros': pros, 'cons': cons}, sources[:5]

# 데이터베이스에 저장
def save_to_database(product_name, pros, cons):
    supabase = get_supabase_client()
    embeddings_helper = OpenAIEmbeddings()
    
    try:
        # 임베딩 생성
        embedding = embeddings_helper.get_embedding(product_name)
        
        data = []
        for pro in pros:
            data.append({
                'product_name': product_name,
                'type': 'pro',
                'content': pro,
                'embedding': embedding
            })
        
        for con in cons:
            data.append({
                'product_name': product_name,
                'type': 'con',
                'content': con,
                'embedding': embedding
            })
        
        if data:
            supabase.table('laptop_pros_cons').insert(data).execute()
            st.success("✅ 데이터베이스에 저장 완료!")
    except Exception as e:
        st.error(f"저장 오류: {e}")

# 메인 검색 UI
col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    product_name = st.text_input(
        "🔍 제품명을 입력하세요",
        placeholder="예: 맥북 프로 M3, LG 그램 2024, 갤럭시북4 프로"
    )
    
    search_button = st.button("검색하기", use_container_width=True)

# 검색 실행
if search_button and product_name:
    with st.spinner(f"'{product_name}' 검색 중..."):
        # 1. 데이터베이스 검색
        db_results, match_type, similar_product = search_database(product_name)
        
        if db_results:
            # DB에서 찾은 경우
            if match_type == "exact":
                st.success(f"✅ 데이터베이스에서 '{product_name}' 정보를 찾았습니다!")
            elif match_type == "partial":
                st.info(f"📌 데이터베이스에서 유사한 제품 정보를 찾았습니다!")
            else:  # similarity
                st.info(f"🤖 AI 유사도 검색으로 '{similar_product}' 정보를 찾았습니다!")
            
            results = db_results
            sources = []
            search_method = "database"
        else:
            # 웹 크롤링
            st.warning(f"🔄 웹에서 '{product_name}' 정보를 수집 중...")
            results, sources = crawl_and_analyze(product_name)
            search_method = "web_crawling"
            
            # 결과가 있으면 DB에 저장
            if results['pros'] or results['cons']:
                save_to_database(product_name, results['pros'], results['cons'])
    
    # 결과 표시
    if results and (results['pros'] or results['cons']):
        # 검색 정보 표시
        st.markdown(f"""
        <div class="search-info">
            <strong>검색 방법:</strong> {'데이터베이스' if search_method == 'database' else '웹 크롤링'} | 
            <strong>장점:</strong> {len(results['pros'])}개 | 
            <strong>단점:</strong> {len(results['cons'])}개
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
            
            for idx, pro in enumerate(results['pros'], 1):
                st.write(f"{idx}. {pro}")
        
        with col2:
            st.markdown("""
            <div class="cons-section">
                <h3>❌ 단점</h3>
            </div>
            """, unsafe_allow_html=True)
            
            for idx, con in enumerate(results['cons'], 1):
                st.write(f"{idx}. {con}")
        
        # 출처 (웹 크롤링인 경우)
        if sources:
            with st.expander("📚 출처 보기"):
                for idx, source in enumerate(sources, 1):
                    st.write(f"{idx}. [{source['title']}]({source['link']})")
        
        # 통계
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("총 장점", f"{len(results['pros'])}개")
        with col2:
            st.metric("총 단점", f"{len(results['cons'])}개")
        with col3:
            st.metric("검색 방법", "DB" if search_method == "database" else "웹")
    else:
        st.error(f"'{product_name}'에 대한 정보를 찾을 수 없습니다.")

# 하단 정보
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.info("💡 한 번 검색된 제품은 데이터베이스에 저장되어 더 빠르게 검색됩니다.")
with col2:
    st.info("🤖 OpenAI 임베딩을 사용하여 유사한 제품도 찾아드립니다.")

current_date = datetime.now().strftime('%Y년 %m월 %d일')
st.markdown(f"""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>마지막 업데이트: {current_date}</p>
    <p>Powered by OpenAI & Naver API</p>
</div>
""", unsafe_allow_html=True)
