import streamlit as st
import os

st.set_page_config(
    page_title="테스트 앱",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 스마트 쇼핑 앱 - 테스트 버전")

# 환경 변수 체크
st.header("1. 환경 변수 확인")
env_vars = ["SUPABASE_URL", "SUPABASE_KEY", "OPENAI_API_KEY", "NAVER_CLIENT_ID", "NAVER_CLIENT_SECRET"]

for var in env_vars:
    value = os.getenv(var) or st.secrets.get(var, "")
    if value:
        st.success(f"✅ {var}: 설정됨")
    else:
        st.error(f"❌ {var}: 설정 안됨")

# 패키지 임포트 테스트
st.header("2. 패키지 임포트 테스트")

try:
    import pandas as pd
    st.success("✅ pandas 임포트 성공")
except:
    st.error("❌ pandas 임포트 실패")

try:
    from openai import OpenAI
    st.success("✅ OpenAI 임포트 성공")
except:
    st.error("❌ OpenAI 임포트 실패")

try:
    from supabase import create_client
    st.success("✅ Supabase 임포트 성공")
except:
    st.error("❌ Supabase 임포트 실패")

try:
    import requests
    st.success("✅ requests 임포트 성공")
except:
    st.error("❌ requests 임포트 실패")

try:
    from bs4 import BeautifulSoup
    st.success("✅ BeautifulSoup 임포트 성공")
except:
    st.error("❌ BeautifulSoup 임포트 실패")

# 간단한 검색 UI
st.header("3. 기본 UI 테스트")

product_name = st.text_input("제품명을 입력하세요", placeholder="예: 맥북 프로 M3")

if st.button("검색"):
    with st.spinner("검색 중..."):
        st.balloons()
        st.success(f"'{product_name}' 검색 완료! (테스트)")

st.info("이 앱은 테스트 버전입니다. 모든 기능이 정상 작동하면 전체 기능을 추가합니다.")
