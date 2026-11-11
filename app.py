import streamlit as st
import pandas as pd

# 1. 전월세 데이터 전처리 함수
def process_rent_data(df):
    initial_count = len(df)
    
    essential_cols = ['deposit', 'monthlyRent', 'dealYear', 'dealMonth', 'dealDay']
    check_cols = [col for col in essential_cols if col in df.columns]
    
    if check_cols:
        df = df.dropna(subset=check_cols).copy()
    else:
        df = df.copy()
        
    dropped_count = initial_count - len(df)
    
    # 보증금 전처리
    if 'deposit' in df.columns:
        df['보증금'] = df['deposit'].astype(str).str.replace(',', '').astype(int)
    
    # 월세 및 거래유형 전처리
    if 'monthlyRent' in df.columns:
        df['월세'] = df['monthlyRent'].astype(int)
        df['거래유형'] = df['월세'].apply(lambda x: '전세' if x == 0 else '월세')
        
    if 'area' in df.columns:
        df['면적'] = df['area'].astype(float)
    
    # 계약일 문자열 조합
    if all(c in df.columns for c in ['dealYear', 'dealMonth', 'dealDay']):
        df['계약일'] = (df['dealYear'].astype(str) + "-" + 
                      df['dealMonth'].astype(str).str.zfill(2) + "-" + 
                      df['dealDay'].astype(str).str.zfill(2))
    
    # 영문 컬럼명을 한글로 변경
    rename_dict = {'dong': '동', 'houseType': '주택유형', 'buildYear': '건축년도'}
    df = df.rename(columns=rename_dict)
    
    # 중복 생성된 열 제거
    df = df.loc[:, ~df.columns.duplicated()]
    
    return df, dropped_count

# 2. 인프라 데이터 전처리 함수
def process_infra_data(df):
    initial_count = len(df)
    
    check_cols = [col for col in ['장소명', '경도(X)', '위도(Y)', '경도', '위도'] if col in df.columns]
    if check_cols:
        df = df.dropna(subset=check_cols).copy()
    else:
        df = df.copy()
        
    dropped_count = initial_count - len(df)
    
    # 검색어를 구분 컬럼으로 이름 변경
    if '검색어' in df.columns:
        df = df.rename(columns={'검색어': '구분'})
        
    df = df.loc[:, ~df.columns.duplicated()]
        
    return df, dropped_count

# 3. CSV 파일 읽기 헬퍼 함수 (에러 및 인코딩 방지)
def load_uploaded_csv(uploaded_file):
    try:
        # utf-8 인코딩으로 먼저 시도 (파싱 에러 시 해당 줄 건너뜀)
        return pd.read_csv(uploaded_file, on_bad_lines='skip')
    except UnicodeDecodeError:
        # 한글 깨짐(인코딩) 에러 발생 시 파일 포인터를 되돌리고 EUC-KR(cp949)로 재시도
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, on_bad_lines='skip', encoding='cp949')

# 4. 메인 애플리케이션 실행 함수
def main():
    st.set_page_config(layout="wide")
    
    col_left, col_right = st.columns([1, 2])

    # 왼쪽 영역: 데이터 업로드
    with col_left:
        st.subheader("데이터 파일 업로드")
        st.write("PC에 있는 CSV 파일을 직접 선택해 올려주세요.")
        
        # 조회할 데이터 선택
        data_type = st.radio("업로드할 데이터의 종류를 선택하세요.", ["전월세", "인프라"])
        
        # 파일 업로더 위젯
        uploaded_file = st.file_uploader("CSV 파일 선택", type=['csv'])

    # 오른쪽 영역: 결과 요약 및 테이블 표시
    with col_right:
        st.subheader("결과 요약 및 데이터")
        
        # 파일이 업로드된 경우에만 실행
        if uploaded_file is not None:
            raw_df = load_uploaded_csv(uploaded_file)
            
            if data_type == "전월세":
                rent_df, dropped = process_rent_data(raw_df)
                
                st.write(f"결측치 처리 안내: {dropped}개의 행이 제외되었습니다.")
                
                # 요약 지표 계산
                total_count = len(rent_df)
                avg_deposit = rent_df['보증금'].mean() if '보증금' in rent_df.columns and total_count > 0 else 0
                
                if '거래유형' in rent_df.columns and '월세' in rent_df.columns:
                    monthly_df = rent_df[rent_df['거래유형'] == '월세']
                    avg_rent = monthly_df['월세'].mean() if not monthly_df.empty else 0
                else:
                    avg_rent = 0
                
                m1, m2, m3 = st.columns(3)
                m1.metric("총 거래 수", f"{total_count}건")
                m2.metric("평균 보증금", f"{avg_deposit:,.0f}만 원")
                m3.metric("평균 월세(전세제외)", f"{avg_rent:,.0f}만 원")
                
                display_cols = ['계약일', '동', '주택유형', '보증금', '월세', '면적', '건축년도', '거래유형']
                final_cols = [c for c in display_cols if c in rent_df.columns]
                st.dataframe(rent_df[final_cols])
                
            elif data_type == "인프라":
                infra_df, dropped = process_infra_data(raw_df)
                
                st.write(f"결측치 처리 안내: {dropped}개의 행이 제외되었습니다.")
                
                # 요약 지표 계산
                total_places = len(infra_df)
                cvs_count = len(infra_df[infra_df['구분'].str.contains('편의점', na=False)]) if '구분' in infra_df.columns else 0
                laundry_count = len(infra_df[infra_df['구분'].str.contains('세탁', na=False)]) if '구분' in infra_df.columns else 0
                
                m1, m2, m3 = st.columns(3)
                m1.metric("전체 장소 수", f"{total_places}곳")
                m2.metric("편의점 수", f"{cvs_count}곳")
                m3.metric("코인세탁소 수", f"{laundry_count}곳")
                
                display_cols = ['구분', '장소명', '경도', '위도']
                final_cols = [c for c in display_cols if c in infra_df.columns]
                st.dataframe(infra_df[final_cols])
        else:
            st.info("왼쪽 영역에서 CSV 파일을 업로드해 주세요.")

if __name__ == "__main__":
    main()
