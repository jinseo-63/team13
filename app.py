import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# 부산 금정구 법정동별 대표 좌표 (위도, 경도) 매핑 딕셔너리
GEUMJEONG_DONG_COORDS = {
    '장전동': (35.2335, 129.0846),
    '부곡동': (35.2255, 129.0921),
    '구서동': (35.2476, 129.0913),
    '남산동': (35.2654, 129.0923),
    '서동': (35.2144, 129.1032),
    '금사동': (35.2156, 129.1155),
    '회동동': (35.2223, 129.1245),
    '청룡동': (35.2768, 129.0901),
    '노포동': (35.2844, 129.0951),
    '선동': (35.2450, 129.1150),
    '두구동': (35.2720, 129.1080),
    '오륜동': (35.2350, 129.1100),
    '금성동': (35.2458, 129.0558)
}

# 0. CSV 파일 읽기 헬퍼 함수 (인코딩 및 파싱 에러 방지)
def load_uploaded_csv(uploaded_file):
    try:
        return pd.read_csv(uploaded_file, on_bad_lines='skip')
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, on_bad_lines='skip', encoding='cp949')

# 1. 전월세 데이터 전처리 함수
def process_rent_data(df):
    initial_count = len(df)
    
    # 필수 가격/날짜 컬럼에 결측치가 있는 행만 제외 (지번, 건물명 결측치는 유지)
    essential_cols = ['deposit', 'monthlyRent', 'dealYear', 'dealMonth', 'dealDay']
    check_cols = [col for col in essential_cols if col in df.columns]
    df = df.dropna(subset=check_cols).copy() if check_cols else df.copy()
    dropped_count = initial_count - len(df)
    
    # 보증금, 월세, 면적 숫자형 변환
    if 'deposit' in df.columns:
        df['보증금'] = df['deposit'].astype(str).str.replace(',', '').astype(float).astype(int)
    if 'monthlyRent' in df.columns:
        df['월세'] = df['monthlyRent'].astype(str).str.replace(',', '').astype(float).astype(int)
        df['거래유형'] = df['월세'].apply(lambda x: '전세' if x == 0 else '월세')
    if 'area' in df.columns:
        df['면적'] = pd.to_numeric(df['area'], errors='coerce')
        
    # 계약일 조합
    if all(c in df.columns for c in ['dealYear', 'dealMonth', 'dealDay']):
        df['계약일'] = (df['dealYear'].astype(str) + "-" + 
                      df['dealMonth'].astype(str).str.zfill(2) + "-" + 
                      df['dealDay'].astype(str).str.zfill(2))
        
    # 컬럼명 통일 (법정동명 또는 dong -> 동)
    rename_dict = {'법정동명': '동', 'dong': '동', '법정동': '동', 'houseType': '주택유형', 'buildYear': '건축년도'}
    df = df.rename(columns=rename_dict)
    df = df.loc[:, ~df.columns.duplicated()]
    
    # 동 이름 공백 제거 및 대표 위도/경도 매핑
    if '동' in df.columns:
        df['동'] = df['동'].astype(str).str.strip()
        df['위도'] = df['동'].map(lambda x: GEUMJEONG_DONG_COORDS.get(x, (np.nan, np.nan))[0])
        df['경도'] = df['동'].map(lambda x: GEUMJEONG_DONG_COORDS.get(x, (np.nan, np.nan))[1])
        
    return df, dropped_count

# 2. 인프라 데이터 전처리 함수
def process_infra_data(df):
    initial_count = len(df)
    
    # 컬럼명 통일
    rename_dict = {'검색어': '구분', '경도(X)': '경도', '위도(Y)': '위도', 'x': '경도', 'y': '위도'}
    df = df.rename(columns=rename_dict)
    df = df.loc[:, ~df.columns.duplicated()]
    
    # 좌표 숫자형 변환 및 결측치 제외
    for col in ['경도', '위도']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    check_cols = [col for col in ['장소명', '경도', '위도'] if col in df.columns]
    df = df.dropna(subset=check_cols).copy() if check_cols else df.copy()
    dropped_count = initial_count - len(df)
    
    return df, dropped_count

# 하버사인(Haversine) 직선거리 계산 보조 함수 (단위: m)
def haversine_m(lat1, lon1, lat2_array, lon2_array):
    R = 6371000  # 지구 반지름 (미터)
    phi1, phi2 = np.radians(lat1), np.radians(lat2_array)
    dphi = np.radians(lat2_array - lat1)
    dlambda = np.radians(lon2_array - lon1)
    
    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# [분석 함수 1] 인프라 최단 도보 거리 및 접근성 점수 계산
def calculate_infra_distance(rent_df, infra_df):
    """
    [입력] 전처리된 전월세 데이터프레임(rent_df), 전처리된 인프라 데이터프레임(infra_df)
    [처리] 하버사인 공식으로 각 매물에서 4대 인프라별 최단 직선거리를 구한 뒤,
           도보 우회 계수(1.3)를 곱해 도보 거리를 산출하고 100점 만점의 접근성 점수를 계산
    [출력] 인프라별 최단 도보 거리(m) 및 '인프라 접근성 점수' 컬럼이 추가된 데이터프레임
    """
    merged_df = rent_df.copy()
    infra_categories = ['편의점', '코인세탁소', '대형마트', '지하철역']
    walk_factor = 1.3  # 직선거리 대비 실제 도보 거리 보정 계수
    
    # 카테고리별 최단 도보 거리 계산
    for cat in infra_categories:
        # 구분 컬럼에 해당 키워드가 포함된 인프라 필터링
        sub_infra = infra_df[infra_df['구분'].astype(str).str.contains(cat, na=False)]
        col_name = f'{cat}_최단거리(m)'
        
        if sub_infra.empty:
            merged_df[col_name] = np.nan
            continue
            
        infra_lats = sub_infra['위도'].values
        infra_lons = sub_infra['경도'].values
        
        min_distances = []
        for _, row in merged_df.iterrows():
            if pd.isna(row['위도']) or pd.isna(row['경도']):
                min_distances.append(np.nan)
            else:
                dists = haversine_m(row['위도'], row['경도'], infra_lats, infra_lons)
                min_distances.append(round(float(np.min(dists)) * walk_factor, 1))
                
        merged_df[col_name] = min_distances

    # 인프라 접근성 점수 계산 (거리가 0m에 가까울수록 100점, 2000m 이상이면 0점 기준 평균)
    dist_cols = [f'{cat}_최단거리(m)' for cat in infra_categories]
    score_list = []
    for _, row in merged_df.iterrows():
        valid_dists = [row[c] for c in dist_cols if pd.notna(row[c])]
        if not valid_dists:
            score_list.append(np.nan)
        else:
            # 각 인프라별 점수(최대 100점, 2000m당 100점 감점)의 평균 산출
            cat_scores = [max(0.0, 100.0 - (d / 20.0)) for d in valid_dists]
            score_list.append(round(float(np.mean(cat_scores)), 2))
            
    merged_df['인프라 접근성 점수'] = score_list
    return merged_df

# [분석 함수 2] 가격(월세, 보증금)과 인프라 거리 간 피어슨 상관관계 분석
def analyze_correlation(merged_df):
    """
    [입력] 인프라 거리가 추가된 데이터프레임(merged_df)
    [처리] 월세(전세 제외) 및 보증금과 각 인프라 최단 도보 거리 간의 피어슨 상관계수(r) 및 p-value 계산
    [출력] 변수 쌍별 상관계수(r), p-value, 통계적 유의성 여부(p < 0.05)를 담은 결과 데이터프레임
    """
    dist_cols = [c for c in merged_df.columns if c.endswith('_최단거리(m)')]
    results = []
    
    # 1) 월세(전세 제외)와 인프라 거리 간 상관분석
    monthly_df = merged_df[merged_df['거래유형'] == '월세']
    for d_col in dist_cols:
        valid = monthly_df[['월세', d_col]].dropna()
        # 상관계수 계산을 위해 최소 3개 이상의 데이터 및 분산(>0) 필요
        if len(valid) >= 3 and valid['월세'].nunique() > 1 and valid[d_col].nunique() > 1:
            r, p = pearsonr(valid['월세'], valid[d_col])
            results.append({
                '가격 지표': '월세(전세제외)',
                '인프라 거리': d_col,
                '피어슨 상관계수(r)': round(r, 4),
                'p-value': round(p, 4),
                '유의미한 관계 여부(p<0.05)': '유의미함' if p < 0.05 else '유의미하지 않음'
            })
            
    # 2) 보증금(전체)과 인프라 거리 간 상관분석
    for d_col in dist_cols:
        valid = merged_df[['보증금', d_col]].dropna()
        if len(valid) >= 3 and valid['보증금'].nunique() > 1 and valid[d_col].nunique() > 1:
            r, p = pearsonr(valid['보증금'], valid[d_col])
            results.append({
                '가격 지표': '보증금',
                '인프라 거리': d_col,
                '피어슨 상관계수(r)': round(r, 4),
                'p-value': round(p, 4),
                '유의미한 관계 여부(p<0.05)': '유의미함' if p < 0.05 else '유의미하지 않음'
            })
            
    return pd.DataFrame(results)

# [분석 함수 3] 동(구역)별 가성비 자취 구역 순위표 생성
def get_value_ranking(merged_df):
    """
    [입력] 인프라 접근성 점수가 포함된 데이터프레임(merged_df)
    [처리] 전세를 제외한 월세 매물을 대상으로 동별 평균 월세와 평균 인프라 접근성 점수를 집계하고,
           '평균 인프라 접근성 점수 / 평균 월세' 비율로 가성비 지수를 산출해 내림차순 정렬
    [출력] 가성비 순위가 매겨진 동별 요약 데이터프레임
    """
    monthly_df = merged_df[merged_df['거래유형'] == '월세'].dropna(subset=['동', '월세', '인프라 접근성 점수'])
    
    if monthly_df.empty:
        return pd.DataFrame()
        
    grouped = monthly_df.groupby('동').agg(
        월세_거래수=('월세', 'count'),
        평균_월세=('월세', 'mean'),
        평균_보증금=('보증금', 'mean'),
        평균_인프라점수=('인프라 접근성 점수', 'mean')
    ).reset_index()
    
    # 가성비 지수 = 평균 인프라 접근성 점수 / 평균 월세 (월세 1만 원당 얻는 인프라 점수)
    grouped['가성비 지수'] = (grouped['평균_인프라점수'] / grouped['평균_월세']).round(2)
    
    # 반올림 및 가성비 지수 기준 내림차순 정렬
    grouped['평균_월세'] = grouped['평균_월세'].round(1)
    grouped['평균_보증금'] = grouped['평균_보증금'].round(0).astype(int)
    grouped['평균_인프라점수'] = grouped['평균_인프라점수'].round(2)
    
    ranked_df = grouped.sort_values(by='가성비 지수', ascending=False).reset_index(drop=True)
    ranked_df.insert(0, '순위', range(1, len(ranked_df) + 1))
    
    return ranked_df

# 3. 메인 애플리케이션 실행 함수
def main():
    st.set_page_config(layout="wide")
    st.title("부산대 인근 원룸 실거래가 및 인프라 통합 분석")
    
    col_left, col_right = st.columns([1, 2.5])

    # 왼쪽 영역: 2개 CSV 파일 동시 업로드
    with col_left:
        st.subheader("1. 데이터 파일 업로드")
        st.write("전월세 CSV와 인프라 CSV 파일을 모두 업로드해 주세요.")
        
        rent_file = st.file_uploader("전월세 CSV 파일 선택", type=['csv'], key="rent")
        infra_file = st.file_uploader("인프라 CSV 파일 선택", type=['csv'], key="infra")
        
        st.info("안내: 전월세 데이터에 위도/경도가 없으므로 법정동명 기준 대표 좌표를 자동 매핑하여 도보 거리를 계산합니다.")

    # 오른쪽 영역: 단계별 분석 결과 표시
    with col_right:
        st.subheader("2. 분석 결과")
        
        if rent_file is not None and infra_file is not None:
            # 데이터 로드 및 전처리
            raw_rent = load_uploaded_csv(rent_file)
            raw_infra = load_uploaded_csv(infra_file)
            
            rent_df, rent_dropped = process_rent_data(raw_rent)
            infra_df, infra_dropped = process_infra_data(raw_infra)
            
            st.write(f"전처리 완료 - 전월세 결측치 제외: {rent_dropped}건 / 인프라 결측치 제외: {infra_dropped}건")
            
            # 3단계 분석 함수 실행
            merged_df = calculate_infra_distance(rent_df, infra_df)
            corr_df = analyze_correlation(merged_df)
            rank_df = get_value_ranking(merged_df)
            
            # 탭으로 결과 구분 표시
            tab1, tab2, tab3 = st.tabs(["1단계: 인프라 거리 및 점수", "2단계: 상관관계 분석", "3단계: 가성비 구역 순위"])
            
            with tab1:
                m1, m2, m3 = st.columns(3)
                m1.metric("총 거래 수", f"{len(merged_df)}건")
                avg_score = merged_df['인프라 접근성 점수'].mean()
                m2.metric("평균 인프라 점수", f"{avg_score:.1f}점" if pd.notna(avg_score) else "계산 불가")
                m3.metric("등록된 인프라 수", f"{len(infra_df)}곳")
                
                show_cols = ['계약일', '동', '거래유형', '보증금', '월세', 
                             '편의점_최단거리(m)', '코인세탁소_최단거리(m)', '대형마트_최단거리(m)', '지하철역_최단거리(m)', 
                             '인프라 접근성 점수']
                final_cols = [c for c in show_cols if c in merged_df.columns]
                st.dataframe(merged_df[final_cols], use_container_width=True)
                
            with tab2:
                st.write("월세(전세 제외) 및 보증금과 4대 인프라 최단 도보 거리 간의 피어슨 상관관계 분석 결과입니다.")
                if not corr_df.empty:
                    st.dataframe(corr_df, use_container_width=True)
                else:
                    st.warning("상관관계를 계산할 수 있는 유효한 데이터가 부족합니다.")
                    
            with tab3:
                st.write("동별 평균 월세(전세 제외) 대비 평균 인프라 접근성 점수를 비교한 가성비 순위표입니다.")
                if not rank_df.empty:
                    st.dataframe(rank_df, use_container_width=True)
                else:
                    st.warning("가성비 순위를 산출할 수 있는 월세 데이터가 없습니다.")
        else:
            st.warning("왼쪽 영역에서 전월세 CSV와 인프라 CSV 파일을 모두 업로드해 주세요.")

if __name__ == "__main__":
    main()
