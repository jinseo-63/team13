import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

# 인프라 카테고리별 지도 마커 스타일 설정
INFRA_STYLE = {
    '편의점': {'color': 'blue', 'icon': 'shopping-cart'},
    '지하철역': {'color': 'red', 'icon': 'train'},
    '코인세탁소': {'color': 'green', 'icon': 'tint'},
    '마트': {'color': 'orange', 'icon': 'shopping-bag'}
}


# ==========================================
# 1. 데이터 로드 및 전처리 함수
# ==========================================
def load_uploaded_csv(uploaded_file):
    """UTF-8 및 CP949 인코딩과 파싱 오류를 방어하며 CSV를 읽는 함수"""
    try:
        return pd.read_csv(uploaded_file, on_bad_lines='skip')
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, on_bad_lines='skip', encoding='cp949')


def get_dong_coords(dong_val):
    """법정동명 문자열에서 대표 좌표(위도, 경도)를 반환하는 보조 함수"""
    if pd.isna(dong_val):
        return (np.nan, np.nan)
    s = str(dong_val).strip()
    if s in GEUMJEONG_DONG_COORDS:
        return GEUMJEONG_DONG_COORDS[s]
    for dong_key, coords in GEUMJEONG_DONG_COORDS.items():
        prefix = dong_key[:-1]
        if prefix and prefix in s:
            return coords
    return (np.nan, np.nan)


def process_rent_data(df):
    """전월세 CSV 전처리 및 동별 대표 좌표 매핑 함수"""
    initial_count = len(df)
    df = df.copy()
    df.columns = df.columns.astype(str).str.replace('\ufeff', '', regex=False).str.strip()

    rename_dict = {
        '법정동명': '동', 'umdNm': '동', 'dong': '동', '법정동': '동',
        'houseType': '주택유형', 'buildYear': '건축년도',
        'deposit': '보증금', '보증금액': '보증금',
        'monthlyRent': '월세', '월세금액': '월세',
        'area': '면적', 'excluUseAr': '면적', '전용면적': '면적', '계약면적': '면적'
    }
    df = df.rename(columns=rename_dict)
    df = df.loc[:, ~df.columns.duplicated()]

    # '동' 컬럼 자동 탐색
    if '동' not in df.columns:
        for col in df.columns:
            sample_vals = df[col].dropna().astype(str).head(20)
            if any(any(k[:-1] in v for k in GEUMJEONG_DONG_COORDS) for v in sample_vals):
                df['동'] = df[col]
                break

    # 필수 가격 컬럼 결측치만 제외 (지번, 건물명 결측치는 유지)
    essential_cols = [c for c in ['보증금', '월세'] if c in df.columns]
    if essential_cols:
        df = df.dropna(subset=essential_cols).copy()
    dropped_count = initial_count - len(df)

    if '보증금' in df.columns:
        df['보증금'] = pd.to_numeric(df['보증금'].astype(str).str.replace(',', ''), errors='coerce').fillna(0).astype(int)
    if '월세' in df.columns:
        df['월세'] = pd.to_numeric(df['월세'].astype(str).str.replace(',', ''), errors='coerce').fillna(0).astype(int)
        df['거래유형'] = df['월세'].apply(lambda x: '전세' if x == 0 else '월세')
    if '면적' in df.columns:
        df['면적'] = pd.to_numeric(df['면적'], errors='coerce')

    if '계약일' not in df.columns and all(c in df.columns for c in ['dealYear', 'dealMonth', 'dealDay']):
        df['계약일'] = (
            df['dealYear'].astype(str) + "-" +
            df['dealMonth'].astype(str).str.zfill(2) + "-" +
            df['dealDay'].astype(str).str.zfill(2)
        )

    if '동' in df.columns:
        df['동'] = df['동'].astype(str).str.strip()
        coords_series = df['동'].map(get_dong_coords)
        df['위도'] = [c[0] for c in coords_series]
        df['경도'] = [c[1] for c in coords_series]
    else:
        df['위도'] = np.nan
        df['경도'] = np.nan

    return df, dropped_count


def process_infra_data(df):
    """인프라 CSV 전처리 및 좌표 정제 함수"""
    initial_count = len(df)
    df = df.copy()
    df.columns = df.columns.astype(str).str.replace('\ufeff', '', regex=False).str.strip()

    rename_dict = {
        '검색어': '구분', 'category_group_name': '구분',
        'place_name': '장소명',
        '경도(X)': '경도', '위도(Y)': '위도', 'x': '경도', 'y': '위도'
    }
    df = df.rename(columns=rename_dict)
    df = df.loc[:, ~df.columns.duplicated()]

    for col in ['경도', '위도']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = np.nan

    check_cols = [col for col in ['장소명', '경도', '위도'] if col in df.columns]
    df = df.dropna(subset=check_cols).copy() if check_cols else df.copy()
    dropped_count = initial_count - len(df)

    return df, dropped_count


# ==========================================
# 2. 핵심 분석 함수 3종
# ==========================================
def haversine_m(lat1, lon1, lat2_array, lon2_array):
    """하버사인 공식을 이용한 두 지점 간 직선거리(m) 계산"""
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2_array)
    dphi = np.radians(lat2_array - lat1)
    dlambda = np.radians(lon2_array - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def calculate_infra_distance(rent_df, infra_df, selected_infras=None):
    """
    [함수 1] 각 매물에서 4대 인프라별 최단 도보 거리(m) 및 선택 인프라 기반 접근성 점수 계산
    """
    merged_df = rent_df.copy()
    all_categories = ['편의점', '지하철역', '코인세탁소', '마트']
    active_categories = selected_infras if selected_infras else all_categories
    walk_factor = 1.3  # 직선거리 대비 도보 우회 계수

    has_rent_coords = ('위도' in merged_df.columns) and ('경도' in merged_df.columns)
    has_infra_cols = all(c in infra_df.columns for c in ['구분', '위도', '경도'])

    for cat in all_categories:
        col_name = f'{cat}_최단거리(m)'
        if not has_rent_coords or not has_infra_cols:
            merged_df[col_name] = np.nan
            continue

        # '마트' 검색 시 '대형마트'와 '마트' 모두 매칭
        sub_infra = infra_df[infra_df['구분'].astype(str).str.contains(cat, na=False)]
        if sub_infra.empty:
            merged_df[col_name] = np.nan
            continue

        infra_lats = sub_infra['위도'].values
        infra_lons = sub_infra['경도'].values

        min_distances = []
        for _, row in merged_df.iterrows():
            lat = row.get('위도', np.nan)
            lon = row.get('경도', np.nan)
            if pd.isna(lat) or pd.isna(lon):
                min_distances.append(np.nan)
            else:
                dists = haversine_m(lat, lon, infra_lats, infra_lons)
                min_distances.append(round(float(np.min(dists)) * walk_factor, 1))

        merged_df[col_name] = min_distances

    # 선택된 필수 인프라 기준 평균 도보 거리 및 접근성 점수(100점 만점) 산출
    active_dist_cols = [f'{cat}_최단거리(m)' for cat in active_categories if f'{cat}_최단거리(m)' in merged_df.columns]
    avg_dist_list = []
    score_list = []

    for _, row in merged_df.iterrows():
        valid_dists = [row[c] for c in active_dist_cols if pd.notna(row[c])]
        if not valid_dists:
            avg_dist_list.append(np.nan)
            score_list.append(np.nan)
        else:
            avg_dist_list.append(round(float(np.mean(valid_dists)), 1))
            cat_scores = [max(0.0, 100.0 - (d / 20.0)) for d in valid_dists]
            score_list.append(round(float(np.mean(cat_scores)), 2))

    merged_df['선택인프라_평균거리(m)'] = avg_dist_list
    merged_df['인프라 접근성 점수'] = score_list
    return merged_df


def analyze_correlation(merged_df):
    """
    [함수 2] 월세(전세 제외) 및 보증금과 인프라 최단 도보 거리 간 피어슨 상관계수(r) 및 p-value 계산
    """
    dist_cols = [c for c in merged_df.columns if c.endswith('_최단거리(m)') or c == '선택인프라_평균거리(m)']
    results = []

    if '거래유형' in merged_df.columns and '월세' in merged_df.columns:
        monthly_df = merged_df[merged_df['거래유형'] == '월세']
        for d_col in dist_cols:
            valid = monthly_df[['월세', d_col]].dropna()
            if len(valid) >= 3 and valid['월세'].nunique() > 1 and valid[d_col].nunique() > 1:
                r, p = pearsonr(valid['월세'], valid[d_col])
                results.append({
                    '가격 지표': '월세(전세제외)',
                    '인프라 거리': d_col,
                    '피어슨 상관계수(r)': round(r, 4),
                    'p-value': round(p, 4),
                    '유의미한 관계 여부(p<0.05)': '유의미함' if p < 0.05 else '유의미하지 않음'
                })

    if '보증금' in merged_df.columns:
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


def get_value_ranking(merged_df):
    """
    [함수 3] 구역(동)별 평균 월세 대비 인프라 접근성 점수를 비교한 가성비 자취 구역 순위표 반환
    """
    required_cols = ['거래유형', '동', '월세', '보증금', '인프라 접근성 점수']
    if not all(c in merged_df.columns for c in required_cols):
        return pd.DataFrame()

    monthly_df = merged_df[merged_df['거래유형'] == '월세'].dropna(subset=['동', '월세', '인프라 접근성 점수'])
    if monthly_df.empty:
        return pd.DataFrame()

    grouped = monthly_df.groupby('동').agg(
        월세_거래수=('월세', 'count'),
        평균_월세=('월세', 'mean'),
        평균_보증금=('보증금', 'mean'),
        평균_인프라점수=('인프라 접근성 점수', 'mean')
    ).reset_index()

    grouped = grouped[grouped['평균_월세'] > 0].copy()
    if grouped.empty:
        return pd.DataFrame()

    grouped['가성비 지수'] = (grouped['평균_인프라점수'] / grouped['평균_월세']).round(2)
    grouped['평균_월세'] = grouped['평균_월세'].round(1)
    grouped['평균_보증금'] = grouped['평균_보증금'].round(0).astype(int)
    grouped['평균_인프라점수'] = grouped['평균_인프라점수'].round(2)

    ranked_df = grouped.sort_values(by='가성비 지수', ascending=False).reset_index(drop=True)
    ranked_df.insert(0, '순위', range(1, len(ranked_df) + 1))

    return ranked_df


# ==========================================
# 3. 지도 및 시각화 차트 생성 함수
# ==========================================
def create_folium_map(filtered_df, infra_df, selected_infras):
    """원룸 실거래 히트맵과 선택 인프라 마커를 오버레이한 Folium 지도 생성"""
    valid_rent = filtered_df.dropna(subset=['위도', '경도'])
    if not valid_rent.empty:
        center_lat = valid_rent['위도'].mean()
        center_lon = valid_rent['경도'].mean()
    else:
        center_lat, center_lon = 35.2335, 129.0846  # 부산대 기본 좌표

    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles='CartoDB positron')

    # 1) 원룸 실거래가 분포 히트맵 레이어
    if not valid_rent.empty:
        # 월세 또는 거래 밀도를 가중치로 활용
        weights = valid_rent['월세'].replace(0, valid_rent['월세'].mean() if valid_rent['월세'].mean() > 0 else 1)
        heat_data = [[row['위도'], row['경도'], float(w)] for (_, row), w in zip(valid_rent.iterrows(), weights)]
        HeatMap(
            heat_data,
            name='원룸 실거래 히트맵',
            radius=28,
            blur=20,
            max_zoom=16
        ).add_to(m)

    # 2) 사이드바에서 선택한 필수 인프라 마커 오버레이
    if not infra_df.empty and selected_infras:
        for cat in selected_infras:
            style = INFRA_STYLE.get(cat, {'color': 'cadetblue', 'icon': 'info-sign'})
            sub_infra = infra_df[infra_df['구분'].astype(str).str.contains(cat, na=False)]
            fg = folium.FeatureGroup(name=f'인프라: {cat}')

            for _, row in sub_infra.iterrows():
                place_name = row.get('장소명', '이름 없음')
                category_label = row.get('구분', cat)
                tooltip_text = f"[{category_label}] {place_name}"
                popup_html = f"<b>{place_name}</b><br>구분: {category_label}"

                folium.Marker(
                    location=[row['위도'], row['경도']],
                    tooltip=tooltip_text,
                    popup=folium.Popup(popup_html, max_width=250),
                    icon=folium.Icon(color=style['color'], icon=style['icon'], prefix='fa')
                ).add_to(fg)

            fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def plot_dong_comparison(rank_df):
    """[차트 1] 구역별 평균 월세 vs 인프라 접근성 점수 이중축 비교 차트"""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=rank_df['동'],
            y=rank_df['평균_월세'],
            name='평균 월세 (만 원)',
            marker_color='#4A90E2',
            text=rank_df['평균_월세'],
            textposition='auto'
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=rank_df['동'],
            y=rank_df['평균_인프라점수'],
            name='인프라 접근성 점수 (점)',
            mode='lines+markers+text',
            marker=dict(size=10, color='#E74C3C'),
            line=dict(width=3, color='#E74C3C'),
            text=rank_df['평균_인프라점수'],
            textposition='top center'
        ),
        secondary_y=True
    )

    fig.update_layout(
        title='구역(동)별 평균 월세 vs 인프라 접근성 점수 비교',
        xaxis_title='법정동',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=420,
        margin=dict(t=70, b=40, l=40, r=40)
    )
    fig.update_yaxes(title_text='평균 월세 (만 원)', secondary_y=False)
    fig.update_yaxes(title_text='인프라 접근성 점수 (100점 만점)', range=[0, 105], secondary_y=True)
    return fig


def plot_correlation_scatter(filtered_df, dist_col='선택인프라_평균거리(m)'):
    """[차트 2] 월세와 인프라 도보 거리 상관관계 산점도 (추세선 및 상관계수 표시)"""
    monthly_df = filtered_df[filtered_df['거래유형'] == '월세'].dropna(subset=['월세', dist_col]).copy()

    if monthly_df.empty:
        fig = go.Figure()
        fig.update_layout(title='월세 데이터가 없어 산점도를 표시할 수 없습니다.', height=420)
        return fig

    fig = px.scatter(
        monthly_df,
        x=dist_col,
        y='월세',
        color='동' if '동' in monthly_df.columns else None,
        hover_data=[c for c in ['계약일', '동', '보증금', '면적'] if c in monthly_df.columns],
        labels={dist_col: f'{dist_col}', '월세': '월세 (만 원)'}
    )

    # 추세선 및 상관계수 계산
    annotation_text = "상관계수 산출 불가 (표본 또는 분산 부족)"
    if len(monthly_df) >= 3 and monthly_df[dist_col].nunique() > 1 and monthly_df['월세'].nunique() > 1:
        r, p = pearsonr(monthly_df[dist_col], monthly_df['월세'])
        slope, intercept = np.polyfit(monthly_df[dist_col], monthly_df['월세'], 1)
        x_range = np.linspace(monthly_df[dist_col].min(), monthly_df[dist_col].max(), 100)
        y_pred = slope * x_range + intercept

        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_pred,
                mode='lines',
                name='회귀 추세선',
                line=dict(color='#2C3E50', dash='dash', width=2)
            )
        )
        annotation_text = f"Pearson r = {r:.4f} (p = {p:.4f})"

    fig.add_annotation(
        xref='paper', yref='paper',
        x=0.02, y=0.98,
        text=annotation_text,
        showarrow=False,
        bgcolor='rgba(255, 255, 255, 0.85)',
        bordercolor='#2C3E50',
        borderwidth=1,
        font=dict(size=12, color='#2C3E50')
    )

    fig.update_layout(
        title=f'월세 vs {dist_col} 상관관계 분석',
        height=420,
        margin=dict(t=70, b=40, l=40, r=40)
    )
    return fig


def plot_price_distribution(filtered_df):
    """[차트 3] 선택한 조건의 보증금 및 월세 분포 히스토그램 + 박스플롯"""
    fig = make_subplots(
        rows=2, cols=2,
        row_heights=[0.25, 0.75],
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=('보증금 분포 (만 원)', '월세 분포 (만 원, 전세 제외)')
    )

    # 1) 보증금 박스플롯 + 히스토그램
    if '보증금' in filtered_df.columns and not filtered_df['보증금'].dropna().empty:
        fig.add_trace(
            go.Box(x=filtered_df['보증금'], name='보증금', marker_color='#3498DB', boxmean=True),
            row=1, col=1
        )
        fig.add_trace(
            go.Histogram(x=filtered_df['보증금'], name='보증금 빈도', marker_color='#3498DB', opacity=0.75),
            row=2, col=1
        )

    # 2) 월세(전세 제외) 박스플롯 + 히스토그램
    monthly_df = filtered_df[filtered_df['거래유형'] == '월세'] if '거래유형' in filtered_df.columns else filtered_df
    if '월세' in monthly_df.columns and not monthly_df['월세'].dropna().empty:
        fig.add_trace(
            go.Box(x=monthly_df['월세'], name='월세', marker_color='#2ECC71', boxmean=True),
            row=1, col=2
        )
        fig.add_trace(
            go.Histogram(x=monthly_df['월세'], name='월세 빈도', marker_color='#2ECC71', opacity=0.75),
            row=2, col=2
        )

    fig.update_layout(
        title='필터링된 매물의 보증금 및 월세 가격 분포 (박스플롯 & 히스토그램)',
        showlegend=False,
        height=440,
        margin=dict(t=70, b=40, l=40, r=40)
    )
    fig.update_yaxes(title_text='매물 수 (건)', row=2, col=1)
    fig.update_yaxes(title_text='매물 수 (건)', row=2, col=2)
    return fig


# ==========================================
# 4. 메인 Streamlit 애플리케이션 실행 함수
# ==========================================
def main():
    st.set_page_config(page_title="부산대 원룸 및 인프라 분석 대시보드", layout="wide")
    st.title("부산대 인근 원룸 실거래가 및 생활 인프라 통합 분석")

    # ------------------------------------------
    # [왼쪽 사이드바] 데이터 업로드 및 검색 필터
    # ------------------------------------------
    st.sidebar.header("1. 데이터 파일 업로드")
    rent_file = st.sidebar.file_uploader("전월세 CSV 파일", type=['csv'], key="rent_csv")
    infra_file = st.sidebar.file_uploader("인프라 CSV 파일", type=['csv'], key="infra_csv")

    st.sidebar.markdown("---")
    st.sidebar.header("2. 검색 및 분석 필터")

    # 희망 주거 형태 라디오 버튼
    housing_choice = st.sidebar.radio(
        "희망 주거 형태",
        options=["전체", "전세", "월세"],
        index=0,
        horizontal=True
    )

    # 필수 인프라 선택 체크박스
    st.sidebar.write("필수 인프라 선택 (지도 표시 및 점수 반영)")
    chk_cvs = st.sidebar.checkbox("편의점", value=True)
    chk_subway = st.sidebar.checkbox("지하철역", value=True)
    chk_laundry = st.sidebar.checkbox("코인세탁소", value=True)
    chk_mart = st.sidebar.checkbox("마트", value=True)

    selected_infras = []
    if chk_cvs:
        selected_infras.append("편의점")
    if chk_subway:
        selected_infras.append("지하철역")
    if chk_laundry:
        selected_infras.append("코인세탁소")
    if chk_mart:
        selected_infras.append("마트")

    # 파일이 모두 업로드되지 않은 경우 안내 메시지 출력
    if rent_file is None or infra_file is None:
        st.info("왼쪽 사이드바에서 '전월세 CSV 파일'과 '인프라 CSV 파일'을 모두 업로드해 주세요.")
        return

    # 데이터 로드 및 전처리 실행
    raw_rent = load_uploaded_csv(rent_file)
    raw_infra = load_uploaded_csv(infra_file)

    rent_df, rent_dropped = process_rent_data(raw_rent)
    infra_df, infra_dropped = process_infra_data(raw_infra)

    # 1단계 분석: 인프라 거리 및 선택 인프라 기반 접근성 점수 계산
    merged_df = calculate_infra_distance(rent_df, infra_df, selected_infras)

    # 보증금 및 월세 예산 범위 조절 슬라이더 (데이터 범위 기반 동적 설정)
    min_dep = int(merged_df['보증금'].min()) if not merged_df.empty else 0
    max_dep = int(merged_df['보증금'].max()) if not merged_df.empty else 10000
    if min_dep >= max_dep:
        max_dep = min_dep + 100

    min_rent = int(merged_df['월세'].min()) if not merged_df.empty else 0
    max_rent = int(merged_df['월세'].max()) if not merged_df.empty else 200
    if min_rent >= max_rent:
        max_rent = min_rent + 10

    st.sidebar.markdown("---")
    deposit_range = st.sidebar.slider(
        "보증금 예산 범위 (만 원)",
        min_value=min_dep,
        max_value=max_dep,
        value=(min_dep, max_dep),
        step=50 if (max_dep - min_dep) >= 50 else 1
    )

    rent_range = st.sidebar.slider(
        "월세 예산 범위 (만 원)",
        min_value=min_rent,
        max_value=max_rent,
        value=(min_rent, max_rent),
        step=5 if (max_rent - min_rent) >= 5 else 1
    )

    # 사이드바 필터 조건 적용
    filtered_df = merged_df.copy()
    if housing_choice != "전체" and '거래유형' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['거래유형'] == housing_choice]

    filtered_df = filtered_df[
        (filtered_df['보증금'] >= deposit_range[0]) & (filtered_df['보증금'] <= deposit_range[1]) &
        (filtered_df['월세'] >= rent_range[0]) & (filtered_df['월세'] <= rent_range[1])
    ]

    st.caption(f"데이터 전처리 완료: 전월세 필수값 결측 제외 {rent_dropped}건 / 인프라 좌표 결측 제외 {infra_dropped}건")

    # ------------------------------------------
    # [메인 화면 상단] 핵심 요약 지표 (KPI 카드)
    # ------------------------------------------
    st.subheader("1. 핵심 요약 지표")
    total_cnt = len(filtered_df)
    avg_deposit = filtered_df['보증금'].mean() if total_cnt > 0 else 0

    monthly_only = filtered_df[filtered_df['거래유형'] == '월세'] if '거래유형' in filtered_df.columns else filtered_df
    avg_monthly_rent = monthly_only['월세'].mean() if not monthly_only.empty else 0

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("조건 부합 원룸 매물 수", f"{total_cnt:,}건")
    kpi2.metric("평균 보증금", f"{avg_deposit:,.0f}만 원")
    kpi3.metric("평균 월세 (전세 제외)", f"{avg_monthly_rent:,.1f}만 원")

    st.markdown("---")

    # ------------------------------------------
    # [메인 화면 중단] 인터랙티브 지도 (Folium)
    # ------------------------------------------
    st.subheader("2. 원룸 실거래 분포 히트맵 및 필수 인프라 지도")
    st.write("지도 우측 상단 레이어 컨트롤을 통해 히트맵과 인프라별 마커를 켜고 끌 수 있으며, 마커에 마우스를 올리거나 클릭하면 장소 정보를 확인할 수 있습니다.")

    folium_map = create_folium_map(filtered_df, infra_df, selected_infras)
    st_folium(folium_map, width=None, height=500, returned_objects=[])

    st.markdown("---")

    # ------------------------------------------
    # [메인 화면 하단] 상세 데이터 및 차트 3종
    # ------------------------------------------
    st.subheader("3. 구역별 가성비 순위 및 상관관계 분석")
    rank_df = get_value_ranking(filtered_df)
    corr_df = analyze_correlation(filtered_df)

    col_rank, col_corr = st.columns([1.2, 1])
    with col_rank:
        st.markdown("**구역(동)별 가성비 및 인프라 접근성 순위표**")
        if not rank_df.empty:
            st.dataframe(rank_df, use_container_width=True, hide_index=True)
        else:
            st.warning("순위를 산출할 수 있는 월세 매물 데이터가 부족합니다.")

    with col_corr:
        st.markdown("**가격(월세/보증금)과 인프라 거리 피어슨 상관관계**")
        if not corr_df.empty:
            st.dataframe(corr_df, use_container_width=True, hide_index=True)
        else:
            st.info("상관계수 계산을 위한 표본 또는 거리 분산이 부족합니다.")

    st.markdown("---")
    st.subheader("4. 보고서용 분석 시각화 차트 3종")

    # 차트 1 & 차트 2 가로 2단 배치
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        if not rank_df.empty:
            fig1 = plot_dong_comparison(rank_df)
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.warning("구역별 비교 차트를 그릴 데이터가 부족합니다.")

    with chart_col2:
        dist_options = [c for c in ['선택인프라_평균거리(m)', '편의점_최단거리(m)', '지하철역_최단거리(m)', '코인세탁소_최단거리(m)', '마트_최단거리(m)'] if c in filtered_df.columns]
        selected_dist_col = st.selectbox("산점도 X축 거리 기준 선택", options=dist_options, index=0)
        fig2 = plot_correlation_scatter(filtered_df, dist_col=selected_dist_col)
        st.plotly_chart(fig2, use_container_width=True)

    # 차트 3 전체 폭 배치
    fig3 = plot_price_distribution(filtered_df)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # ------------------------------------------
    # [메인 화면 최하단] 필터링된 원룸 목록 테이블 + CSV 다운로드
    # ------------------------------------------
    st.subheader("5. 필터링된 원룸 매물 목록 및 데이터 다운로드")
    display_cols = [
        '계약일', '동', '거래유형', '보증금', '월세', '면적', '건축년도',
        '편의점_최단거리(m)', '지하철역_최단거리(m)', '코인세탁소_최단거리(m)', '마트_최단거리(m)',
        '선택인프라_평균거리(m)', '인프라 접근성 점수'
    ]
    final_cols = [c for c in display_cols if c in filtered_df.columns]
    export_df = filtered_df[final_cols]

    st.dataframe(export_df, use_container_width=True, hide_index=True)

    csv_bytes = export_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="필터링된 원룸 데이터 CSV 다운로드",
        data=csv_bytes,
        file_name="pnu_filtered_oneroom_infra.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    main()
