
import streamlit as st
import pandas as pd
import os


# 1. 전월세 데이터 전처리 함수
def process_rent_data(df):
    initial_count = len(df)

    # 💡 필수 컬럼에 결측치가 있을 때만 해당 행 삭제
    essential_cols = ['deposit', 'monthlyRent', 'dealYear', 'dealMonth', 'dealDay']
    check_cols = [col for col in essential_cols if col in df.columns]

    if check_cols:
        df = df.dropna(subset=check_cols).copy()
    else:
        df = df.copy()

    dropped_count = initial_count - len(df)

    # 보증금 전처리 (쉼표 제거 후 숫자형 변환)
    if 'deposit' in df.columns:
        df['보증금'] = df['deposit'].astype(str).str.replace(',', '').astype(int)

    # 월세 및 거래유형 전처리
    if 'monthlyRent' in df.columns:
        df['월세'] = df['monthlyRent'].astype(int)
        df['거래유형'] = df['월세'].apply(lambda x: '전세' if x == 0 else '월세')

    if 'area' in df.columns:
        df['면적'] = df['area'].astype(float)

    # 계약일 문자열 조합 (YYYY-MM-DD 형식)
    if all(c in df.columns for c in ['dealYear', 'dealMonth', 'dealDay']):
        df['계약일'] = (df['dealYear'].astype(str) + "-" +
                     df['dealMonth'].astype(str).str.zfill(2) + "-" +
                     df['dealDay'].astype(str).str.zfill(2))

    # 영문 컬럼명을 한글로 변경
    rename_dict = {'dong': '동', 'houseType': '주택유형', 'buildYear': '건축년도'}
    df = df.rename(columns=rename_dict)

    # 이름 중복 컬럼 방지 (중복 생성된 열 제거)
    df = df.loc[:, ~df.columns.duplicated()]

    return df, dropped_count


# 2. 인프라 데이터 전처리 함수
def process_infra_data(df):
    initial_count = len(df)

    # 💡 장소명이나 좌표가 없는 경우만 삭제
    check_cols = [col for col in ['장소명', '경도(X)', '위도(Y)', '경도', '위도'] if col in df.columns]
    if check_cols:
        df = df.dropna(subset=check_cols).copy()
    else:
        df = df.copy()

    dropped_count = initial_count - len(df)

    # 검색어를 구분 컬럼으로 이름 변경
    if '검색어' in df.columns:
        df = df.rename(columns={'검색어': '구분'})

    # 이름 중복 컬럼 방지
    df = df.loc[:, ~df.columns.duplicated()]

    return df, dropped_count


# 3. 메인 애플리케이션 실행 함수
def main():
    st.set_page_config(layout="wide")

    # 좌우 2단 레이아웃 구성
    col_left, col_right = st.columns([1, 2])

    # 왼쪽 영역: 데이터 불러오기 및 안내
    with col_left:
        st.subheader("데이터 불러오기")
        st.write("로컬 환경에 저장된 CSV 파일을 불러옵니다. (API 호출 없음)")

        # 조회할 데이터 선택
        data_type = st.radio("조회할 데이터를 선택하세요.", ["전월세", "인프라"])
        load_btn = st.button("데이터 불러오기")

    # 오른쪽 영역: 결과 요약 및 테이블 표시
    with col_right:
        st.subheader("결과 요약 및 데이터")

        if load_btn:
            if data_type == "전월세":
                file_path = "rent_clean.csv"
                if os.path.exists(file_path):
                    raw_df = pd.read_csv(file_path)
                    rent_df, dropped = process_rent_data(raw_df)

                    st.write(f"결측치 처리 안내: {dropped}개의 행이 제외되었습니다.")

                    # 요약 지표 계산
                    total_count = len(rent_df)
                    avg_deposit = rent_df['보증금'].mean() if '보증금' in rent_df.columns and total_count > 0 else 0

                    # 월세 평균 (전세 제외)
                    if '거래유형' in rent_df.columns and '월세' in rent_df.columns:
                        monthly_df = rent_df[rent_df['거래유형'] == '월세']
                        avg_rent = monthly_df['월세'].mean() if not monthly_df.empty else 0
                    else:
                        avg_rent = 0

                    # 상단 요약 매트릭 (3열 가로 배치)
                    m1, m2, m3 = st.columns(3)
                    m1.metric("총 거래 수", f"{total_count}건")
                    m2.metric("평균 보증금", f"{avg_deposit:,.0f}만 원")
                    m3.metric("평균 월세(전세제외)", f"{avg_rent:,.0f}만 원")

                    # 하단 테이블 표시
                    display_cols = ['계약일', '동', '주택유형', '보증금', '월세', '면적', '건축년도', '거래유형']
                    final_cols = [c for c in display_cols if c in rent_df.columns]
                    st.dataframe(rent_df[final_cols])
                else:
                    st.warning(f"같은 폴더에 '{file_path}' 파일이 없습니다.")

            elif data_type == "인프라":
                file_path = "infra_clean.csv"
                if os.path.exists(file_path):
                    raw_df = pd.read_csv(file_path)
                    infra_df, dropped = process_infra_data(raw_df)

                    st.write(f"결측치 처리 안내: {dropped}개의 행이 제외되었습니다.")

                    # 요약 지표 계산
                    total_places = len(infra_df)
                    cvs_count = len(
                        infra_df[infra_df['구분'].str.contains('편의점', na=False)]) if '구분' in infra_df.columns else 0
                    laundry_count = len(
                        infra_df[infra_df['구분'].str.contains('세탁', na=False)]) if '구분' in infra_df.columns else 0

                    # 상단 요약 매트릭 (3열 가로 배치)
                    m1, m2, m3 = st.columns(3)
                    m1.metric("전체 장소 수", f"{total_places}곳")
                    m2.metric("편의점 수", f"{cvs_count}곳")
                    m3.metric("코인세탁소 수", f"{laundry_count}곳")

                    # 하단 테이블 표시
                    display_cols = ['구분', '장소명', '경도', '위도']
                    final_cols = [c for c in display_cols if c in infra_df.columns]
                    st.dataframe(infra_df[final_cols])
                else:
                    st.warning(f"같은 폴더에 '{file_path}' 파일이 없습니다.")


if __name__ == "__main__":
    main()
