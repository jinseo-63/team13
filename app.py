import os
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime
from dotenv import load_dotenv

# ==========================================
# 1. API 키 설정
# ==========================================
# 방법 A: .env 파일에서 불러오기

# 방법 B: .env 파일 사용이 계속 안 된다면 위 두 줄을 지우고 아래 줄의 주석(#)을 해제하여 직접 입력하세요.
api_key = "여기에_본인의_공공데이터포털_일반인증키(Decoding)_입력"

if not api_key:
    raise ValueError("API 키를 찾을 수 없습니다. .env 파일에 MOLIT_API_KEY를 설정하거나 코드에 직접 입력해주세요.")


# ==========================================
# 2. 실행일 기준 최근 3개월(계약년월) 동적 생성
# ==========================================
def get_recent_3_months():
    today = datetime.now()
    ymd_list = []
    for i in range(3):
        month = today.month - i
        year = today.year
        if month <= 0:
            month += 12
            year -= 1
        ymd_list.append(f"{year}{month:02d}")
    return ymd_list


# ==========================================
# 3. 국토교통부 실거래가 데이터 수집 함수 (오류 무시 로직 적용)
# ==========================================
def get_molit_rent_data(api_key, lawd_cd, deal_ymd, housing_type="단독다가구"):
    if housing_type == "단독다가구":
        url = "http://apis.data.go.kr/1613000/RTMSDataSvcSHRent/getRTMSDataSvcSHRent"
    else:
        url = "http://apis.data.go.kr/1613000/RTMSDataSvcRHRent/getRTMSDataSvcRHRent"

    params = {
        'serviceKey': api_key,
        'LAWD_CD': lawd_cd,
        'DEAL_YMD': deal_ymd
    }

    try:
        response = requests.get(url, params=params)
        root = ET.fromstring(response.text)
    except ET.ParseError:
        print(f"[{housing_type}] XML 데이터를 읽을 수 없습니다. (API 서버 응답 지연)")
        return pd.DataFrame()

    # 인증 에러 체크
    if root.tag == 'OpenAPI_ServiceResponse':
        err_msg = root.find('.//returnAuthMsg')
        if err_msg is not None:
            print(f"[{housing_type}] API 인증 오류: {err_msg.text}")
            return pd.DataFrame()

    # 에러 조건 무시하고 실제 데이터(item)부터 바로 찾기
    items = root.findall('.//item')

    data_list = []
    for item in items:
        row = {}
        for child in item:
            row[child.tag] = child.text.strip() if child.text else ""
        row['주택유형'] = housing_type
        row['조회년월'] = deal_ymd
        data_list.append(row)

    # 만약 해당 월에 거래 데이터가 비어있다면 안내
    if not data_list:
        result_msg = root.find('.//resultMsg')
        msg_text = result_msg.text if result_msg is not None else ""
        # 정상적인 "OK" 응답인데 거래내역만 없는 거라면 굳이 에러메시지를 띄우지 않음
        if "OK" not in msg_text.upper():
            print(f"[{housing_type}] API 안내 메시지: {msg_text}")

    return pd.DataFrame(data_list)


# ==========================================
# 4. 데이터 수집 실행 및 CSV 저장
# ==========================================
if __name__ == "__main__":
    target_lawd_cd = "26410"  # 부산 금정구
    deal_ymds = get_recent_3_months()

    print(f"부산 금정구({target_lawd_cd}) 최근 3개월 조회 대상: {deal_ymds}")

    all_data_frames = []

    for ymd in deal_ymds:
        print(f"[{ymd}] 실거래가 데이터 수집 중...")
        df_single = get_molit_rent_data(api_key, target_lawd_cd, ymd, "단독다가구")
        df_town = get_molit_rent_data(api_key, target_lawd_cd, ymd, "연립다세대")

        if not df_single.empty:
            all_data_frames.append(df_single)
        if not df_town.empty:
            all_data_frames.append(df_town)

    # ==========================================
    # 5. 데이터 병합 후 CSV 저장
    # ==========================================
    if all_data_frames:
        final_df = pd.concat(all_data_frames, ignore_index=True)

        # 엑셀에서 한글이 깨지지 않도록 utf-8-sig 인코딩 지정
        csv_filename = "rent_clean.csv"
        final_df.to_csv(csv_filename, index=False, encoding="utf-8-sig")

        print(f"\n데이터 수집 완료! 총 {len(final_df)}건의 데이터가 '{csv_filename}'로 저장되었습니다.")

        # 수집된 데이터 형태 미리보기
        print("\n--- 데이터 미리보기 (상위 3건) ---")
        # 컬럼이 존재하는지 확인 후 출력
        display_cols = ['조회년월', '주택유형', '법정동', '지번', '보증금액', '월세금액', '전용면적', '계약면적']
        actual_cols = [col for col in display_cols if col in final_df.columns]
        print(final_df[actual_cols].head(3))
    else:
        print("조건에 맞는 실거래가 데이터가 존재하지 않거나 수집에 실패했습니다.")

import os
import requests
import pandas as pd
from dotenv import load_dotenv

# ==========================================
# 1. 카카오 API 키 설정
# ==========================================
# 방법 A: .env 파일에서 불러오기 (KAKAO_API_KEY=본인의_REST_API_키)

# 방법 B: 환경변수 설정이 번거롭다면 아래 주석을 풀고 키를 직접 입력하세요.
kakao_api_key = "여기에_카카오_REST_API_키_입력"

if not kakao_api_key:
    raise ValueError("카카오 API 키를 찾을 수 없습니다. 키를 설정해주세요.")


# ==========================================
# 2. 카카오 키워드 장소 검색 함수 (페이징 처리 포함)
# ==========================================
def get_kakao_places(api_key, keyword):
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    headers = {
        "Authorization": f"KakaoAK {api_key}"
    }

    all_data = []

    # 카카오 장소 검색은 1~3페이지(최대 45개)까지 제공됩니다.
    for page in range(1, 4):
        params = {
            "query": keyword,
            "page": page,
            "size": 15
        }

        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            result = response.json()
            documents = result.get('documents', [])

            if not documents:
                break

            for doc in documents:
                all_data.append({
                    '검색어': keyword,
                    '장소명': doc.get('place_name'),
                    '카테고리': doc.get('category_name'),
                    '지번주소': doc.get('address_name'),
                    '도로명주소': doc.get('road_address_name'),
                    '경도(X)': doc.get('x'),
                    '위도(Y)': doc.get('y'),
                    '장소URL': doc.get('place_url')
                })

            # 마지막 페이지인지 확인 (is_end가 True면 더 이상 데이터 없음)
            meta = result.get('meta', {})
            if meta.get('is_end'):
                break
        else:
            print(f"[{keyword}] API 요청 오류: {response.status_code}")
            break

    return pd.DataFrame(all_data)


# ==========================================
# 3. 데이터 수집 실행 및 CSV 저장
# ==========================================
if __name__ == "__main__":
    # 수집할 타겟 키워드 목록
    keywords = ["부산대 편의점", "부산대 코인세탁소", "부산대 대형마트", "부산대 지하철역"]

    all_infra_frames = []

    for kw in keywords:
        print(f"'{kw}' 데이터 수집 중...")
        df_places = get_kakao_places(kakao_api_key, kw)

        if not df_places.empty:
            print(f" -> {len(df_places)}건 수집 완료")
            all_infra_frames.append(df_places)
        else:
            print(f" -> 수집된 데이터가 없습니다.")

    # 4. 데이터 병합 후 CSV 저장
    if all_infra_frames:
        final_df = pd.concat(all_infra_frames, ignore_index=True)

        # 엑셀에서 한글 깨짐을 방지하기 위해 utf-8-sig로 저장
        csv_filename = "infra_clean.csv"
        final_df.to_csv(csv_filename, index=False, encoding="utf-8-sig")

        print(f"\n데이터 수집 성공! 총 {len(final_df)}건의 데이터가 '{csv_filename}'로 저장되었습니다.")
        print("\n--- 수집된 데이터 미리보기 (상위 5건) ---")
        print(final_df[['검색어', '장소명', '경도(X)', '위도(Y)']].head())
    else:
        print("조건에 맞는 인프라 데이터가 수집되지 않았습니다.")

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
