이 프로젝트는 Stack Overflow 개발자 설문 데이터를 이용하여
AI 사용 강도(AI_Index) 와 연봉의 관계를 분석하는 Python 코드입니다.

1. 필수 환경

Python 3.8 이상

필수 패키지

pandas

numpy

matplotlib

statsmodels

2. 프로젝트 폴더 구조

GitHub 저장소를 클론한 후, 기본 폴더 구조는 다음과 같다.

HomeworkRepository/
 ├─ analysis.py
 ├─ data_prep.py
 ├─ config.py
 ├─ data.zip          # 설문 원본 데이터(압축 파일)
 └─ output/           # 분석 결과(그래프, CSV)가 저장되는 폴더

3. 데이터 압축 해제

GitHub 용량 제한 때문에 원본 CSV 파일은 data.zip 으로 압축되어 있습니다.
분석을 실행하기 전에 반드시 다음과 같이 압축을 해제해야 합니다.

unzip data.zip -d data

압축 해제 후 폴더 구조는 다음과 같다.

HomeworkRepository/
 ├─ data/
 │   ├─ survey_results_public.csv
 │   └─ survey_results_schema.csv
 └─ ...
config.py 에서는 DATA_DIR = "data", OUTPUT_DIR = "output" 으로 설정되어 있으므로
위와 같은 구조만 맞으면 추가 수정 없이 실행할 수 있습니다.

4. 프로그램 실행 방법

모든 준비가 끝난 후, 프로젝트 루트에서 다음 명령을 실행하면
전체 분석이 한 번에 수행된다.

python analysis.py

data_prep.py 가 자동으로 호출되어 원본 데이터를 정제하고,

analysis.py 가 Part 1 ~ Part 5 전체 분석을 수행한 뒤
모든 그래프와 요약 통계 파일을 output/ 폴더에 저장한다.
