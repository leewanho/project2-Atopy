# AtoCatch — 아토피 통합 진단 보조 시스템

세미나2 프로젝트 2팀. 이미지 + 문진 데이터 기반의 이중 모델 구조로 아토피 가능성을 평가합니다.

---

## 1. 프로젝트 개요

**AtoCatch**는 두 개의 독립 모델로 사용자의 아토피 위험도를 추정합니다.

| 모델 | 입력 | 출력 | 알고리즘 |
|---|---|---|---|
| **이미지 모델** | 피부 사진 | 5개 클래스 분류 (정상 / 아토피 / 건선 / 여드름 / 주사) | EfficientNetV2-S (PyTorch + timm) |
| **문진 모델** | 7개 binary 설문 응답 | 아토피 유무 (이진 분류) + 보정 확률 | LogReg / RandomForest / LightGBM / MLP 후보 비교 |

두 모델의 산출물을 통합해 사용자에게 종합 결과를 제공하는 구조입니다.

---

## 2. 디렉터리 구조

```
project2/
├── 1. 이미지 데이터 및 모델링/
│   └── 이미지 모델링/
│       ├── preprocess.py                          # AI Hub + DermNet 전처리 (300px 리사이즈)
│       ├── tune_mixed.py                          # Optuna 하이퍼파라미터 튜닝 (30 trials)
│       ├── train_mixed_tuned.py                   # 최종 모델 학습 (튜닝 결과 반영)
│       ├── evaluate_test_mixed.py                 # AI Hub / DermNet / 전체 test 평가
│       ├── model_comparison_mixed.py              # 모델 비교 실험 (10 epoch 빠른 평가)
│       ├── skin_disease_6class_side_dataset_for_ta (1).csv   # AI Hub 측면 데이터 메타
│       └── 이미지데이터_링크.txt                  # 원본 데이터셋 출처
│
├── 2. 문진용 아토피 설문 데이터/
│   ├── 데이터 및 전처리/
│   │   ├── hn222324_all.csv                       # 국민건강영양조사 원본 (28MB)
│   │   └── 아토피분석_전처리_모델링코드.py        # 전처리 + 초기 모델링
│   └── 모델링/
│       ├── 03_Hyperparameter_tune_cv.py           # 5-fold CV + RandomizedSearch / GridSearch
│       ├── 04_select_retrain_test.py              # Best 모델 선정 + 재학습 + 보정 + 부트스트랩 CI
│       └── preprocessed_for_ta.csv                # 전처리 완료 데이터
│
├── 세미2 프로젝트 기획서_2팀.docx                 # 프로젝트 기획서
├── 세미2 프로젝트 발표자료_2팀.pdf                # 최종 발표자료
└── README.md
```

---

## 3. 이미지 모델 파이프라인

### 데이터셋
- **AI Hub** — 한국인 피부질환 합성 이미지 (측면 뷰)
- **DermNet** — 실제 임상 이미지 ([Kaggle](https://www.kaggle.com/datasets/shubhamgoel27/dermnet))

### 실행 순서
```bash
cd "1. 이미지 데이터 및 모델링/이미지 모델링"

# 1) 전처리 (300x300 JPEG, 멀티스레드)
python preprocess.py

# 2) 하이퍼파라미터 탐색 (Optuna, 30 trial × 7 epoch)
python tune_mixed.py

# 3) 본 학습 (20 epoch)
python train_mixed_tuned.py

# 4) 평가 (AI Hub / DermNet / 전체 test)
python evaluate_test_mixed.py
```

### 주요 설정
- 백본: `EfficientNetV2-S` (timm)
- 이미지 크기: 300×300
- 손실: CrossEntropy
- 클래스: 5개 (정상, 아토피, 건선, 여드름, 주사)

> **주의**: 코드 내 경로(`E:\skin\...`)는 작성 PC 기준. 실행 시 본인 환경에 맞게 수정 필요.

---

## 4. 문진 모델 파이프라인

### 데이터
- **출처**: 국민건강영양조사 (KNHANES) 22~24차
- **타겟**: `DL1_dg` (아토피 진단 유무)
- **피처 (7개 binary)**:
  - `DJ8_dg_1.0` — 아토피 관련
  - `DJ4_dg_1.0` — 천식
  - `marri_1_2.0` — 미혼
  - `age_group_15-34` — 나이대
  - `town_t_2.0` — 사는곳
  - `BD1_11_6.0` — 음주(BD1_11=6)
  - `sm_presnt_1.0` — 흡연

### 실행 순서
```bash
cd "2. 문진용 아토피 설문 데이터"

# (선택) 원본부터 다시 → 전처리/모델링 통합 스크립트
python "데이터 및 전처리/아토피분석_전처리_모델링코드.py"

# 1) 5-fold CV로 모델 4종 × 불균형 처리 4종 튜닝
python 모델링/03_Hyperparameter_tune_cv.py

# 2) Best 모델 선정 + 재학습 + Isotonic 보정 + bootstrap CI
python 모델링/04_select_retrain_test.py
```

### 모델 후보
| 모델 | 불균형 처리 |
|---|---|
| Logistic Regression | none / class_weight / SMOTEN_0.5 / SMOTEN_1.0 |
| Random Forest | 동일 |
| LightGBM | 동일 |
| MLP (PyTorch) | 동일 |

### Best 모델 선정 기준 (tie-break)
1. `CV_ROC_AUC_mean` 최대
2. (동률 ±0.001) `CV_PR_AUC` 최대
3. (동률) `CV_Brier` 최소
4. (동률) 단순한 모델 우선 (LogReg < LightGBM < RF < MLP)

### 평가
- Test set: 15% 홀드아웃 (`test_split_indices.npz`로 split 재현)
- 메트릭: ROC_AUC, PR_AUC, Brier, Sens@95Spec
- 보정: Isotonic Regression (calib set에서 fit)
- 신뢰구간: 1000회 bootstrap

---

## 5. 시연 자료

### Demo 1 — 전체 흐름 (1분 24초)
![Demo 1](demo/demo1.gif)

> 원본 (음성 포함, 22MB): [`demo/demo1.mp4`](demo/demo1.mp4)

### Demo 2 — 추가 시연 (18초)
![Demo 2](demo/demo2.gif)

> 원본 (음성 포함, 1.9MB): [`demo/demo2.mp4`](demo/demo2.mp4)

### 기타 자료
- 발표자료 PDF: [`세미2 프로젝트 발표자료_2팀.pdf`](프로젝트 발표자료.pdf)
- 기획서: [`세미2 프로젝트 기획서_2팀.docx`](프로젝트 기획서.docx)


---


## 6. 의존성

### 공통
```
python>=3.9
numpy
pandas
scikit-learn
matplotlib
seaborn
```

### 이미지 모델
```
torch
torchvision
timm
pillow
optuna
tqdm
```

### 문진 모델
```
torch
lightgbm
imbalanced-learn
statsmodels
scipy
shap
plotly
```

---

## 7. 라이선스 / 데이터 사용

- AI Hub 데이터: AI Hub 이용약관에 따름
- DermNet (Kaggle): 데이터셋 라이선스에 따름
- KNHANES: 질병관리청 공공데이터 (학술 사용 가능)

코드 자체는 학술/교육 목적으로 작성됨.
