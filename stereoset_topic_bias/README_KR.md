# LLM의 암묵적 편향과 환각의 상관관계 (모호한 맥락 하에서)

> **NLP / AI Safety / LLM 평가**
> 입력 맥락이 모호할 때, LLM의 **암묵적 편향(implicit bias)** 경향이
> **환각(hallucination)** 발생과 상관이 있는가?

## 한눈에 보기

- **연구 질문:** 모호한 맥락에서 LLM의 편향성과 환각률은 상관이 있는가?
- **접근:**
  - **환각 유발(hallucination trigger) 데이터셋** 설계 — 모호한 상황에서
    편향 섞인 응답을 유도.
  - **Llama-3-8B-Instruct**(주 모델) + Gemini 계열로 생성 비교.
  - **StereoSet 스타일** 평가 파이프라인 (`llm-ss`) 사용.
  - 파일럿: **Gender × Major** 속성 쌍.
  - **RAG** 추가 — retrieval 기반 grounding이 bias-hallucination 관계를 바꾸는지.

## 폴더 구조

```
stereoset_topic_bias/
├── README.md / README_KR.md
├── .gitignore
├── notebooks/
│   ├── 01_correlation_bias_hallucination.ipynb   메인 실험 설계
│   ├── 02_llama3_generation.ipynb                Llama-3-8B 생성
│   ├── 03_rag_experiment.ipynb                   RAG 증강 생성
│   ├── 04_big_dataset_processing.ipynb           대규모 데이터 처리
│   └── 05_llm_ss_evaluation.ipynb                StereoSet 스타일 평가
└── docs/
    ├── pilot_experiment_gender_major_en.pdf
    └── pilot_experiment_gender_major_kr.pdf
```

## 방법론

### 파일럿 실험 — Gender × Major
**성별 × 전공** 속성을 교차한 소규모 실험. 모호한 질의에서 고정관념과
정렬된 출력이 나오는지 측정. 본 실험의 토대.

### 환각 유발 세트
의도적으로 핵심 정보가 빠진 프롬프트로 모델이 (a) 거부 / (b) 보정된 응답 /
(c) 작화(confabulation) 중 어느 것을 하는지. (c) 비율과 고정관념 정렬 출력
비율의 상관을 측정.

### LLM-SS 평가
StereoSet 스타일로 stereotypical vs anti-stereotypical 연속에 대한 모델 선호도
측정. seed 고정, BF16 추론.

### 사용 모델
- **`meta-llama/Meta-Llama-3-8B-Instruct`** (주)
- 비교용 생성 모델 (노트북 02 참고)

## 사용 기술

`Python` · `PyTorch` · `transformers` · `accelerate` · `sentencepiece` ·
`scipy` · `huggingface_hub`

## 실행 방법

```bash
# 1. 라이브러리 설치
pip install transformers torch pandas numpy sentencepiece accelerate \
            huggingface_hub scipy

# 2. HuggingFace 토큰은 환경변수로만 (절대 하드코딩 X)
export HF_TOKEN=your_hf_token_here

# 3. 노트북 순서대로
jupyter notebook notebooks/01_correlation_bias_hallucination.ipynb
```

### 보안 안내

모든 노트북은 `os.environ.get("HF_TOKEN")` 으로 토큰을 읽도록 정리되어 있음.
`.env`나 하드코딩된 토큰은 절대 커밋 X. `.gitignore` 참고.

## 의의

- **AI Safety / Responsible AI**: LLM의 두 대표 실패 모드인 *편향*과 *환각*은
  보통 따로 연구되는데, 이 둘이 **모호성 하 모델 불확실성**이라는 공통 기원을
  공유하는지 가설.
- **실용적 함의**: 상관이 있다면 **calibration 기반** 완화(거부·retrieval grounding)로
  둘을 동시에 줄일 수 있을 가능성.
- **방법론**: 표준 LLM 평가(StereoSet 스타일) + 자체 환각 트리거 결합으로
  상관 측정 가능.

## 참고

- StereoSet (Nadeem et al., 2020)
- Llama 3 (Meta AI, 2024)
- `llm-ss` — 본 프로젝트에서 사용한 StereoSet-style 평가 도구

---

*작성: 이선재 (Sunjae Lee) · 고려대 세종 빅데이터전공*
*[NLP](https://github.com/Vongole25/NLP) repo. 2027 전기 NLP 대학원 진학 준비 중.*
