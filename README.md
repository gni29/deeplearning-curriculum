# 🧠 딥러닝 심화 커리큘럼 (3개월) - 산업 데이터사이언티스트 되기

> **목표**: 확률론부터 Transformer까지, 이론과 실무를 겸비한 산업 데이터사이언티스트로 성장  
> **기간**: 2025.05.28 ~ 2025.08.28 (84일)  
> **수준**: 딥러닝 기초 → 중급 → 실무 적용

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?&logo=Jupyter&logoColor=white)](https://jupyter.org/)

## 📊 현재 진행률

```
전체 진행률: ▓░░░░░░░░░ 0% (0/84일)

📅 1개월차: 확률론 & CNN & 생성모델
├── Week 1: 확률론 기초 ░░░░░░░ 0/7일
├── Week 2: 고급 CNN & 객체탐지 ░░░░░░░ 0/7일  
├── Week 3: VAE 완전정복 ░░░░░░░ 0/7일
└── Week 4: GAN & 고급 생성모델 ░░░░░░░ 0/7일

📅 2개월차: NLP & Transformer
├── Week 5: RNN & Attention ░░░░░░░ 0/7일
├── Week 6: Seq2Seq & Attention ░░░░░░░ 0/7일
├── Week 7: Transformer 분석 ░░░░░░░ 0/7일
└── Week 8: BERT & GPT ░░░░░░░ 0/7일

📅 3개월차: 고급주제 & 실무프로젝트  
├── Week 9: 멀티모달 & 최신기법 ░░░░░░░ 0/7일
├── Week 10: 모델최적화 & 배포 ░░░░░░░ 0/7일
├── Week 11: 종합프로젝트 1 ░░░░░░░ 0/7일
└── Week 12: 종합프로젝트 2 ░░░░░░░ 0/7일
```

## 🎯 학습 목표 및 성과

### 🏆 최종 목표
- [ ] **확률론 마스터**: KL Divergence, 베이즈 추론, 변분 추론
- [ ] **CNN 고급**: ResNet, EfficientNet, YOLO, U-Net 직접 구현
- [ ] **생성모델 전문가**: VAE, GAN, Diffusion Model 이론 & 구현
- [ ] **NLP & Transformer**: BERT, GPT, T5 파인튜닝 및 활용
- [ ] **실무 역량**: 모델 최적화, 배포, MLOps 기초
- [ ] **포트폴리오**: 3개 이상의 완성된 프로젝트

### 📈 습득 예정 기술 스택
**이론**: 확률론, 정보이론, 베이즈 추론, 변분 추론  
**프레임워크**: PyTorch, Hugging Face Transformers, FastAPI  
**도구**: Docker, Git, Jupyter, Weights & Biases  
**배포**: AWS/GCP, Docker, API 서버 구축

## 🗂️ 저장소 구조

```
📁 deeplearning-curriculum/
├── 📘 README.md                    # 이 파일
├── 📂 daily-logs/                  # 일일 학습 기록
│   ├── 📁 week01/ (확률론 기초)
│   ├── 📁 week02/ (고급 CNN)
│   └── ...
├── 📂 theory-notes/               # 개념별 이론 정리
│   ├── 📁 probability/            # 확률론 완전정복
│   ├── 📁 deep-learning/          # 딥러닝 고급 개념
│   └── 📁 nlp/                    # NLP & Transformer
├── 📂 code-implementations/       # 직접 구현한 모든 코드
│   ├── 📁 week01-probability/     # 확률론 구현
│   ├── 📁 week02-cnn/            # CNN 모델들
│   └── ...
├── 📂 projects/                   # 주요 프로젝트
│   ├── 📁 month1-vae-project/     # VAE 얼굴 생성
│   ├── 📁 month2-transformer/     # 기계번역 시스템
│   └── 📁 month3-multimodal/      # 이미지 캡셔닝
├── 📂 resources/                  # 학습 자료 모음
│   ├── 📁 papers/                 # 논문 요약
│   └── 📄 useful-links.md         # 유용한 링크
└── 📂 portfolio/                  # 포트폴리오 정리
    ├── 📄 project-summaries.md    # 프로젝트 요약
    └── 📄 skills-acquired.md      # 습득한 기술
```

## 🚀 주요 프로젝트 미리보기

### 1️⃣ 1개월차: VAE 얼굴 생성 프로젝트
**목표**: 확률론 기반 VAE로 CelebA 얼굴 이미지 생성  
**핵심 기술**: KL Divergence, ELBO, Reparameterization Trick  
**예상 결과**: 새로운 얼굴 생성 + 잠재공간 탐색

### 2️⃣ 2개월차: Transformer 기계번역 시스템
**목표**: Transformer를 처음부터 구현하여 영→한 번역  
**핵심 기술**: Self-Attention, Multi-Head Attention, Positional Encoding  
**예상 결과**: 실용적인 번역 성능 + Attention 시각화

### 3️⃣ 3개월차: 멀티모달 이미지 캡셔닝
**목표**: CNN + Transformer로 이미지 설명 생성  
**핵심 기술**: Vision Transformer, Cross-Attention, CLIP  
**예상 결과**: 배포 가능한 웹 서비스

## 📚 학습 철학 및 방법론

### 🎯 핵심 원칙
1. **이론 먼저, 구현으로 확인**: 수식 → 코드 → 실험
2. **라이브러리 의존 최소화**: 핵심 알고리즘은 직접 구현
3. **실무 관점 유지**: 모든 프로젝트를 서비스 수준으로
4. **체계적 기록**: 일일 학습 일지 + 개념 정리

### 📖 학습 사이클
```
📝 이론 학습 → 💻 직접 구현 → 🧪 실험 및 검증 → 📊 결과 분석 → 📖 복습 및 정리
```

### 🔍 평가 기준
- **이해도**: 개념을 남에게 설명할 수 있는가?
- **구현 능력**: 라이브러리 없이 직접 구현 가능한가?
- **응용 능력**: 새로운 문제에 적용할 수 있는가?
- **실무 적용**: 실제 서비스에 배포 가능한가?

## 🛠️ 개발 환경 및 도구

### 💻 하드웨어 요구사항
- **최소**: Google Colab (무료 GPU)
- **권장**: RTX 3060 이상 또는 클라우드 GPU

### 🐍 소프트웨어 스택
```python
# 주요 라이브러리
torch>=1.12.0
transformers>=4.20.0
numpy>=1.21.0
matplotlib>=3.5.0
jupyter>=1.0.0
fastapi>=0.80.0
docker>=6.0.0
```

### 🔧 개발 도구
- **코드 에디터**: VS Code + Python Extension
- **실험 관리**: Weights & Biases
- **버전 관리**: Git + GitHub
- **배포**: Docker + FastAPI

## 📈 일일 학습 루틴

### ⏰ 평일 스케줄 (4-5시간)
```
🌅 오전 (2시간)
├── 30분: 이론 복습 (어제 내용)
├── 60분: 새로운 개념 학습
└── 30분: 논문 읽기 또는 수식 유도

🌆 오후 (2-3시간)  
├── 90분: 실습 및 코딩
├── 60분: 프로젝트 작업
└── 30분: 학습 일지 작성 + Git 커밋
```
### 🎯 주말 스케줄
- **토요일**: 프로젝트 집중 작업 (4시간)
- **일요일**: 복습 + 다음 주 계획 (2시간)

## 📊 성과 측정 지표

### 📈 정량적 지표
- [ ] **커밋 횟수**: 84일 동안 매일 커밋 (목표: 84회 이상)
- [ ] **구현 완료율**: 주요 모델 직접 구현 (목표: 15개 이상)
- [ ] **프로젝트 완성도**: 배포 가능한 프로젝트 (목표: 3개)
- [ ] **블로그 포스팅**: 학습 내용 정리 (목표: 12편)

### 🎯 정성적 지표
- [ ] **개념 이해도**: 면접관에게 설명 가능한 수준
- [ ] **문제 해결 능력**: 새로운 데이터셋에 모델 적용
- [ ] **코드 품질**: 가독성 좋고 재사용 가능한 코드
- [ ] **실무 적용성**: 실제 비즈니스 문제 해결 가능

## 🏆 마일스톤 및 체크포인트

### 🎯 월별 주요 성과
| 월차 | 주요 성과 | 검증 방법 |
|------|-----------|-----------|
| 1개월 | 확률론 마스터 + VAE 구현 | 개념 설명 + 코드 리뷰 |
| 2개월 | Transformer 완전 구현 | 번역 성능 + 시각화 |
| 3개월 | 실무 프로젝트 완성 | 배포 + 포트폴리오 |

### 📅 주간 체크포인트
매주 일요일마다:
- [ ] 이번 주 학습 목표 달성도 평가
- [ ] 어려웠던 점과 해결 방법 정리
- [ ] 다음 주 계획 수립 및 우선순위 설정

## 📝 기여 및 피드백

이 저장소는 개인 학습 기록이지만, 다른 분들께도 도움이 되기를 바랍니다!

### 🎯 기여 방법
- **⭐ Star**: 프로젝트가 도움이 되었다면 Star를 눌러주세요
- **🐛 Issue**: 오류나 개선사항을 발견하시면 Issue로 알려주세요
- **💡 Suggestion**: 더 좋은 학습 방법이나 자료가 있다면 공유해주세요

### 📧 연락처
- **Email**: gni29@daum.net

---

## 📜 라이센스

이 프로젝트는 MIT License 하에 배포됩니다. 자유롭게 사용하되, 출처를 명시해주세요.

---

<div align="center">

**🚀 함께 성장하는 딥러닝 여정에 오신 것을 환영합니다! 🚀**

*"Every expert was once a beginner. Every pro was once an amateur."*

**시작일**: 2025년 5월 28일  
**현재**: Day 0 - 준비 단계  
**목표**: 84일 후 산업 데이터사이언티스트로 성장

</div>
