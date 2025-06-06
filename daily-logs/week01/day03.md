markdown# Day 03 - [오늘의 주제]

**날짜**: 2025-05-30  
**학습 시간**: 6시간 00분  
**주요 주제**:


## 📚 오늘 배운 내용

### 이론 학습
- **개념 1**: KL-Divergence
- **개념 2**: ELBO(Evidence Lower Bound)
- **개념 3**: 정보이론의 본질
- **개념 4**: VAE Loss 이해
- **개념 5**: VAE의 동작 원리


### 실습 완료
- **cond-implementations/week01/day03/**: 실습 내용


## 💡 핵심 깨달음

- 정보 이론에서의 정보량이란 희귀한 사건일수록 정보량이 많고 일반적인 사건일수록 정보량이 적다.
- VAE의 학습 과정과 수리적, 통계적 근거

### 정보량에서부터 VAE Loss 까지 이어지는 수리적 흐름 

```
I(x) = -log P(x)                    [정보량]
     ↓ (평균)
H(X) = E[I(X)]                      [엔트로피]
     ↓ (비교)  
D_KL(P||Q) = E_P[log(P/Q)]          [KL Divergence]
     ↓ (하한)
ELBO = E_q[log p(x|z)] - D_KL(...)  [ELBO]
     ↓ (최적화)
min(-ELBO)                          [VAE Loss]
```

## 🤔 어려웠던 점

### 문제 1: [구체적 문제]
- **상황**: ELBO 의 Jensen's Inequality 에 대한 증명
- **원인**: 증명 과정에서 헷갈리는 부분이 발생
- **해결**: E(f(x)) 계산식을 다시 복기하여 이해 완료
- **참고**: 도움된 자료/링크


## 📊 일일 성과

**이론 이해도**: 6/10  
**실습 완성도**: 5/10  
**전체 만족도**: 5/10  

**총평**: KL-Divergency와 ELBO의 관계에 대한 이해를 통해 VAE의 오차 계산의 근거를 이해함.