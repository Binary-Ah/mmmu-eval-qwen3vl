# Qwen3-VL-4B MMMU-val Evaluation Pipeline & Ablations

Qwen3-VL-4B-Instruct를 MMMU validation 900문항에 **공식 generation recipe 그대로** 돌려 61.11%를 얻고,
공식 수치 67.4와의 격차(−6.29)를 finish_reason·토큰 길이·문항 유형·난이도·이미지 유형별로 분해한 평가 파이프라인입니다.
이후 같은 파이프라인으로 2건의 프롬프트 ablation을 수행하고 McNemar 검정으로 비교했습니다.

> 이 저장소는 2026-2학기 멀티모달딥러닝(MMDL) 수업 팀 프로젝트
> [Chaeeun4/MMDL](https://github.com/Chaeeun4/MMDL) (team4: 박채은·류다연·이진아·정성원)에서
> **이진아가 담당한 부분(`jina` 브랜치 산출물)** 만 떼어낸 것입니다.
> 팀 레포에는 tar.gz로 올라가 있던 파일을 풀어서 정리했습니다. 팀 제출본(58.56%)의 초기 파이프라인은 팀 레포를 참고하세요.

## Problem

- 목표: 단일 24 GB GPU에서 Qwen3-VL-4B-Instruct의 공식 MMMU-val 점수(67.4)를 재현하고, 격차가 어디서 생기는지 분해한다.
- 팀의 첫 제출본은 시간 절약을 위해 `max_new_tokens=4096`·축소 해상도(256·32²~1280·32² px)로 돌려 58.56%였고, 900문항 중 124문항이 4096 토큰에서 잘렸다(그중 오답 92).
- 그래서 생성 설정 전체를 Qwen 공식 Instruct recipe(32768 토큰, 체크포인트 기본 해상도)로 되돌린 baseline을 다시 확보하고, 이 run을 이후 파인튜닝 실험의 기준점으로 쓴다.

## Data

| 항목 | 값 |
|---|---|
| 데이터셋 | `MMMU/MMMU` validation split, 900문항 (30과목 × 30), revision `98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68` |
| 모델 | `Qwen/Qwen3-VL-4B-Instruct`, revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`, bf16, 양자화 없음 |
| 문항 구성 | multiple-choice 847 / open 53, Easy 295 / Medium 424 / Hard 181 |
| Judge 모델 | 미사용 (결정적 파서로 채점) |

## Method

- **추론**: vLLM 0.19.1 (continuous batching), 과목별 `generate()` 호출 → 과목별 소요 시간 측정 + 체크포인트/재개(`--resume`).
- **프롬프트**: MMMU 공식 `llava1.5.yaml` 템플릿 + Qwen3-VL chat template. 시스템 프롬프트 없음.
- **파서**: 공식 `eval_utils.py` 기반. 파싱 실패 시 공식 코드의 랜덤 선택 대신 `None`(오답)을 반환해 재현성 확보.
- **생성 설정** (공식 Instruct recipe, manifest에서 `matches_official_instruct_recipe == true`로 자동 검증):
  seed 3407 / temperature 0.7 / top_p 0.8 / top_k 20 / repetition_penalty 1.0 / presence_penalty 1.5 / max_new_tokens 32768 / min_pixels·max_pixels = 체크포인트 processor 기본값(65536 / 16777216 px).
- **지표**: 과목별 accuracy, Overall = 30과목 macro 평균(과목당 30문항이므로 micro와 동일).
- **기록**: `manifest.json`(모델·데이터 revision, 환경, git hash, 프롬프트 예시), `metrics.json`, `predictions.jsonl`(finish_reason·토큰 수·logprob 요약·이미지 해상도 포함), `gpu_monitor.csv`(nvidia-smi 0.5 s 폴링), `pip_freeze.txt`.
- **Ablation 비교**: 같은 문항에 대해 baseline과 run의 정답 전이(W→C / C→W)를 세고 McNemar 검정(`experiments/compare_two.md`).

## Results

### Baseline (`experiments/baseline_modelcard_v1`)

| 설정 | Overall (MMMU-val) | 비고 |
|---|---|---|
| 공식 (Qwen3-VL Technical Report) | 67.4 | |
| 팀 제출본 (4096 tok, 축소 해상도, RTX 4090) | 58.56 | 124/900 length stop, 그중 오답 92 — 팀 레포 `reports/mmmu_baseline.md` |
| **공식 recipe 재현 (32768 tok, 기본 해상도, RTX 3090)** | **61.11** | Δ −6.29, 40/900 length stop, 그중 오답 31, parse 실패 3 |

- 환경: RTX 3090 24 GB (host adsl-ubuntu), peak VRAM 22.41 GB, 총 13,235 s (3.68 h; 추론 13,083 s + 모델 로드 151 s), 실행 2026-10-02 13:59~17:40 KST.
- 문항 유형: multiple-choice 63.87% (n=847) vs open 16.98% (n=53).
- 난이도: Easy 70.17 / Medium 61.32 / Hard 45.86.
- 이미지 유형: Tables 73.53 (n=170), Diagrams 60.54 (n=223), Chemical Structures 26.67 (n=30), Technical Blueprints 28.57 (n=14), Sheet Music 33.33 (n=30).
- 과목: 최고 Marketing·Public_Health 90.0, 최저 Chemistry 30.0.
- completion tokens: mean 2,384 / median 8 / p99 32,571. 오답의 mean 토큰 3,932 vs 정답 1,400.

### Ablations (`experiments/abl_cot_open`, `experiments/p3_sym_cot`)

| run | 대상 | n | baseline | run | Δ (%p) | W→C / C→W | McNemar p |
|---|---|---|---|---|---|---|---|
| `abl_cot_open` | open 문항 전체, CoT 프롬프트 | 53 | 17.0% | 39.6% | +22.6 | 15 / 3 | 0.0075 |
| `p3_sym_cot` | 기호 이미지 4종(Sheet Music, Chemical Structures, Sketches and Drafts, Technical Blueprints), CoT 프롬프트 | 95 | 32.6% | 40.0% | +7.4 | 16 / 9 | 0.23 (유의하지 않음) |

- open 문항은 "단어 하나로 답하라"는 공식 프롬프트 탓에 계산 과정을 생략하는 경향이 있었고, CoT 허용 시 +22.6%p로 유의하게 올랐다.
- 기호 이미지 문항은 CoT만으로는 유의한 개선이 없어 perception 쪽 원인을 의심하고 후속 실험 p4~p6을 돌렸다.

### 후속 실험 (`experiments/proposal_table.md`, 상세 해석은 `docs/qwen3vl_experiments.md`)

| run | 설정 | n | baseline | run | Δ (%p) | W→C / C→W | McNemar p |
|---|---|---|---|---|---|---|---|
| `p1_noise_seed1234` | seed만 변경 (노이즈 기준선) | 900 | 61.1 | 60.1 | −1.0 | 25 / 34 | 0.298 |
| `abl_temp0` | temperature=0 | 900 | 61.1 | 58.6 | −2.6 | 21 / 44 | 0.0059 |
| `p4_sym_textonly` | 기호 이미지 95문항, 이미지 제거 | 95 | 32.6 | 24.2 | −8.4 | 7 / 15 | 0.134 |
| `p5_sym_hires` | 같은 95문항, 이미지당 ≥1024 visual tokens | 95 | 32.6 | 37.9 | +5.3 | 11 / 6 | 0.332 |
| `p6_sym_describe` | 같은 95문항, "이미지 먼저 옮겨 적기" 프롬프트 | 95 | 32.6 | 45.3 | +12.6 | 22 / 10 | 0.050 |
| `lora_smoke_eval` | LoRA 30 step 병합 체크포인트, 2과목 smoke | 60 | 68.3 | 46.7 | −21.7 | 4 / 17 | 0.0072 |

- seed만 바꿔도 900문항 중 59문항(6.6%)의 정오가 뒤집힌다 → 그 이하의 차이는 노이즈로 판정.
- temperature=0(greedy)은 반복·길이 초과가 늘어 오히려 −2.6%p (유의).
- 기호 이미지는 이미지를 빼면 떨어지고(−8.4), 해상도만 올려서는 유의하지 않으며, describe-first 프롬프트가 가장 컸다(+12.6, p=0.050 경계).
- LoRA smoke(30 step)는 파이프라인 검증용이며 성능은 크게 떨어졌다 — 본 학습 전 출력 형식 붕괴를 먼저 확인할 필요.
- 대표 오답/개선 사례와 이미지 유형별 표는 `evidence/`에 정리했다.

## My role

- `run_mmmu.py` 확장 (432 → 868줄): `--run_dir` 기반 산출물 구조, 과목별 체크포인트/재개, `manifest.json`(revision·환경·git hash·프롬프트 예시), `--logprobs` 요약, 공식 recipe 자동 검증, `auto` 해상도 해석.
- `summarize_run.py` 신규: run 디렉터리 → 과목별 표 + finish_reason/토큰 통계 + 문항유형·난이도·이미지유형 분해 + 두 run 비교표(`--compare`).
- 실행 스크립트 신규: `check_gpu.sh`, `run_baseline_modelcard.sh`(nohup 백그라운드, 로그·GPU 시계열 포함), `status_run.sh`, `gpu_monitor.sh`.
- 공식 recipe baseline 전체 run(900문항, 3.68 h) 설계·실행 및 runbook 문서화.
- 격차 분석과 ablation 2건 + McNemar 비교표.
- 후속: `--prompt_variant`/`--question_types`/`--img_types` 옵션과 프롬프트 변형(`prompt.py`), 실험 큐(`run_proposal_experiments.sh`), seed 노이즈·temperature·이미지 제거·고해상도·describe-first 실험 p1~p6.
- `compare_runs.py`(같은 문항 비교표 + McNemar), `analyze_subject_errors.py`, `build_evidence_pack.py`(사례·메타데이터 묶음), `lora_smoke_train.py`·`merge_lora.py`(LoRA smoke 학습·병합).
- `prompt.py`·`parser.py`·`evaluator.py`·`dataset.py`와 `analyze_*.py`는 팀 공통 코드(주로 박채은)이며, 파이프라인이 돌아가도록 함께 포함했습니다. `dataset.py`에는 MMMU `explanation` 필드를 오답 분석용으로 보존하는 3줄을 추가했습니다.
- 팀원별 상세 역할: TODO: 확인 (류다연·정성원 담당 범위).

## How to run

```bash
# 0. 환경 (vllm 0.19.1 / torch 2.10 / transformers 5.18 조합이 검증됨)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt      # 실패하면 pip install -r requirements.txt
export HF_HOME=/path/to/hf_cache          # 모델·데이터 캐시 위치

# 1. GPU 확인 → 2. 2과목 smoke → 3. 본 실행(백그라운드) → 4. 진행 확인 → 5. 요약
bash scripts/check_gpu.sh
bash scripts/run_baseline_modelcard.sh --smoke --gpu 0
bash scripts/run_baseline_modelcard.sh --gpu 0 --run_name baseline_modelcard_v1
bash scripts/status_run.sh runs/baseline_modelcard_v1
python code/summarize_run.py runs/baseline_modelcard_v1

# 직접 호출 (run_baseline_modelcard.sh가 내부에서 실행하는 커맨드)
python code/run_mmmu.py --model_path Qwen/Qwen3-VL-4B-Instruct --data_root $HF_HOME \
  --run_dir runs/baseline_modelcard_v1 --batch_mode subject --logprobs 5 \
  --seed 3407 --temperature 0.7 --top_p 0.8 --top_k 20 --repetition_penalty 1.0 \
  --presence_penalty 1.5 --max_new_tokens 32768 --min_pixels auto --max_pixels auto \
  --max_model_len 32768 --gpu_memory_utilization 0.90
```

산출물: `runs/<run_name>/{manifest.json, metrics.json, predictions.jsonl, per_subject/*.jsonl, summary.md, summary.json, gpu_monitor.csv, run.log}`.

ablation은 같은 `run_mmmu.py`에 `--prompt_variant`(cot_open / sym_cot / sym_describe 등), `--question_types`, `--img_types`로 대상 문항과 프롬프트를 고르고,
`scripts/run_proposal_experiments.sh`가 p1~p6 큐를 순서대로 실행합니다. 비교표는 `python code/compare_runs.py`로 다시 만들 수 있습니다.

## Files

```
code/
  run_mmmu.py                 # 평가 파이프라인 (과목별 체크포인트/재개, manifest, logprobs)
  summarize_run.py            # run 디렉터리 → summary.md / summary.json, 두 run 비교
  compare_runs.py             # 여러 run을 같은 문항 기준으로 비교 (W→C / C→W, McNemar)
  analyze_subject_errors.py   # 과목·이미지 유형별 오답 분석
  build_evidence_pack.py      # 대표 사례·run 메타데이터를 evidence/로 묶기
  lora_smoke_train.py, merge_lora.py   # LoRA smoke 학습 / 병합
  prompt.py, parser.py, evaluator.py, dataset.py   # 프롬프트·파서·채점·데이터 로딩 (팀 공통)
  analyze_*.py                # 오답/토큰 길이/반복 분석 (팀 공통)
scripts/
  check_gpu.sh, run_baseline_modelcard.sh, status_run.sh, gpu_monitor.sh   # 실행·모니터링
  run_proposal_experiments.sh, make_evidence_pack.sh                         # 후속 실험 큐, evidence 생성
  setup.sh, run.sh, run_max_length.sh, run_resolution.sh, run_optimized_eval.sh, run_temp.sh   # 팀 초기 실험 스크립트
docs/
  baseline_modelcard_runbook.md   # 서버 실행 절차 (서버 주소는 placeholder로 치환)
  qwen3vl_experiments.md          # 전체 실험 정리·해석
  proposal_experiments.md         # 후속 실험 설계
  SUBMISSION_TEMPLATE.md          # 과제 제출 템플릿
reports/
  mmmu_baseline_modelcard.md      # 공식 recipe run 리포트 (제출 템플릿 구조)
experiments/
  baseline_modelcard_v1/          # 900문항 baseline: manifest, metrics, summary, predictions.jsonl.gz, run.log.gz, gpu_monitor.csv.gz, pip_freeze.txt
  abl_cot_open/                   # open 53문항 CoT ablation
  p3_sym_cot/                     # 기호 이미지 95문항 CoT ablation
  p1_noise_seed1234/, abl_temp0/, p4_sym_textonly/, p5_sym_hires/, p6_sym_describe/   # 후속 실험
  lora_smoke/, lora_smoke_eval/   # LoRA smoke 로그·요약 (가중치 제외)
  compare_two.md, proposal_table.md   # McNemar 비교표
evidence/                         # 헤드라인 수치, 이미지 유형별 표, 대표 사례(MMMU 이미지 포함), run 메타데이터
  proposal_queue.log              # 후속 실험 큐 기록 (p3~p6)
requirements.txt, requirements-lock.txt
```

모델 가중치와 MMMU 데이터는 포함하지 않습니다 (Hugging Face에서 revision 고정 다운로드).
