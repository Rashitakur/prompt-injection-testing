# Auto-generated findings (verify every claim in the paper against this)

## ID test F1
  1. distilbert: 0.9815 (±0.0055, seeds=3)
  2. tfidf_lr: 0.9756 (±0.0000, seeds=1)
  3. emb_bge_lr: 0.9583 (±0.0000, seeds=1)
  4. emb_minilm_lr: 0.9563 (±0.0000, seeds=1)
  5. judge_qwen7b: 0.9332 (±0.0000, seeds=1)
  6. ref_protectai: 0.9314 (±0.0000, seeds=1)
  7. judge_qwen1p5b: 0.8742 (±0.0000, seeds=1)
  8. ppl_gpt2-lr: 0.5585 (±0.0000, seeds=1)
  9. ppl_gpt2-ppl: 0.5242 (±0.0000, seeds=1)
  10. ppl_gpt2-winppl: 0.4937 (±0.0000, seeds=1)

## ID test AUROC
  1. distilbert: 0.9985 (±0.0002, seeds=3)
  2. tfidf_lr: 0.9971 (±0.0000, seeds=1)
  3. emb_bge_lr: 0.9954 (±0.0000, seeds=1)
  4. emb_minilm_lr: 0.9937 (±0.0000, seeds=1)
  5. judge_qwen7b: 0.9893 (±0.0000, seeds=1)
  6. ref_protectai: 0.9831 (±0.0000, seeds=1)
  7. judge_qwen1p5b: 0.9594 (±0.0000, seeds=1)
  8. ppl_gpt2-lr: 0.7076 (±0.0000, seeds=1)
  9. ppl_gpt2-ppl: 0.6704 (±0.0000, seeds=1)
  10. ppl_gpt2-winppl: 0.5403 (±0.0000, seeds=1)

## benign_noisy FPR (lower is better)
  1. tfidf_lr: 0.0017 (±0.0000, seeds=1)
  2. distilbert: 0.0172 (±0.0199, seeds=3)
  3. emb_minilm_lr: 0.0283 (±0.0000, seeds=1)
  4. emb_bge_lr: 0.0533 (±0.0000, seeds=1)
  5. judge_qwen7b: 0.0700 (±0.0000, seeds=1)
  6. judge_qwen1p5b: 0.1167 (±0.0000, seeds=1)
  7. ref_protectai: 0.1967 (±0.0000, seeds=1)
  8. ppl_gpt2-lr: 0.5733 (±0.0000, seeds=1)
  9. ppl_gpt2-ppl: 0.7383 (±0.0000, seeds=1)
  10. ppl_gpt2-winppl: 1.0000 (±0.0000, seeds=1)

## benign_notinject FPR (lower is better)
  1. judge_qwen7b: 0.0826 (±0.0000, seeds=1)
  2. judge_qwen1p5b: 0.1357 (±0.0000, seeds=1)
  3. tfidf_lr: 0.2478 (±0.0000, seeds=1)
  4. emb_minilm_lr: 0.3628 (±0.0000, seeds=1)
  5. emb_bge_lr: 0.3953 (±0.0000, seeds=1)
  6. distilbert: 0.4926 (±0.1483, seeds=3)
  7. ppl_gpt2-ppl: 0.5192 (±0.0000, seeds=1)
  8. ref_protectai: 0.5664 (±0.0000, seeds=1)
  9. ppl_gpt2-lr: 0.7493 (±0.0000, seeds=1)
  10. ppl_gpt2-winppl: 0.8466 (±0.0000, seeds=1)

## ood_gandalf_dolly AUROC
  1. ref_protectai: 1.0000 (±0.0000, seeds=1)
  2. judge_qwen7b: 0.9995 (±0.0000, seeds=1)
  3. tfidf_lr: 0.9969 (±0.0000, seeds=1)
  4. distilbert: 0.9964 (±0.0019, seeds=3)
  5. emb_minilm_lr: 0.9954 (±0.0000, seeds=1)
  6. judge_qwen1p5b: 0.9942 (±0.0000, seeds=1)
  7. emb_bge_lr: 0.9907 (±0.0000, seeds=1)
  8. ppl_gpt2-winppl: 0.7436 (±0.0000, seeds=1)
  9. ppl_gpt2-ppl: 0.7405 (±0.0000, seeds=1)
  10. ppl_gpt2-lr: 0.7374 (±0.0000, seeds=1)

## ood_jailbreak AUROC
  1. ref_protectai: 0.9815 (±0.0000, seeds=1)
  2. tfidf_lr: 0.9785 (±0.0000, seeds=1)
  3. judge_qwen7b: 0.9704 (±0.0000, seeds=1)
  4. distilbert: 0.9630 (±0.0044, seeds=3)
  5. emb_bge_lr: 0.9452 (±0.0000, seeds=1)
  6. emb_minilm_lr: 0.9423 (±0.0000, seeds=1)
  7. judge_qwen1p5b: 0.9211 (±0.0000, seeds=1)
  8. ppl_gpt2-winppl: 0.7954 (±0.0000, seeds=1)
  9. ppl_gpt2-lr: 0.4810 (±0.0000, seeds=1)
  10. ppl_gpt2-ppl: 0.4066 (±0.0000, seeds=1)

## ood_spml AUROC
  1. ref_protectai: 0.9967 (±0.0000, seeds=1)
  2. judge_qwen7b: 0.9886 (±0.0000, seeds=1)
  3. judge_qwen1p5b: 0.9822 (±0.0000, seeds=1)
  4. ppl_gpt2-winppl: 0.9690 (±0.0000, seeds=1)
  5. tfidf_lr: 0.9565 (±0.0000, seeds=1)
  6. distilbert: 0.9522 (±0.0033, seeds=3)
  7. emb_bge_lr: 0.9244 (±0.0000, seeds=1)
  8. emb_minilm_lr: 0.8963 (±0.0000, seeds=1)
  9. ppl_gpt2-lr: 0.8210 (±0.0000, seeds=1)
  10. ppl_gpt2-ppl: 0.7791 (±0.0000, seeds=1)

## stress_adv_suffix AUROC
  1. distilbert: 0.9999 (±0.0000, seeds=3)
  2. ref_protectai: 0.9991 (±0.0000, seeds=1)
  3. emb_bge_lr: 0.9989 (±0.0000, seeds=1)
  4. tfidf_lr: 0.9988 (±0.0000, seeds=1)
  5. judge_qwen7b: 0.9968 (±0.0000, seeds=1)
  6. emb_minilm_lr: 0.9946 (±0.0000, seeds=1)
  7. judge_qwen1p5b: 0.9857 (±0.0000, seeds=1)
  8. ppl_gpt2-winppl: 0.9850 (±0.0000, seeds=1)
  9. ppl_gpt2-ppl: 0.9038 (±0.0000, seeds=1)
  10. ppl_gpt2-lr: 0.8862 (±0.0000, seeds=1)

## Mean OOD AUROC
  ref_protectai: 0.9927
  judge_qwen7b: 0.9862
  tfidf_lr: 0.9773
  distilbert: 0.9705
  judge_qwen1p5b: 0.9658
  emb_bge_lr: 0.9534
  emb_minilm_lr: 0.9447
  ppl_gpt2-winppl: 0.8360
  ppl_gpt2-lr: 0.6798
  ppl_gpt2-ppl: 0.6421

## Latency (p50 ms, bs=1)
  [cpu] tfidf_lr: p50=1.28 ms, bs32 throughput=1842.6/s
  [cpu] emb_minilm_lr: p50=8.36 ms, bs32 throughput=58.1/s
  [cpu] emb_bge_lr: p50=14.58 ms, bs32 throughput=15.0/s
  [cpu] distilbert: p50=18.07 ms, bs32 throughput=18.1/s
  [cpu] ppl_gpt2: p50=61.80 ms, bs32 throughput=1.5/s
  [cpu] ref_protectai: p50=82.40 ms, bs32 throughput=2.5/s
  [cpu] judge_qwen1p5b: p50=1388.07 ms, bs32 throughput=0.2/s
  [cuda] tfidf_lr: p50=1.31 ms, bs32 throughput=1786.3/s
  [cuda] emb_minilm_lr: p50=14.21 ms, bs32 throughput=432.9/s
  [cuda] distilbert: p50=15.71 ms, bs32 throughput=704.8/s
  [cuda] emb_bge_lr: p50=16.08 ms, bs32 throughput=163.4/s
  [cuda] ref_protectai: p50=25.08 ms, bs32 throughput=106.9/s
  [cuda] ppl_gpt2: p50=31.01 ms, bs32 throughput=21.6/s
  [cuda] judge_qwen1p5b: p50=51.18 ms, bs32 throughput=21.6/s
  [cuda] judge_qwen7b: p50=95.99 ms, bs32 throughput=6.1/s

## Paired bootstrap vs best (distilbert) on ID F1
  tfidf_lr: dF1=+0.0104 p=0.0030
  emb_bge_lr: dF1=+0.0277 p=0.0010
  emb_minilm_lr: dF1=+0.0298 p=0.0010
  ppl_gpt2-lr: dF1=+0.4275 p=0.0010
  ppl_gpt2-ppl: dF1=+0.4618 p=0.0010
  ppl_gpt2-winppl: dF1=+0.4923 p=0.0010
  judge_qwen1p5b: dF1=+0.1118 p=0.0010
  judge_qwen7b: dF1=+0.0528 p=0.0010
  ref_protectai: dF1=+0.0547 p=0.0010

## Cascade (cheap -> judge)
            split  frac_target  frac_escalated       f1      fpr   recall  exp_latency_ms latency_device      cheap    expensive
          test_id         0.00        0.000000 0.986019 0.001386 0.975255       15.711152           cuda distilbert judge_qwen7b
          test_id         0.05        0.043662 0.984638 0.004851 0.979622       19.902221           cuda distilbert judge_qwen7b
          test_id         0.10        0.096244 0.980335 0.009009 0.979622       24.949530           cuda distilbert judge_qwen7b
          test_id         0.20        0.196244 0.975362 0.013860 0.979622       34.548430           cuda distilbert judge_qwen7b
          test_id         0.30        0.307042 0.969087 0.020790 0.981077       45.183831           cuda distilbert judge_qwen7b
          test_id         1.00        1.000000 0.933237 0.033264 0.935953       95.989000           cuda distilbert judge_qwen7b
ood_gandalf_dolly         0.00        0.000000 0.961578 0.001014 0.926927       15.711152           cuda distilbert judge_qwen7b
ood_gandalf_dolly         0.05        0.067003 0.986829 0.001014 0.974975       22.142657           cuda distilbert judge_qwen7b
ood_gandalf_dolly         0.10        0.137028 0.986842 0.002028 0.975976       28.864305           cuda distilbert judge_qwen7b
ood_gandalf_dolly         0.20        0.268514 0.988889 0.002028 0.979980       41.485528           cuda distilbert judge_qwen7b
ood_gandalf_dolly         0.30        0.386398 0.988390 0.003043 0.979980       52.801108           cuda distilbert judge_qwen7b
ood_gandalf_dolly         1.00        1.000000 0.981170 0.002028 0.964965       95.989000           cuda distilbert judge_qwen7b
ood_gandalf_dolly         0.00        0.000000 0.961578 0.001014 0.926927       15.711152           cuda distilbert judge_qwen7b
ood_gandalf_dolly         0.05        0.067003 0.986829 0.001014 0.974975       22.142657           cuda distilbert judge_qwen7b
ood_gandalf_dolly         0.10        0.137028 0.986842 0.002028 0.975976       28.864305           cuda distilbert judge_qwen7b
ood_gandalf_dolly         0.20        0.268514 0.988889 0.002028 0.979980       41.485528           cuda distilbert judge_qwen7b
ood_gandalf_dolly         0.30        0.386398 0.988390 0.003043 0.979980       52.801108           cuda distilbert judge_qwen7b
ood_gandalf_dolly         1.00        1.000000 0.981170 0.002028 0.964965       95.989000           cuda distilbert judge_qwen7b
ood_gandalf_dolly         0.00        0.000000 0.961578 0.001014 0.926927       15.711152           cuda distilbert judge_qwen7b
ood_gandalf_dolly         0.05        0.067003 0.986829 0.001014 0.974975       22.142657           cuda distilbert judge_qwen7b
ood_gandalf_dolly         0.10        0.137028 0.986842 0.002028 0.975976       28.864305           cuda distilbert judge_qwen7b
ood_gandalf_dolly         0.20        0.268514 0.988889 0.002028 0.979980       41.485528           cuda distilbert judge_qwen7b
ood_gandalf_dolly         0.30        0.386398 0.988390 0.003043 0.979980       52.801108           cuda distilbert judge_qwen7b
ood_gandalf_dolly         1.00        1.000000 0.981170 0.002028 0.964965       95.989000           cuda distilbert judge_qwen7b
    ood_jailbreak         0.00        0.000000 0.679070 0.423862 0.979866       15.711152           cuda distilbert judge_qwen7b
    ood_jailbreak         0.05        0.154011 0.679027 0.427002 0.983221       30.494485           cuda distilbert judge_qwen7b
    ood_jailbreak         0.10        0.183957 0.677457 0.430141 0.983221       33.369022           cuda distilbert judge_qwen7b
    ood_jailbreak         0.20        0.254545 0.675894 0.433281 0.983221       40.144716           cuda distilbert judge_qwen7b
    ood_jailbreak         0.30        0.308021 0.677419 0.433281 0.986577       45.277817           cuda distilbert judge_qwen7b
    ood_jailbreak         1.00        1.000000 0.736976 0.312402 0.973154       95.989000           cuda distilbert judge_qwen7b
    ood_jailbreak         0.00        0.000000 0.679070 0.423862 0.979866       15.711152           cuda distilbert judge_qwen7b
    ood_jailbreak         0.05        0.154011 0.679027 0.427002 0.983221       30.494485           cuda distilbert judge_qwen7b
    ood_jailbreak         0.10        0.183957 0.677457 0.430141 0.983221       33.369022           cuda distilbert judge_qwen7b
    ood_jailbreak         0.20        0.254545 0.675894 0.433281 0.983221       40.144716           cuda distilbert judge_qwen7b
    ood_jailbreak         0.30        0.308021 0.677419 0.433281 0.986577       45.277817           cuda distilbert judge_qwen7b
    ood_jailbreak         1.00        1.000000 0.736976 0.312402 0.973154       95.989000           cuda distilbert judge_qwen7b
    ood_jailbreak         0.00        0.000000 0.679070 0.423862 0.979866       15.711152           cuda distilbert judge_qwen7b
    ood_jailbreak         0.05        0.154011 0.679027 0.427002 0.983221       30.494485           cuda distilbert judge_qwen7b
    ood_jailbreak         0.10        0.183957 0.677457 0.430141 0.983221       33.369022           cuda distilbert judge_qwen7b
    ood_jailbreak         0.20        0.254545 0.675894 0.433281 0.983221       40.144716           cuda distilbert judge_qwen7b
    ood_jailbreak         0.30        0.308021 0.677419 0.433281 0.986577       45.277817           cuda distilbert judge_qwen7b
    ood_jailbreak         1.00        1.000000 0.736976 0.312402 0.973154       95.989000           cuda distilbert judge_qwen7b
         ood_spml         0.00        0.000000 0.858545 0.007034 0.753591       15.711152           cuda distilbert judge_qwen7b
         ood_spml         0.05        0.164827 0.888692 0.002345 0.800192       31.532721           cuda distilbert judge_qwen7b
         ood_spml         0.10        0.231059 0.894802 0.003517 0.810405       37.890246           cuda distilbert judge_qwen7b
         ood_spml         0.20        0.290015 0.898683 0.003517 0.816789       43.549407           cuda distilbert judge_qwen7b
         ood_spml         0.30        0.345961 0.899298 0.004689 0.818066       48.919589           cuda distilbert judge_qwen7b
         ood_spml         1.00        1.000000 0.792373 0.002345 0.656559       95.989000           cuda distilbert judge_qwen7b
         ood_spml         0.00        0.000000 0.858545 0.007034 0.753591       15.711152           cuda distilbert judge_qwen7b
         ood_spml         0.05        0.164827 0.888692 0.002345 0.800192       31.532721           cuda distilbert judge_qwen7b
         ood_spml         0.10        0.231059 0.894802 0.003517 0.810405       37.890246           cuda distilbert judge_qwen7b
         ood_spml         0.20        0.290015 0.898683 0.003517 0.816789       43.549407           cuda distilbert judge_qwen7b
         ood_spml         0.30        0.345961 0.899298 0.004689 0.818066       48.919589           cuda distilbert judge_qwen7b
         ood_spml         1.00        1.000000 0.792373 0.002345 0.656559       95.989000           cuda distilbert judge_qwen7b
         ood_spml         0.00        0.000000 0.858545 0.007034 0.753591       15.711152           cuda distilbert judge_qwen7b
         ood_spml         0.05        0.164827 0.888692 0.002345 0.800192       31.532721           cuda distilbert judge_qwen7b
         ood_spml         0.10        0.231059 0.894802 0.003517 0.810405       37.890246           cuda distilbert judge_qwen7b
         ood_spml         0.20        0.290015 0.898683 0.003517 0.816789       43.549407           cuda distilbert judge_qwen7b
         ood_spml         0.30        0.345961 0.899298 0.004689 0.818066       48.919589           cuda distilbert judge_qwen7b
         ood_spml         1.00        1.000000 0.792373 0.002345 0.656559       95.989000           cuda distilbert judge_qwen7b
