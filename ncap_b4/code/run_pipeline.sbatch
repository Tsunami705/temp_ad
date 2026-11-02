#!/usr/bin/env bash
#SBATCH -A naiss2026-1-47-gpu -p gpu --gpus-per-node=1
#SBATCH -t 02:00:00
#SBATCH -J riskmon_ncap
#SBATCH -o /nobackup/proj/disk/cvls/personal/shipengl/ad/slurm_logs/riskmon_%j.out
# NCAP latents -> Table 2 (UniAD): MC Dropout, GMM, RBD (no training; also GMM-4 / RBD-4 over the label horizon),
# LP-FCN, RiskMonitor-T, RiskMonitor-G (4 bagging members each, epoch chosen on val), test scores, scene-bootstrap CIs.
# Code: riskmonitor_g (copy of Ziliang's shared folder) + ncap_to_cache.py / score_cv.py / ncap_eval.py;
# image: UniAD/uniad.sif (has pandas, sklearn, mmdet3d).
#   LATENTS=.../latents/ncap_uniad_nocol_r10 NCAP_OUT=.../neuro-ncap/outputs/<TIME> OUT=.../latents/riskmonitor/<name> \
#     sbatch run_pipeline.sbatch
set -euo pipefail
AD=/nobackup/proj/disk/cvls/personal/shipengl/ad
RM=/nobackup/proj/disk/cvls/personal/shipengl/riskmonitor_g   # copy of naiss2024-6-300/shared/riskmonitor_g
: "${LATENTS:?}" "${NCAP_OUT:?}" "${OUT:?}"
EPOCHS=${EPOCHS:-20}                    # T / G / LP-FCN, checkpoints every 2 epochs
SPLIT=${SPLIT:-cv}                      # cv (scene cross-validation, FOLDS folds) | scene | random | plan
PLAN=${PLAN:-}                          # SPLIT=plan: ncap_plan.py output (split, runs per pair, label key)
FOLDS=${FOLDS:-5}
MEMBERS=${MEMBERS:-4}
LABEL_STEPS=${LABEL_STEPS:-4}           # plan steps checked for the per-step label (6 or 4)
RUN=(apptainer exec --nv -B /nobackup:/nobackup --pwd $RM --env PYTHONPATH=$RM,PYTHONNOUSERSITE=1 $AD/UniAD/uniad.sif)

"${RUN[@]}" python $AD/riskmonitor/ncap_to_cache.py --latents $LATENTS --ncap-out $NCAP_OUT --out $OUT \
  --epochs $EPOCHS --split $SPLIT --folds $FOLDS --members $MEMBERS --label-steps $LABEL_STEPS ${PLAN:+--plan $PLAN}

if [ $SPLIT = cv ]; then DIRS=$(ls -d $OUT/f[0-9]*); else DIRS=$OUT; fi
for d in $DIRS; do
  echo "== $d"
  for m in $(seq 1 $MEMBERS); do
    for r in T G LPFCN; do
      mkdir -p $d/$r/m$m
      script=$([ $r = LPFCN ] && echo rebuttal2/lpfcn_train.py || echo train_transformer_mixup.py)
      "${RUN[@]}" python $script -c $d/configs/${r}_m$m.json > $d/$r/m$m/train.log 2>&1 &
    done
  done
  wait
  "${RUN[@]}" python $AD/riskmonitor/score_cv.py --out $d > $d/score_cv.log 2>&1
  grep -E "^(T|G|LPFCN): " $d/score_cv.log
done

"${RUN[@]}" python $AD/riskmonitor/ncap_eval.py --out $OUT
