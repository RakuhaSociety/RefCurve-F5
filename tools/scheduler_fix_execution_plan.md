# Scheduler Fix Execution Plan

## Status: Ready for Remote Execution

### ✅ Local Preparation Complete

1. **Code fixes implemented**:
   - `src/f5_tts/model/trainer.py`: `resolve_scheduler_contract()` with unified multiplier
   - `src/f5_tts/model/trainer.py`: `resolve_split_batches()` explicit resolution
   - `src/f5_tts/model/trainer.py`: `build_scheduler()` extracted function
   - `src/f5_tts/model/trainer.py`: `step_was_skipped` check in training loop
   - `src/f5_tts/model/trainer.py`: `lr_trace_path` for LR forensics

2. **Tests passing**:
   - `tests/test_trainer_scheduler.py`: 35/35 tests passed locally
   - Coverage: contract semantics, world-size invariance, grad accumulation, resume, smoke diagnostics

3. **Adjudication tooling**:
   - `src/f5_tts/train/run_adjudication.py`: simplified (193 lines)
   - Evidence file: `eval_specs/abc_v1_calibration_5000_adjudication_evidence.json`

4. **Smoke scripts**:
   - `tools/smoke_scheduler_1gpu_120.sh`
   - `tools/smoke_scheduler_8gpu_120.sh`

5. **Code synced to GitHub**: commit `82aa5b3`

---

## Remote Execution Steps

### Step 1: Sync Code to Training Server

```bash
ssh -p 9000 root@183.147.142.130
cd /root/F5-TTS
git stash  # save any local changes
git pull origin main
git log --oneline -3  # verify at 82aa5b3 or later
```

Expected HEAD: `82aa5b3 refactor: simplify run_adjudication + add checkpoint hashes to eval spec`

---

### Step 2: Run Remote Tests

```bash
cd /root/F5-TTS
python3 -m pytest tests/test_trainer_scheduler.py -v
```

Expected: 35/35 tests pass (same as local)

---

### Step 3: Write Old Run Adjudication

**IMPORTANT**: This modifies remote filesystem. Requires explicit authorization.

```bash
cd /root/F5-TTS

# Dry run first
python3 -m f5_tts.train.run_adjudication \
  --run-dir /root/F5-TTS/ckpts/visualnovel_calibration_ja/abc-v1-calibration-5000-20260829 \
  --status invalid_for_calibration \
  --reason "Scheduler horizon not scaled by world size: warmup multiplied by 8, total was not. LR reached floor at ~update 625 instead of 5000. Training ineffective after that point." \
  --evidence @eval_specs/abc_v1_calibration_5000_adjudication_evidence.json \
  --dry-run

# If output looks correct, write it
python3 -m f5_tts.train.run_adjudication \
  --run-dir /root/F5-TTS/ckpts/visualnovel_calibration_ja/abc-v1-calibration-5000-20260829 \
  --status invalid_for_calibration \
  --reason "Scheduler horizon not scaled by world size: warmup multiplied by 8, total was not. LR reached floor at ~update 625 instead of 5000. Training ineffective after that point." \
  --evidence @eval_specs/abc_v1_calibration_5000_adjudication_evidence.json
```

Verifies:
- `run_adjudication.json` created in old run dir
- Manifest and checkpoints unchanged
- Atomic write, idempotent

---

### Step 4: Run 1 GPU Smoke (120 updates)

```bash
cd /root/F5-TTS
bash tools/smoke_scheduler_1gpu_120.sh
```

Expected duration: ~5-10 minutes
Expected output: LR trace showing correct warmup/decay trajectory

Key anchors to verify:
- update 1: ~1.10e-6 (warmup start)
- update 50: ~5.05e-6 (mid-warmup)
- update 100: 1.00e-05 (warmup peak)
- update 101: ~9.90e-06 (decay start)
- update 120: ~9.73e-06 (slight decay)

---

### Step 5: Run 8 GPU Smoke (120 updates)

```bash
cd /root/F5-TTS
bash tools/smoke_scheduler_8gpu_120.sh
```

Expected duration: ~5-10 minutes
Expected output: **Identical LR trajectory to 1 GPU test**

This proves world-size invariance.

---

### Step 6: Generate Corrected Run Dry-Run

```bash
cd /root/F5-TTS

# Create new run manifest
RUN_DATE=$(date +%Y%m%d)
NEW_RUN_DIR="/root/F5-TTS/ckpts/visualnovel_calibration_ja/corrected-calibration-5000-$RUN_DATE"

# Show what will be executed (DO NOT RUN YET)
cat > /tmp/corrected_run_preview.sh <<'EOF'
python3 -m accelerate.commands.launch \
  --num_processes 8 \
  --mixed_precision bf16 \
  --num_machines 1 \
  --machine_rank 0 \
  --main_process_port 29500 \
  src/f5_tts/train/train.py \
    --config-name F5TTS_v1_JA_Base \
    ++trainer.epochs=999 \
    ++trainer.max_updates=5000 \
    ++trainer.num_warmup_updates=100 \
    ++trainer.save_per_updates=500 \
    ++trainer.last_per_updates=100 \
    ++trainer.keep_last_n_checkpoints=10 \
    ++trainer.checkpoint_path="$NEW_RUN_DIR" \
    ++trainer.lr_trace_path="$NEW_RUN_DIR/lr_trace.jsonl" \
    ++trainer.pretrained_init="ckpts/F5TTS_v1_Base/model_1250000.safetensors" \
    ++trainer.pretrained_init_options.vocab_path="src/f5_tts/configs/vocab/ja_48khz.txt" \
    ++trainer.pretrained_init_options.vocab_contract_path="src/f5_tts/configs/vocab/ja_contract.json" \
    ++trainer.pretrained_init_options.seed=666 \
    ++trainer.logger=null \
    ++datasets.batch_size_per_gpu=3200 \
    ++datasets.max_samples=8 \
    ++datasets.data_dir="/data/visualnovel/aggregates/wave-abc-v1-exclude-conflicts" \
    ++accelerate_kwargs.split_batches=False
EOF

cat /tmp/corrected_run_preview.sh
```

Review the command. Verify:
- New run dir (not overwriting old run)
- From v1 EMA update 0 (pretrained_init with vocab remap)
- 8 GPU, split_batches=False (same as bug run)
- global warmup=100, total=5000 (scheduler will scale to 800/40000 internally)
- LR trace enabled
- Same dataset as bug run

---

### Step 7: Authorization Gate for Full Run

**STOP HERE**. The full 5000-update run requires separate authorization:

- **Duration**: ~8-12 hours (8× RTX 4090)
- **Cost**: Significant GPU time
- **Scope**: Creates 9 checkpoints (500/1000/.../4500 + model_last.pt)
- **Risk**: Low (new run dir, old run untouched)

After smoke tests pass and dry-run is reviewed, request explicit authorization:

```
Authorization request for corrected calibration run:
- Target: corrected-calibration-5000-20260906 (new dir)
- Source: v1 EMA update 0 + vocab remap seed 666
- Dataset: wave-abc-v1-exclude-conflicts (same as bug run)
- Duration: ~8-12 hours
- Expected checkpoints: 9 EMA + model_last.pt
- LR verification: update 500 ~9.18e-6, update 5000 ~1e-13
```

Only after explicit "yes" to this specific request, execute `/tmp/corrected_run_preview.sh`.

---

## Verification Checklist

After full run completes:

- [ ] 9 EMA checkpoints exist (update 500-4500 by 500)
- [ ] model_last.pt exists (= update 5000)
- [ ] lr_trace.jsonl contains 5000+ records
- [ ] LR at update 500: ~9.18e-6 (4× higher than bug run's 2.38e-6)
- [ ] LR at update 1000: ~8.16e-6 (not 1e-13 floor)
- [ ] LR at update 5000: ~1e-13 (correct floor timing)
- [ ] run_manifest.json created with correct run identity
- [ ] Old run dir unchanged

---

## Current Blockers

- SSH connection to 183.147.142.130:9000 is slow/timing out
- git pull on remote may still be in progress
- Need stable connection to execute remote tests and smoke runs

---

## Next Actions

1. Wait for SSH connection to stabilize
2. Verify remote code is at commit 82aa5b3
3. Execute Steps 2-5 (tests + smokes)
4. Review Step 6 dry-run output
5. Request authorization for Step 7 (full run)
