# Abandonment ML — Production Pipeline

## Architecture

```
SDK (7 raw features)
  → POST /ml/predictions/abandonment
  → NestJS enrichAbandonmentFeatures()  ← abandonment-derived-features.ts
  → Python predict.py enrich_features_dict()  ← ml/training/derived_features.py
  → LightGBM predict_proba
```

**Rule:** derived features must match between `ml/training/derived_features.py` and `backend/src/ml/abandonment-derived-features.ts`.

## Retrain

```bash
cd trustdev_onboarding_sdk

# 1. Regenerate labels (scrollDepth 0–100 scale fixed)
docker compose -f docker-compose.dev.yml exec backend python3 ml/training/create_realistic_labels.py

# 2. Train (runtime deps only — works in backend container)
docker compose -f docker-compose.dev.yml exec backend python3 ml/training/retrain_runtime.py ml/data/raw/dataset_realistic_labels.json

# 3. Smoke profiles
docker compose -f docker-compose.dev.yml exec backend python3 ml/training/smoke_predict_profiles.py

# 4. Restart backend to reload model
docker compose -f docker-compose.dev.yml restart backend
```

Expected smoke output (v2 aligned model):
- 2 click-miss @ 5s → ~2% (not 24%)
- 2 click-miss @ 72s → ~42%
- 2 click-miss @ 83s + 30% scroll → ~33% (below 35% toast)

Artifacts written to:
- `ml/models/lightgbm_model.pkl`
- `ml/models/feature_names.json`
- `ml/results/metrics.json`

## Hybrid SDK strategy

| Phase | Condition | Display |
|-------|-----------|---------|
| Pre-gate | `clickMiss + scrollHesitation < minSignals` | Local heuristic |
| ML | Gate open + API 200 | LightGBM probability |
| Throttle | < 10s since last success | Cache / local if friction rose |
| Fallback | API error | Local heuristic |

Proactive toast (production): **ML only**, `status === 'ok'`, `confidence >= 0.7`, `risk >= threshold`.

## Rate limits

- Backend: 1 req / 10s / `sessionId` (recorded only after successful inference)
- Client: debounce 800ms + same 10s cooldown + fingerprint buckets (time/10s, scroll/5%)

## Known model limits

- Labels are behavioral proxies until real `user_progress.is_abandoned` is wired into export
- Retrain when production abandonment outcomes are available
- Monitor precision/recall on held-out test set before changing toast threshold
