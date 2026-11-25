# Code copied from immich-memories

Source: <https://github.com/sam-dumont/immich-memories>, tag **v1.0.0** (released 24 Oct 2026).
Licence: MIT, see `LICENSE` in this folder (Copyright (c) 2024 Immich Memories Contributors).
Copied on 24 Nov 2026 for the highlight-video pipeline (phase 4 of `docs/plan.md`).

| File here | Original path (`src/immich_memories/...`) | Changes |
|---|---|---|
| `renderer.py` | `photos/renderer.py` | Removed `move_frames` (GPU path importing `photos.gpu_moves`); the rest unchanged. |
| `ken_burns_moves.py` | `photos/ken_burns_moves.py` | Imports made relative to this folder. |
| `framing_target.py` | `analysis/facts/framing_target.py` | Import made relative. |
| `subject_framing.py` | `analysis/facts/subject_framing.py` | Unchanged. |
| `beat_grid.py` | `audio/beat_grid.py` | Unchanged. |
| `track_tempo.py` | `audio/track_tempo.py` | `cache.file_facts.remembered_fact` replaced by an in-process cache; ffmpeg path configurable via `FFMPEG`; import made relative. |
| `audio_decode.py` | `audio/audio_decode.py` | `ffmpeg` executable passed in as a parameter. |

Not copied: `analysis`, `titles`, `web`, `db`, `cli`, `automation` and the rest of `processing`.
Fixes from upstream don't arrive automatically; compare against a newer tag before updating.
