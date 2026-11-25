# Photos companion: plan

A separate Docker service that reads photos from an existing Immich server (v3.4.2, on the LAN) through its API, makes **collages**, **cinematic photos** (with depth maps) and **highlight videos with music** on the CPU, and uploads the results back into Immich. It has its own small web UI for the job queue, suggestions and settings.

Decisions so far (from the conversation on 24 Nov 2026):

| Topic | Decision |
|---|---|
| Stack | Python 3.13, FastAPI, server-rendered Jinja pages + a little vanilla JS, SQLite. No frontend build step, no Redis. |
| Immich client | `immichpy` 10.3.0 (typed, async, generated for Immich v3.4.2) |
| Deployment | Own compose file, reaches Immich via `IMMICH_URL` + API key |
| Results | Back into Immich: album, tag, stack with source, Memories strip |
| Models | Local CPU only, no cloud APIs. Licences must allow use (MIT / Apache-2.0). |
| Project licence | MIT. AGPL projects (DepthFlow, immich-power-tools) are reference only. |
| UI access | Single user, one password |
| UI layout | Mockup option C, "command deck" (`docs/mockups/2026-11-24-companion-ui.html`) |
| Suggestions | Both modes selectable: "Suggest, I approve" (default) and "Create and upload automatically" |
| Music | User-mounted folder `/music`, read-only |
| Depth map | Uploaded to Immich as a PNG, stacked with the source |
| Highlight video | Copy selected modules from immich-memories v1.0.0 (MIT) into `vendor/` |
| Tests | pytest with a fake Immich; no writes to the real library yet |

## 1. Architecture

```
            LAN                                   photos companion (one container)
┌──────────────────────┐   HTTP + x-api-key   ┌───────────────────────────────────────────┐
│ Immich v3.4.2        │◄─────────────────────┤ app.immich  (immichpy AsyncClient wrapper) │
│  /api/assets, albums │                      │ app.worker  (asyncio loop, 1 job at a time)│
│  tags, stacks,       │                      │ app.pipelines.{collage,cinematic,highlight}│
│  memories, search    │                      │ app.jobs / app.db  (SQLite in /data)       │
└──────────────────────┘                      │ app.main   (FastAPI UI, password login)    │
                                              └───────────────▲───────────────────────────┘
                                         browser on the LAN ──┘  port 8765
```

- The API key lives only in `.env`. The UI never shows it.
- Thumbnails shown in the UI are proxied through the companion, so the browser never talks to Immich and the CSP can stay `'self'`.
- One worker, one job at a time by default (setting). Jobs left `running` after a restart go back to `queued`.

## 2. Immich API key permissions

Checked against the v3.4.2 OpenAPI spec (`x-immich-permission`). `GET /api/api-keys/me` needs no permission and returns the key's own permission list, so the Settings page can check the key exactly, without probing.

| Permission | Why |
|---|---|
| `asset.read` | Asset info, EXIF, people, search, timeline |
| `asset.view` | Thumbnails / previews for analysis; video playback |
| `asset.download` | Originals for final renders |
| `asset.upload` | Upload results |
| `asset.update` | Set the description on results |
| `album.read`, `album.create`, `albumAsset.create` | Source albums; the "Creations" album |
| `person.read`, `face.read` | People filters; face boxes for crops |
| `tag.read`, `tag.create`, `tag.asset` | `creation:*` tags |
| `stack.create` | Stack depth map (and cinematic MP4) with the source |
| `memory.create`, `memoryAsset.create` | Memories strip |

To verify on the real server (can't be proven with mocks):
- whether `GET /assets/{id}` returns face boxes without `face.read`;
- whether a stack can mix a video with a photo;
- how a birthday memory looks. Immich only has the memory types `on_this_day` and `birthday`, and Immich already shows "N years ago" memories, so only birthday events add a memory (type `birthday`, the person's name). Other events go into a named album plus an asset description instead.

## 3. Pipelines

| Pipeline | Input | Output | Main libraries / models |
|---|---|---|---|
| Collage | 2–9 assets (12 for auto) | JPEG | Pillow, NumPy, OpenCV (blur score), BRIC layout (MIT) |
| Cinematic | 1 photo | MP4 + 16-bit depth PNG | ONNX Runtime CPU, Depth Anything V2 Small (Apache-2.0), OpenCV, ffmpeg |
| Highlight video | album / person / dates / smart search | MP4 | ffmpeg, copied immich-memories modules (MIT), beat detection |
| Pop-out | 1 photo with people | 1 JPEG per chosen style | ONNX Runtime CPU, MODNet (Apache-2.0), OpenCV; colour splash after aeren108/color_splash and HareColor |

Rules for each pipeline are in `docs/research/immich-vs-google-photos.md` sections 3–5.

Every pipeline gets a context with: the gateway, the job's parameters, a `progress(stage, percent)` callback, and a `cancelled()` check between steps. A pipeline returns the uploaded asset IDs. The worker then does the shared write-back: album, tags, stack, memory.

### Copying from immich-memories

- Source: tag `v1.0.0` (released 24 Oct 2026). Candidates: `src/immich_memories/photos/` (11 files), `audio/` (41), and the render part of `processing/` (65 files total).
- Their imports of `api`, `config`, `titles`, `security` and `analysis` are replaced with small adapters onto our config and `immichpy`. `analysis`, `titles`, `web`, `db`, `cli` and `automation` are not copied.
- Kept: their `LICENSE`, plus `vendor/immich_memories/NOTICE.md` listing every copied file, the source commit and our changes.
- The exact file list is decided in phase 4 by tracing imports file by file.

## 4. Phases

| Phase | Scope | Status (24 Nov 2026) |
|---|---|---|
| 1. Skeleton | Config, Immich gateway, job store, worker, login, UI layout C, settings, Docker | Done |
| 2. Collage | Filters (blur, duplicates, same scene, screenshots), least-crop layouts, face-aware crop; styles grid / tilted prints / polaroid scatter / gallery fan, default random of the three tilted ones | Done; tested on the real server in folder mode |
| 3. Cinematic + depth | Depth Anything V2 Small (pinned + SHA-256), edge cleanup, people pulled forward, zoom / dolly / circle scored by stretch, forward-splat render with inpainted gaps, MP4 + 16-bit depth PNG | Done; tested on the real server in folder mode |
| 4. Highlight video | Copied immich-memories modules (Ken Burns moves, tempo detection, beat grid), on-beat cuts, clips with blurred fill, crossfades, music folder | Done; tested on the real server in folder mode. Not yet: clip sound / ducking, title cards |
| 5. Suggestions | Daily scan, events by time gap and GPS jump, ranked ideas with kinds mixed, approve / dismiss, auto mode | Done; scan tested read-only on the real server |
| 6. Writing to Immich | Upload, album, album per event, tag, stack, description, birthday memory | Tested against the fake Immich only. The test key has no write permissions; the "to verify" points in section 2 are still open |
| 7. Pop-out | MODNet matte (flip-averaged) cleaned with face-based region keep, hole fill, spike removal, head trim, colour guided filter and foreground estimation; styles colour splash, spotlight, portrait blur (lens blur on a clean plate, graded by depth), pop out of frame, sticker | Done; tested on the real server in folder mode |
| 8. Events, editing, housekeeping | Event types (birthday from Immich birth dates, festival tables, OCR text, home vs away by GPS), trips across towns, "through the years"; highlight title cards; edit title on a finished job; presets; retention clean-up | Done; event scan tested read-only on the real server |

## 5. Phase 1 contents

```
photos/
  README.md                 how to run, configure and test
  .env.example              IMMICH_URL, IMMICH_API_KEY, PHOTOS_PASSWORD, PHOTOS_BIND
  docker-compose.yml        one service, ./data and ./music volumes, port 8765
  service/
    Dockerfile              python:3.13-slim, non-root, healthcheck
    requirements.txt        exact pins
    requirements-dev.txt    pytest, httpx (TestClient)
    pyproject.toml          pytest config
    app/
      config.py             environment settings
      permissions.py        required permissions per feature + check
      immich.py             gateway over immichpy
      db.py                 SQLite schema and connection
      jobs.py               job store (queue, claim, progress, cancel, reorder, recover)
      settings_store.py     UI settings with validation and defaults
      auth.py               scrypt password, sessions, lockout
      worker.py             asyncio worker and pipeline registry
      pipelines/            collage.py, cinematic.py, highlight.py
      imaging/              quality, crop, layout, frames (tilted styles), depth, parallax
      vendor/immich_memories/  copied MIT modules + NOTICE.md
      suggestions.py        daily scan, events, ranking
      models.py, media.py   model download (pinned), ffmpeg encode
      main.py               FastAPI routes
      templates/, static/   layout C pages, tokens from the design library
    tests/                  pytest; fake_immich.py serves fake Immich v3.4.2 responses over local HTTP
```

Security in phase 1:
- Password hashed with scrypt. If `PHOTOS_PASSWORD` isn't set, a random one is generated on first start and printed once to the container log.
- 5 failed logins lock sign-in for 5 minutes.
- Session cookie is random, stored hashed, HttpOnly and SameSite=Strict. POSTs must come from the same origin.
- Strict CSP (`default-src 'self'`).
- The compose file binds to `127.0.0.1` unless `PHOTOS_BIND=0.0.0.0` is set for LAN access. Without HTTPS, the password crosses the LAN in clear text, so put a reverse proxy with TLS in front if the LAN isn't trusted.

## 6. Findings while building phase 1

- **immichpy 10.3.0 sends the multipart `metadata` field of `POST /assets` as a Python repr, not JSON.** The gateway leaves it out of the upload and writes the provenance with `PUT /assets/{id}/metadata` (JSON) straight after. Duplicate uploads skip that write.
- `GET /server/ping` and `/server/version` need no API key; `GET /api-keys/me` returns the key's own permission list.
- aioresponses 0.7.9 doesn't work with aiohttp 3.14, so the tests fake Immich over local HTTP with `aiohttp.web` instead.
- Starlette 1.10 prints a deprecation warning for using `httpx` in `TestClient`; it suggests a package named `httpx2`. Not switched: kept the well-known `httpx` until that package is checked.

## 7. Findings from the real-server tests (phases 2-5)

- **The test server runs Immich v3.3.1**, not v3.4.2. immichpy 10.3.0 couldn't read its responses, so `requirements.txt` pins **immichpy 9.1.0** (generated for v3.3.1). For a v3.4.x server switch to 10.3.0. In v3.3 asset `duration` is milliseconds and assets don't carry face boxes, so faces come from `GET /faces`.
- **Strict duplicate checks miss re-framed repeats.** A trip album had pairs 10-40 s apart at dHash 22-23. Same-scene grouping (time window plus loose dHash or colour histogram) removed 37 such shots from 80 scored.
- **Cinematic depth:** pulling whole face-based rectangles forward showed boxes in the depth map. Only pixels already near the person's depth are lifted now, feathered. Backward warping doubled faces at stronger motion; forward splatting (far to near) with inpainted gaps fixed it, at about 2 min per 4 s clip on this CPU.
- **Suggestion ranking:** "first event wins" produced three cinematic ideas from small events. Ideas are now scored by event size, kind and recency, with no kind taking more than half the daily slots.
- **Events split overnight.** A 3-day trip became one event per day with the 3 h gap rule. Consecutive days in the same city are now merged, and away runs across towns also become a trip idea (phase 8).
- **Settings saved by code skipped validation** (pydantic assignment). `SettingsStore.save` now re-validates.

## 8. Findings from phases 7-8

- **Smart search can't name events.** Immich v3.3 `POST /search/smart` returns the nearest photos with no score, so "birthday cake" always matches something. Events use birth dates, festival dates, OCR text and GPS instead.
- **Birth dates:** 26 of 142 named people in the test library have one, enough to find "Atharv's 9th birthday".
- **Home is where most days are.** The 5 km grid cell with the most distinct days is home; events over 40 km away become outings or trips, and festivals only count at home.
- **Matting:** BiRefNet-lite (MIT) took 25-42 s per photo on this CPU with no gain on people over MODNet (under 2 s). A 1024 px pass pulled in tree roots and leaves. Spikes over heads (a pole) are trimmed by column, but a piece that joins the outline sideways at body height stays, so a child's head band doesn't cut the shoulders of the adult behind.
- **Enlarging the subject doubles limbs.** The first "pop out of frame" scaled the people up over the original, which showed two sets of arms. People now stay at their own size and only the frame moves under them.
- **SQLite migration:** renaming `jobs` to `jobs_old` made SQLite re-point `job_log`'s foreign key at the dropped table. The migration now builds `jobs_new` and renames it, and databases already hit by this are repaired on start.
