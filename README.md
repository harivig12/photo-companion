# Photos companion for Immich

A separate Docker service that adds Google-Photos-style creations to an existing Immich server through Immich's API:

- **Collage**: 2–12 photos as one JPEG. Styles: tilted prints, polaroid scatter, gallery fan (the default picks one of these three at random) or a flat grid.
- **Cinematic photo**: one photo becomes a 3D parallax MP4, plus its 16-bit depth map PNG.
- **Highlight video**: photos and clips become an MP4 cut on the beat of a track from your music folder, with an animated title card (fade-up, typewriter or letter-by-letter).
- **Pop-out photo**: the people in a photo are cut out (MODNet, Apache-2.0) and given an effect. Pick any of: colour splash (background in grey), spotlight, portrait blur, pop out of a tilted frame, or a die-cut sticker.
- **Suggestions**: a daily scan groups new photos into events and proposes creations. You approve them, or they run automatically. Events are named by type: birthday, festival (Deepavali, Pongal, Independence Day and others), school function, wedding, outing, trip, get-together and "through the years".

Results go back into Immich: a "Creations" album, an album per event (optional), a `creation:*` tag, a stack with the source photo, and a description. Birthday events also get a birthday memory in the Memories strip; other events don't add memories, so the strip isn't filled with more "N years ago" cards. Everything runs locally on the CPU; nothing is sent to cloud services.

**Editing and presets:** on the Jobs page, **Edit text** on a finished job re-makes it with a new title or caption from the same photos, music and style. The old version is hidden in the companion. **Save as preset** keeps the look (style, ratio, caption, title style) under a name for reuse on **Create**; presets never store photos.

**Housekeeping:** finished jobs, their logs and test-mode output older than **Settings → Housekeeping → keep for N days** (default 30, 0 = forever) are removed every hour. Nothing in Immich is ever deleted.

| Document | What's in it |
|---|---|
| `docs/install.md` | Install with Docker Compose or Portainer (prebuilt image or build from source), settings, updating, troubleshooting |
| `docs/research/immich-vs-google-photos.md` | Feature gaps, Google's rules for each feature, our rules, existing projects and licences |
| `docs/plan.md` | Architecture, API key permissions, phases and status, findings from the real-server tests |
| `docs/mockups/2026-11-24-companion-ui.html` | UI mockup (layout C was chosen) |
| `service/app/vendor/immich_memories/NOTICE.md` | Code copied from immich-memories (MIT) and what was changed |

## Run it

Full steps for Docker Compose and Portainer, including the prebuilt image, are in [`docs/install.md`](docs/install.md). The short version:

Needs Docker Compose v2 and an Immich server the container can reach.

1. In Immich, go to **Account Settings → API Keys → New API key** and grant:
   `asset.read`, `asset.view`, `asset.download`, `asset.upload`, `asset.update`, `album.read`, `album.create`, `albumAsset.create`, `person.read`, `face.read`, `tag.read`, `tag.create`, `tag.asset`, `stack.create`, `memory.create`, `memoryAsset.create`.
2. Match the client to your server version in `service/requirements.txt`: immichpy **9.1.0** for Immich v3.3.x (pinned), **10.3.0** for v3.4.x.
3. Configure and start:
   ```sh
   cp .env.example .env        # set IMMICH_URL and IMMICH_API_KEY
   mkdir -p data music         # put music tracks you have the rights to in ./music
   sudo chown 1000:1000 data   # the container runs as UID 1000
   docker compose up -d --build
   ```
4. Open <http://localhost:8765>. If you left `PHOTOS_PASSWORD` empty, the generated password is printed once:
   `docker compose logs photos | grep "Generated UI password"`.
5. **Settings → Check permissions** confirms the key has everything it needs.

The first cinematic job downloads the depth model (Depth Anything V2 Small, Apache-2.0, about 100 MB) into `data/models`, and the first pop-out job downloads MODNet (Apache-2.0, 25 MB). Both are pinned to a commit and checked by SHA-256.

**Test mode:** set `PHOTOS_OUTPUT_MODE=folder` to save every result under `data/output/<job>-<kind>/` with a `job.json`, and never write to Immich. The UI shows a "Test mode" banner.

### Reaching it from other devices

The port is bound to `127.0.0.1` by default. Set `PHOTOS_BIND=0.0.0.0` in `.env` to open it to the LAN. The UI is protected by its password, but traffic is plain HTTP, so on an untrusted network put a reverse proxy with TLS in front and set `PHOTOS_SECURE_COOKIE=true`.

### Forgot the password

```sh
docker compose exec photos python -m app.reset_password
```

## Command line (inside the container or a dev venv)

```sh
python -m app.run_job collage --album <id> [--style random|prints|scatter|fan|grid] [--ratio 4:5] [--caption "Goa"]
python -m app.run_job cinematic --assets <photo-id>
python -m app.run_job highlight --album <id> | --person <person-id> [--from 2026-01-01 --to 2026-06-30]
python -m app.run_job popout --assets <photo-id>        # styles come from Settings → Pop-out
python -m app.scan_once [--days 90] [--run collage]     # one suggestion scan; optionally run the first idea
python -m app.album_stats                               # photos / videos per album (read-only)
```

## Develop and test

```sh
cd service
python -m venv .venv
.venv/bin/pip install -r requirements-dev.txt      # Windows: .venv\Scripts\pip
PHOTOS_FFMPEG=/path/to/ffmpeg .venv/bin/python -m pytest
```

The tests never touch a real Immich: `tests/fake_immich.py` serves Immich v3.3 responses over local HTTP. Video tests are skipped when ffmpeg isn't found.

## Known limits

- **Untested writes:** Immich write-back (upload, albums, tag, stack, description, birthday memory) has only been tested against the fake server. On a real server, still check:
  - how the birthday memory reads in the Memories strip;
  - whether a stack can mix the cinematic MP4 with photos.
- **Editing keeps the old upload:** without `asset.delete` the old version stays in Immich; delete it there by hand if you don't want both.
- **Highlight sound:** highlight videos use the music only. The clips' own sound and lowering the music under speech aren't done yet.
- **Festival dates:** moving festivals come from tables in `service/app/events.py` that run out (Deepavali and Pongal in 2031, others in 2027–2028). Extend them yearly. Smart search isn't used for events: Immich v3.3 returns no match scores, so there's no way to tell a cake from a cushion.
- **Birthdays** need a birth date on the person in Immich.
- **Speed:** cinematic renders take about 2 minutes per 4 s clip, and a pop-out photo with all 5 styles about 1 minute, on a typical CPU.

## Licence

MIT. Files under `service/app/vendor/immich_memories/` are from immich-memories v1.0.0 (MIT) and keep their licence; see `NOTICE.md`. AGPL projects (DepthFlow, immich-power-tools) were used as references only.
