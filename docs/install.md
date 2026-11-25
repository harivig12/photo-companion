# Install with Docker or Portainer

The companion is one container. It talks to your Immich server over the network and keeps its own small database in a volume. It runs on the CPU only, no GPU needed. Renders are memory-hungry (the cinematic and pop-out models plus full-size photos), so give it a few GB of free RAM.

## Before you start

1. **Immich** must be reachable from the machine that runs the container. Note its address as the container sees it, for example `http://192.168.1.10:2283`. `localhost` won't work: inside the container that means the container itself.
2. **An Immich API key.** In Immich go to **Account Settings → API Keys → New API key**. Read-only is enough for **test mode** (results are saved to a folder, nothing is written to Immich). For real use, grant all of these:

   | Purpose | Permissions |
   |---|---|
   | Reading photos | `asset.read`, `asset.view`, `asset.download`, `person.read`, `face.read`, `album.read`, `tag.read` |
   | Writing results | `asset.upload`, `asset.update`, `album.create`, `albumAsset.create`, `tag.create`, `tag.asset`, `stack.create`, `memory.create`, `memoryAsset.create` |

   After the first login, **Settings → Check permissions** lists anything missing.
3. **Your Immich version.** The image is built for Immich **v3.3.x**. For v3.4.x change `immichpy==9.1.0` to `10.3.0` in `service/requirements.txt` and build the image yourself (option B below).
4. **Music (optional).** A folder of tracks you have the rights to. Highlight videos are cut on the beat of these, so add some before making highlights.

## Settings

| Variable | Default | What it does |
|---|---|---|
| `IMMICH_URL` | required | Immich address as seen from the container |
| `IMMICH_API_KEY` | required | The API key. Keep it out of Git |
| `PHOTOS_PASSWORD` | generated | First UI password, 12+ characters. If empty, one is generated and printed once in the log |
| `PHOTOS_BIND` | `127.0.0.1` | `127.0.0.1` = this machine only. `0.0.0.0` = reachable from the LAN |
| `PHOTOS_OUTPUT_MODE` | `immich` | `folder` = test mode, results are saved under `/data/output` and nothing is written to Immich |
| `PHOTOS_SECURE_COOKIE` | `false` | Set `true` only behind a reverse proxy that serves HTTPS |
| `PHOTOS_SESSION_HOURS` | `12` | Login lifetime |
| `PHOTOS_FFMPEG` | `ffmpeg` | Already on the image's PATH |

Everything else (collage styles, pop-out styles, housekeeping, suggestions) is set in the web UI under **Settings** and stored in the database.

## A. Docker Compose, building from source

Good if you want to change the code or need the Immich v3.4 client.

```sh
git clone https://github.com/harivig12/photos-companion.git
cd photos-companion
cp .env.example .env          # set IMMICH_URL and IMMICH_API_KEY
mkdir -p data music           # put your music tracks in ./music
sudo chown 1000:1000 data     # the container runs as UID 1000
docker compose up -d --build
```

The repository is private, so `git clone` asks for your GitHub login. Use a personal access token with the `repo` scope as the password.

Open `http://<server>:8765`. If `PHOTOS_PASSWORD` was empty:

```sh
docker compose logs photos | grep "Generated UI password"
```

## B. Docker Compose, prebuilt image

Nothing is built on your server. A GitHub Actions workflow publishes `ghcr.io/harivig12/photos-companion` for `linux/amd64` on every push to `main`, tagged `latest`, the short commit, and `1.2.3` for a `v1.2.3` tag.

The repository is private, so the image is private too. Log in once on the server with a token that has **`read:packages`** (GitHub → Settings → Developer settings → Personal access tokens (classic)):

```sh
echo "<token>" | docker login ghcr.io -u harivig12 --password-stdin
```

Then save [`docker-compose.portainer.yml`](../docker-compose.portainer.yml) and a `.env` file in the same folder:

```sh
IMMICH_URL=http://192.168.1.10:2283
IMMICH_API_KEY=your-key
PHOTOS_BIND=0.0.0.0
MUSIC_DIR=/path/to/music
```

```sh
docker compose -f docker-compose.portainer.yml up -d
```

To make the package pullable without a login, open the repository's **Packages → photos-companion → Package settings** and change its visibility. Only do that if you're fine with the image being public (it contains no keys or photos).

## C. Portainer

### C1. Stack from the web editor (prebuilt image)

1. **Registries → Add registry → Custom registry** (only because the image is private):
   - Name: `ghcr`, URL: `ghcr.io`
   - Authentication on, username `harivig12`, password = a token with `read:packages`.
2. **Stacks → Add stack**, name it `photos`, choose **Web editor**, and paste the contents of [`docker-compose.portainer.yml`](../docker-compose.portainer.yml).
3. Under **Environment variables** add `IMMICH_URL` and `IMMICH_API_KEY`. Add `PHOTOS_BIND=0.0.0.0` to reach the UI from other devices, and `MUSIC_DIR` (a folder on the Docker host) if you have music.
4. **Deploy the stack.** Open `http://<server>:8765`.
5. If `PHOTOS_PASSWORD` was empty, open the `photos` container → **Logs** and look for `Generated UI password`.

### C2. Stack from the Git repository (builds on the server)

1. **Stacks → Add stack → Repository**.
2. Repository URL `https://github.com/harivig12/photos-companion`, reference `refs/heads/main`.
3. Turn on **Authentication**: username `harivig12`, password = a personal access token with the `repo` scope.
4. Compose path: `docker-compose.yml`. Add the same environment variables as above. The stack reads `./data` and `./music` next to the compose file, which Portainer places under its own data folder, so for a stable location edit the two volumes in `docker-compose.yml` to absolute paths on your host.
5. **Deploy the stack.** The first build takes a few minutes.

To pick up new code later, open the stack and use **Pull and redeploy**.

## First run

1. Log in with the password from `PHOTOS_PASSWORD` or the log.
2. **Settings → Check permissions** (any missing permission is listed).
3. Try a **test mode** run first: set `PHOTOS_OUTPUT_MODE=folder`, create a collage from an album, and look in `/data/output/<job>-collage/` (inside the volume `photos-data`). The UI shows a "Test mode" banner.
4. Switch to `immich` when you're happy with the results.

The first cinematic job downloads the depth model (about 100 MB) and the first pop-out job downloads the cut-out model (25 MB) into `/data/models`. The container needs internet access for this once. Both are pinned and checked by SHA-256. A cinematic render takes about 2 minutes per 4 s clip, a pop-out photo with all 5 styles about 1 minute.

## Updating

- **Prebuilt image:** `docker compose -f docker-compose.portainer.yml pull && docker compose -f docker-compose.portainer.yml up -d`, or in Portainer open the stack and use **Pull and redeploy** with *Re-pull image* on.
- **From source:** `git pull && docker compose up -d --build`.

The database upgrades itself on start. Settings, presets and job history are kept.

## Backup and removal

All state is in `/data` (the `photos-data` volume): the database, downloaded models and test-mode output. To back up the settings and presets, copy `photos.sqlite3` from it while the container is stopped. Housekeeping removes finished jobs and test-mode output after 30 days by default (**Settings → Housekeeping**). It never deletes anything in Immich.

Remove the stack, then the volume (`docker volume rm photos_photos-data`) to start fresh.

## Reaching it from other devices and HTTPS

The UI has a password and a lock-out after repeated failures, but the traffic is plain HTTP. On a trusted home LAN that's often fine. Otherwise put a reverse proxy with TLS in front (Nginx Proxy Manager, Traefik, Caddy) and set `PHOTOS_SECURE_COOKIE=true`.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Container restarts at once, log says a setting is missing | `IMMICH_URL` or `IMMICH_API_KEY` is empty |
| "Cannot reach Immich" | `IMMICH_URL` uses `localhost`, or the server isn't on the same Docker network or LAN route |
| Settings page says the key lacks permissions | Give the key the permissions listed above (edit it in Immich or create a new one), then restart the container |
| `permission denied` on `/data` | A bind-mounted `./data` isn't owned by UID 1000: `sudo chown 1000:1000 data` |
| `pull access denied` for `ghcr.io/...` | The registry login is missing or the token lacks `read:packages` |
| A response from Immich fails to parse | The client doesn't match your Immich version: see "Before you start", point 3 |
| Cinematic or pop-out jobs fail on the first run | No internet access for the one-time model download |
| Highlight videos can't find music | `MUSIC_DIR` is empty or the path is wrong |
