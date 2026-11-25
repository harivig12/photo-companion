# Photos companion service. CPU only.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PHOTOS_DATA_DIR=/data \
    PHOTOS_MUSIC_DIR=/music

# ffmpeg encodes the cinematic and highlight videos; fonts-dejavu gives collage captions a serif.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1000 photos && useradd --uid 1000 --gid photos --no-create-home --shell /usr/sbin/nologin photos

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

RUN mkdir -p /data /music && chown photos:photos /data
USER photos

EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=4).status == 200 else 1)"

CMD ["python", "-c", "from app.main import run; run()"]
