# Immich vs Google Photos: gaps, rules and references

Research date: 24 Nov 2026. Immich version compared: **v3.4.2** (latest release; v3.4 released Nov 2026, v4 is being planned). Content from sources is paraphrased; links are at the end of each section.

Legend for rules: **[G]** documented by Google, **[R]** taken from an existing open-source project, **[I]** inferred by us (Google does not publish it).

## 1. Feature gap matrix

| Google Photos feature | Immich v3.4.2 | Companion can add it? | Target? |
|---|---|---|---|
| Collage (manual and auto-suggested) | Missing; requested in discussions #5229 and #16976 | Yes | **Yes** |
| Cinematic photo (3D parallax from one photo) | Missing | Yes, CPU depth model | **Yes** |
| Depth map (internal step of cinematic) | Missing | Yes; also uploaded as a PNG | **Yes** |
| Highlight video with music | Missing (#22790); third-party immich-memories exists | Yes | **Yes** |
| Animation (GIF from a burst) | Missing | Yes | Later |
| Subject cut-out / sticker ("pop out") | Missing | Yes, CPU segmentation (rembg / U²-Net) | Documented only |
| Floating / picture-in-picture viewer ("pop out") | Missing | No: lives inside Immich's own web and mobile UI; needs an upstream change or a fork | Documented only |
| Magic Eraser, Unblur, Portrait blur / light | Missing; the editor (web v2.5, mobile v3.0) does crop, rotate, adjust | Eraser (LaMa) and blur on CPU; generative edits need a GPU or cloud | No |
| Moods, Photo to video, Photo / Video remix, Magic Editor, Reimagine | Missing | Generative: needs GPU or cloud, ruled out | No |
| Ask Photos (conversational answers) | Partial: CLIP smart search + OCR, no summarised answers | Needs a local LLM; too heavy for CPU-only | No |
| Memories | Present: "on this day" and birthdays (v3.3) | Creations can be added to the Memories strip | Used |
| Shared albums, partner sharing, people sharing, map, faces, locked folder, OCR, stacks, duplicates, workflows | Present | Not needed | — |

Notes:
- Google's Create tab lists animations, cinematic photos, collage, highlight videos, moods, photo to video, photo remix, video remix and wardrobe. Animations and cinematic photos can't be edited after creation. [G]
- Editor feature names (Magic Eraser, Unblur, Portrait light, Best Take) are from general knowledge of the product and were not re-checked on 24 Nov 2026.

Sources: [Google: Make creations](https://support.google.com/photos/answer/6128862?co=GENIE.Platform%3DAndroid&hl=en) · [Immich v3.0.0 release](https://immich.app/blog/v3.0.0-release) · [Immich releases](https://github.com/immich-app/immich/releases) · [Immich October 2026 recap](https://immich.app/blog/2026-october-recap) · [XDA: features missed after switching](https://www.xda-developers.com/immich-vs-google-photos-features-i-miss/) · [Collage request #5229](https://github.com/immich-app/immich/discussions/5229) · [Collage request #16976](https://github.com/immich-app/immich/discussions/16976) · [Composite video request #22790](https://github.com/immich-app/immich/discussions/22790)

## 2. "Image pop out": the three readings

1. **3D pop (cinematic photo)**: covered in section 4. Target.
2. **Subject cut-out / sticker**: a segmentation model separates the subject; output is a transparent PNG. Feasible on CPU with rembg (MIT) and U²-Net (Apache-2.0). Not a target now.
3. **Floating / picture-in-picture viewer**: a viewer behaviour in Immich's own clients. A separate service can't add it; it would need a pull request to Immich or a fork.

## 3. Collage

**Google's rules**
- Start from selected photos (up to 6 on Android/iOS, up to 9 on web) or from a template, then fill the slots. [G]
- Templates are grouped by theme and filtered by photo count; you can switch photo-count layouts inside the editor. [G]
- Grid templates let you change border width, corner radius and colour. [G]
- You can't move slots, layer photos or drag the borders inside a grid template. [G]
- Android needs 3 GB RAM and Android 8.0+. [G]
- Saved collages are found by searching "Collages". [G]
- Auto collages are suggested from recent events in "New for you" / Memories. Forum users report auto ones with 9 or 12 photos, more than the manual limit. [G for "recent events"; forum report for the counts]
- Google doesn't publish how it picks events or photos.

**Our rules** [I]
- Layout from photo count plus orientation mix:
  - 2 portraits → side by side; 2 landscapes → stacked; mixed → large + small.
  - 3 → one large slot + two small.
  - 4 → 2×2 grid. 5–6 → 2 large + 3–4 small. 7–9 → 3×3 or one large + small ring.
  - Mixed aspect ratios with no clean grid → BRIC layout (stefanwayon/collage, MIT).
- The big slot gets the photo with the most named faces and the best sharpness.
- Crops keep every face box (from Immich) inside its slot; without faces, crop around the centre.
- Canvas ratio 1:1, 4:5, 9:16 or 16:9. Borders: width, radius, colour (same controls as Google).
- Leave out: blurry photos (low variance of the Laplacian), near-duplicates (keep the sharpest), screenshots and documents, archived / locked / trashed assets, very low resolution (warn instead of failing on manual jobs).
- Auto-suggestion:
  - Event = photos with no gap longer than 3 hours (setting) and, when GPS exists, within about 1 km.
  - Suggest only if at least 6 good photos remain (setting) and none were used in an earlier collage (`creation:collage` tag / stored source IDs).
  - Up to 12 photos for auto collages, 9 for manual. At most 3 suggestions per day (setting).

## 4. Cinematic photo and depth map

**Google's method** (Google Research blog)
- A neural network estimates depth from one ordinary photo, using cues like object size, perspective and defocus. [G]
- The depth is relative, not real distance: different training sets disagree on scale and shift, so the output is normalised at inference time. [G]
- Errors at person edges cause the worst artefacts. They clean edges with median filtering and use a person-segmentation mask to pull mis-placed person pixels forward. [G]
- The scene becomes a mesh (the photo extruded onto the depth map). Moving the virtual camera exposes "stretchy" polygons where depth jumps. [G]
- The camera path is optimised per photo with a loss that punishes clusters of stretch artefacts, weighting head and body regions above the background, so any artefacts land behind the subject. [G]
- A saliency model frames the output so the important parts stay in view while every frame stays filled; the field of view may shrink to achieve this. [G]
- Output is a short video; auto ones appear in Memories; they can't be edited. [G]
- Google doesn't publish which photos it chooses.

**Our rules** [I]
- Eligible photos: a clear subject separated from the background (person, pet, object, or scenery with real depth), depth spread above a threshold (default 0.20 on a 0–1 scale), sharp, at least 1080 px on the short side. Skip screenshots, documents and flat scenes.
- Depth: Depth Anything V2 **Small** (Apache-2.0) on CPU via ONNX Runtime. Normalise to 0–1. The Base and Large models are CC-BY-NC, so they're not used.
- Edge cleanup: median filter, then a person/subject mask from Immich face boxes plus segmentation; pull the subject's depth forward.
- Render: 2.5D displacement with OpenCV (zoom, dolly, circle presets). "Auto" tries each preset and keeps the one with the lowest stretch score, counting connected stretch pixels and weighting the subject more, as in Google's method.
- Framing: crop so every frame is filled and the subject stays in view.
- Output: H.264 MP4, 4 s at 30 fps, 1080p (settings). The depth map is uploaded as a 16-bit greyscale PNG and stacked with the source photo.

Sources: [Google Research: the technology behind cinematic photos](https://research.google/blog/the-technology-behind-cinematic-photos/) · [Google: Make creations](https://support.google.com/photos/answer/6128862?co=GENIE.Platform%3DAndroid&hl=en)

## 5. Highlight video with music

**Google's rules**
- Choose people, places, themes or dates and the video is assembled automatically, or hand-pick the items. [G]
- Third-party guides add detail Google's own page doesn't: a 2–50 item limit when picking by hand, themed prompts ("Year in review"), trim and reorder clips, set photo duration, music from mood categories or your own MP3/AAC, a balance between clip audio and music, 16:9 / 9:16 / 1:1 output, and the soundtrack trimmed to fit. [third-party, not Google]

**immich-memories' rules** [R] (MIT)
- Score clips on faces, motion, camera stability and audio, then keep the best ~5 seconds of each clip.
- Always chronological; favourites and named people win close calls.
- Photos get Ken Burns or face-aware panning, with a blurred fill when the photo doesn't fill the frame.
- Music from a bundled set or your own folder, lowered when people speak.
- Live Photos count as motion only if the clip lasts at least 2.5 s and actually moves.

**Our rules** [I, building on R]
- Source: an album, a person, a date range or an Immich smart search.
- Leave out blurry shots, duplicates and screenshots. Needs at least 8 usable items.
- Length 30–90 s (setting); about 3 s per photo; best 3–5 s per video clip.
- Chronological order; favourites first in close calls.
- Cuts land on the music's beat. Music only from the mounted `/music` folder, so there's no licensing risk. Fade in and out.
- Aspect 16:9, 9:16 or 1:1.

Sources: [Google: Make creations](https://support.google.com/photos/answer/6128862?co=GENIE.Platform%3DDesktop&hl=en-GB) · [Digitash: highlight videos guide](https://digitash.com/google/photos/create-edit-custom-highlight-videos-google-photos/) · [immich-memories](https://github.com/sam-dumont/immich-memories)

## 6. Existing projects to reuse or study

| Project | Licence | Use |
|---|---|---|
| [sam-dumont/immich-memories](https://github.com/sam-dumont/immich-memories) v1.0.0 | MIT | Copy `photos`, `audio` and part of `processing` into `vendor/` for the highlight-video render (see plan) |
| [immichpy](https://pypi.org/project/immichpy/) 10.3.0 | see PyPI | Typed async Immich client; targets Immich v3.4.2 |
| [immich-power-tools](https://github.com/immich-power-tools/immich-power-tools) | AGPL-3.0 | Reference only (companion UI patterns); no code copied |
| [BrokenSource/DepthFlow](https://github.com/BrokenSource/DepthFlow) | AGPL-3.0 | Reference only (parallax presets); needs OpenGL; no code copied |
| [stefanwayon/collage](https://github.com/stefanwayon/collage) | MIT | BRIC layout algorithm for mixed aspect ratios |
| [adrienverge/PhotoCollage](https://github.com/adrienverge/PhotoCollage) | GPL-2.0+ | Reference only |
| Collage fork in [#16976](https://github.com/immich-app/immich/discussions/16976) | not upstream | Ideas: templates by count, face-focused crop, low-quality warning, `collage` tag |

Licences checked through the GitHub API on 24 Nov 2026. Copying or importing the AGPL projects would make this project AGPL, so they're reference only.
