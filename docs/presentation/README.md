# Presentation (2026-11-20)

The deck lives at https://claude.ai/artifact/ACihqxzHEn8RWYCWdFEbja (private until shared; Share > Export gives
PowerPoint or PDF). `deck/` is its source: one HTML file per slide, `deck.css`, and `deck.json` for the order.
Images and clips in the slides point at the deck's own uploads (`/_blob/...`), so the HTML does not render alone.

`TBOT_presentation.pptx` is the same 16 slides as a PowerPoint file, with the three clips embedded and the speaker
notes from each slide's `<aside>`. It is built by `python docs/presentation/build_pptx.py` (python-pptx, Pillow,
OpenCV) from `docs/media` and the clips here; fonts are Segoe UI in place of Inter / Space Grotesk.

Clips used in the slides, as files:

- `hit_IMG_0585.mp4`: the last phone clip of 2026-11-19, real time then 4x slow with a zoom on the clay. The clay
  breaks 0.12 s after the blast. No press was kept for this shot.
- `synced_160357_IMG_0565.mp4`: phone beside the gun camera, lined up on the bang (pairing good to 5 ms).
- `synced_160243_IMG_0575.mp4`: the same for press 160243, with the pairing corrected on 2026-11-20
  (see `software/docs/RANGE_MEDIA.md`).
- `cad_turntable.mp4`: the full assembly, one orbit.
