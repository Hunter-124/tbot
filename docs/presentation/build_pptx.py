"""Build TBOT_presentation.pptx from the deck source in deck/ (layout transcribed by hand from the slide HTML).

    python docs/presentation/build_pptx.py

Images come from docs/media and the clips in this folder; the deck's own uploads (/_blob/...) are the same files.
Needs python-pptx, Pillow and opencv-python. Fonts: Segoe UI (Windows) in place of Inter / Space Grotesk.
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import cv2
from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Pt

HERE = Path(__file__).resolve().parent
MEDIA = HERE.parent / "media"
OUT = HERE / "TBOT_presentation.pptx"
TMP = Path(tempfile.mkdtemp(prefix="tbot_pptx_"))

HEAD_FONT = "Segoe UI Semibold"
BODY_FONT = "Segoe UI"

BG, ALT, ACCENT = "12161B", "1A2028", "FF8A3D"
FG, MUTED, DIM, SOFT = "EEF1F4", "B4BEC9", "8D99A6", "D5DCE3"
CARD, CARD_ALT = "1B2129", "232B35"
L, W = 128, 1664  # content left edge and width, px on a 1920 x 1080 slide


def px(v: float) -> Emu:
    return Emu(int(round(v * 6350)))


def rgb(h: str) -> RGBColor:
    return RGBColor.from_string(h)


def lines(text: str, size: float, width: float, k: float = 0.46) -> int:
    per = max(1, int(width / (size * k)))
    n, cur = 1, 0
    for word in text.split():
        add = len(word) + (1 if cur else 0)
        if cur + add > per:
            n, cur = n + 1, len(word)
        else:
            cur += add
    return n


def text(slide, x, y, w, h, paras, size, color=FG, bold=False, font=BODY_FONT, align=PP_ALIGN.LEFT,
         spacing=1.15, anchor=MSO_ANCHOR.TOP, caps=False, tracking=0):
    """paras: str, or list of paragraphs; a paragraph is a str or a list of (text, {color, bold}) runs."""
    tb = slide.shapes.add_textbox(px(x), px(y), px(w), px(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.auto_size = None
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    if isinstance(paras, str):
        paras = [paras]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        runs = [(para, {})] if isinstance(para, str) else para
        for t, st in runs:
            r = p.add_run()
            r.text = t.upper() if caps else t
            f = r.font
            f.size = Pt(size * 0.5)
            f.name = st.get("font", font)
            f.bold = st.get("bold", bold)
            f.color.rgb = rgb(st.get("color", color))
            if tracking:
                r._r.get_or_add_rPr().set("spc", str(int(tracking * 50)))
    return tb


def rect(slide, x, y, w, h, fill, line=None, radius=None, shape=None):
    kind = shape or (MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE)
    s = slide.shapes.add_shape(kind, px(x), px(y), px(w), px(h))
    s.shadow.inherit = False
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = rgb(fill)
    if line:
        s.line.color.rgb = rgb(line)
        s.line.width = Pt(0.75)
    else:
        s.line.fill.background()
    if radius and kind == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = min(0.5, radius / min(w, h))
    return s


def fitted(path: Path, w, h, mode="cover", bg="0B0E12") -> Path:
    im = Image.open(path).convert("RGB")
    W_, H_ = int(w * 2), int(h * 2)
    if mode == "cover":
        s = max(W_ / im.width, H_ / im.height)
        im = im.resize((math.ceil(im.width * s), math.ceil(im.height * s)), Image.LANCZOS)
        l, t = (im.width - W_) // 2, (im.height - H_) // 2
        im = im.crop((l, t, l + W_, t + H_))
    else:
        s = min(W_ / im.width, H_ / im.height)
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
        canvas = Image.new("RGB", (W_, H_), tuple(int(bg[i:i + 2], 16) for i in (0, 2, 4)))
        canvas.paste(im, ((W_ - im.width) // 2, (H_ - im.height) // 2))
        im = canvas
    out = TMP / f"{path.stem}_{int(w)}x{int(h)}_{mode}.jpg"
    im.save(out, quality=90)
    return out


def pic(slide, path, x, y, w, h, mode="cover", bg="0B0E12"):
    return slide.shapes.add_picture(str(fitted(Path(path), w, h, mode, bg)), px(x), px(y), px(w), px(h))


def frame_of(video: Path, at: float = 0.0) -> Path:
    c = cv2.VideoCapture(str(video))
    n = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
    c.set(cv2.CAP_PROP_POS_FRAMES, int(n * at))
    ok, f = c.read()
    assert ok, video
    out = TMP / f"{video.stem}_poster.jpg"
    cv2.imwrite(str(out), f)
    return out


def movie(slide, video: Path, poster: Path, x, y, w, h, bg="0B0E12"):
    """The clip at its own aspect, centred in the box, on a plain backing."""
    c = cv2.VideoCapture(str(video))
    vw, vh = c.get(cv2.CAP_PROP_FRAME_WIDTH), c.get(cv2.CAP_PROP_FRAME_HEIGHT)
    s = min(w / vw, h / vh)
    mw, mh = vw * s, vh * s
    if abs(mw - w) > 1 or abs(mh - h) > 1:
        rect(slide, x, y, w, h, bg)
    poster_fit = fitted(poster, mw, mh, "cover")
    return slide.shapes.add_movie(str(video), px(x + (w - mw) / 2), px(y + (h - mh) / 2), px(mw), px(mh),
                                  poster_frame_image=str(poster_fit), mime_type="video/mp4")


# --------------------------------------------------------------------------------------------- building blocks
prs = Presentation()
prs.slide_width, prs.slide_height = px(1920), px(1080)
BLANK = prs.slide_layouts[6]


def new_slide(bg=BG, notes=""):
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = rgb(bg)
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    # a plain fade between slides, as in the HTML deck (p:transition sits before p:timing, which add_movie adds)
    tr = etree.SubElement(s._element, qn("p:transition"))
    tr.set("spd", "med")
    etree.SubElement(tr, qn("p:fade"))
    return s


def head(s, eyebrow, title, y=128, w=W, eyebrow_color=ACCENT, title_color=FG):
    text(s, L, y, w, 32, eyebrow, 24, eyebrow_color, bold=True, caps=True, tracking=3)
    text(s, L, y + 44, w, 80, title, 64, title_color, font=HEAD_FONT, spacing=1.0)
    return y + 44 + 80 + 36  # where content starts


def card(s, x, y, w, h, tag=None, title=None, body=None, bg=CARD):
    rect(s, x, y, w, h, bg, line="2E353E", radius=16)
    cy, iw = y + 36, w - 72
    if tag:
        text(s, x + 36, cy, iw, 32, tag, 24, ACCENT, bold=True, caps=True, tracking=2)
        cy += 32 + 14
    if title:
        n = lines(title, 36, iw, 0.42)
        text(s, x + 36, cy, iw, 44 * n, title, 36, FG, font=HEAD_FONT, spacing=1.05)
        cy += 44 * n + 14
    if body:
        n = lines(body, 26, iw)
        text(s, x + 36, cy, iw, 36 * n, body, 26, MUTED, spacing=1.2)


def stat_row(s, y, items, gap=32, title_size=36, cap_size=26):
    w = (W - gap * (len(items) - 1)) / len(items)
    for i, (t, c) in enumerate(items):
        x = L + i * (w + gap)
        text(s, x, y, w, 46, t, title_size, FG, font=HEAD_FONT)
        text(s, x, y + 58, w, 80, c, cap_size, MUTED, spacing=1.2)


def steps(s, x, y, w, items, size=28, gap=18):
    for i, item in enumerate(items):
        circ = rect(s, x, y, 48, 48, ACCENT, shape=MSO_SHAPE.OVAL)
        tf = circ.text_frame
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = str(i + 1)
        r.font.size, r.font.name, r.font.color.rgb = Pt(28 * 0.5), HEAD_FONT, rgb(BG)
        raw = item if isinstance(item, str) else "".join(t for t, _ in item)
        n = lines(raw, size, w - 68)
        hh = max(48, size * 1.4 * n)
        para = item if isinstance(item, str) else [item]
        text(s, x + 68, y + 6, w - 68, hh, para, size, SOFT, spacing=1.2)
        y += hh + gap
    return y


# --------------------------------------------------------------------------------------------- slides
# 1 cover
s = new_slide(notes="Open with the one-line idea: I swing the gun and hold a button, the machine picks the instant to fire. Everything bolted to the gun in this photo is 3D printed and clamps on; the gun itself is unmodified. This is the build as of this morning.")
pic(s, MEDIA / "build_2026-11-20_left.jpg", 0, 568, 1920, 512)
text(s, L, 128, W, 32, "Mechanical design project · Oct 25 – Nov 19, 2026", 24, ACCENT, bold=True, caps=True, tracking=3)
text(s, L, 172, W, 160, "TBOT", 150, FG, bold=True, font=HEAD_FONT, spacing=1.0)
text(s, L, 350, W, 60, "A shotgun that decides when to pull its own trigger.", 44, FG)
text(s, L, 428, W, 44, "Hunter Edmonds", 32, MUTED)

# 2 problem
s = new_slide(notes="The shot pattern is well under a metre wide at these ranges, and the clay moves several metres between deciding and hitting. So you cannot shoot where the clay is; you shoot where it will be. People learn that lead over thousands of shots. My approach was to measure every delay in the chain and let the machine do the arithmetic. 170 ms is the measured total on the second range day; 3.4 m is just 20 m/s times that.")
y = head(s, "The problem", "Hitting a clay is a timing problem") + 24
cw = (W - 96) / 3
for i, (num, cap) in enumerate([("20 m/s", "Roughly how fast a crossing clay moves"),
                                ("~170 ms", "From the decision to the shot arriving: see it, press the trigger, fly there"),
                                ("3.4 m", "How far the clay travels in that time")]):
    x = L + i * (cw + 48)
    text(s, x, y, cw, 130, num, 120, ACCENT, font=HEAD_FONT, spacing=1.0)
    text(s, x, y + 150, cw, 80, cap, 28, MUTED, spacing=1.2)
text(s, L, 1080 - 128 - 110, W, 110,
     "A shooter solves this by feel. TBOT solves it by measuring every delay and firing early by exactly that much.",
     44, FG, spacing=1.15)

# 3 concept
s = new_slide(notes="Keep this at the block-diagram level. Camera finds the target, the motion sensor separates the clay’s motion from my swing, the computer extrapolates both forward by the total delay, and the servo presses when the predicted shot covers the predicted clay. I still aim and swing; the machine only chooses when. The software was written with heavy AI assistance, so this talk is about the hardware and the measurements that make it work.")
y = head(s, "How it works", "The shooter swings, the machine picks the instant") + 16
cw, gap, ch = (W - 3 * 56) / 4, 56, 320
cards = [("1 · See", "Camera", "A small camera under the barrel looks along the bore and finds the clay dozens of times a second."),
         ("2 · Sense", "Motion sensor", "A gyro and accelerometer in the clamp measure how the gun itself is swinging."),
         ("3 · Predict", "Lead", "Where will the clay be, and where will the barrel point, once every delay has passed?"),
         ("4 · Press", "Servo", "When the two line up, a hobby servo presses the trigger. One shot per button press.")]
for i, (tag, t, b) in enumerate(cards):
    x = L + i * (cw + gap)
    card(s, x, y, cw, ch, tag, t, b)
    if i < 3:
        c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, px(x + cw + 8), px(y + ch / 2), px(x + cw + gap - 8), px(y + ch / 2))
        c.line.color.rgb, c.line.width = rgb(ACCENT), Pt(3)
        c.line._get_or_add_ln().append(c.line._get_or_add_ln().makeelement(qn("a:tailEnd"), {"type": "triangle"}))
text(s, L, 1080 - 128 - 45, W, 45, "The trigger can only move while the shooter holds an arming button.", 32, MUTED)

# 4 hardware
s = new_slide(ALT, notes="This is the gun as it sits today, right side. Up front: camera in the nose, a Raspberry Pi, a 3-cell battery and two voltage regulators, held by two worm-drive hose clamps seated in grooves. At the guard: a hobby servo beside the trigger with a printed cam reaching inside the guard, and the arming button. Hard requirement from day one: no drilling, no permanent changes to the gun. 8 parts in the front module, 4 in the trigger set.")
y = head(s, "The hardware", "Two printed assemblies, nothing on the gun modified")
pic(s, MEDIA / "build_2026-11-20_right.jpg", L, y, W, 444)
for px_, pw, t in [(330, 480, "Trigger set: servo, cam, arming button"),
                   (900, 680, "Front module: camera, computer, battery, motion sensor")]:
    b = rect(s, L + px_, y + 366, pw, 56, "12161B", line="5A6068", radius=12)
    b.fill.transparency = 0.14
    text(s, L + px_ + 20, y + 366, pw - 40, 56, t, 24, FG, bold=True, anchor=MSO_ANCHOR.MIDDLE)
stat_row(s, y + 444 + 40, [("12 printed parts", "PLA on an Ender 3, M3 screws into brass heat-set inserts"),
                           ("Clamp-on only", "Two hose clamps on the barrel, a saddle clamp on the trigger guard"),
                           ("23 days", "From measuring the gun to the first clay broken by the machine")])

# 5 scan
s = new_slide(notes="The gun has almost no flat or cylindrical reference surfaces where the parts needed to grip. Rather than iterate test prints against guesses, I scanned it. Printed boards with known square size give absolute scale, good to about two thirds of a millimetre. The mesh was then aligned to the same coordinate frame as the CAD, and the clamp surfaces were modelled directly against it. First full CAD was Oct 26; the scan and rebuild were Nov 5.")
y = head(s, "Reverse engineering", "Scanning the gun with a phone")
lw = W - 56 - 760
text(s, L, y, lw, 140, "The forend and trigger guard are compound curves that calipers cannot capture, so the clamps were modelled against a scan.", 32, MUTED, spacing=1.3)
steps(s, L, y + 170, lw, ["Walk-around phone video with printed calibration boards in frame",
                          "Camera positions solved from 407 frames",
                          "3D reconstruction (Gaussian splat), converted to a mesh at 0.54 mm resolution",
                          "True scale from the boards: 0.64 mm RMS",
                          "Mesh registered into the CAD coordinate frame"])
rx = L + lw + 56
pic(s, MEDIA / "splat_capture.jpg", rx, y, 760, 330)
pic(s, MEDIA / "scan_vs_cad.png", rx, y + 350, 760, 219, "contain")
text(s, rx, y + 589, 760, 34, "Scan (white) over the CAD body (red), in millimetres", 24, DIM)

# 6 front module
s = new_slide(ALT, notes="Packaging was the main constraint: it has to sit under the barrel without blocking the sight picture or my support hand, survive recoil, and keep the camera rigidly aligned to the bore. The camera sits in the nose looking along the barrel. The Pi is on one side, the battery and regulators on the other, with the wiring harness and camera ribbon modelled in the assembly. Total mass 231 grams.")
y = head(s, "Front module", "Camera, computer and battery in 231 g")
for i, (f, cap) in enumerate([("cad_open_L.jpg", "Gun-left: Raspberry Pi 3B, camera in the nose"),
                              ("cad_open_R.jpg", "Gun-right: 3-cell battery, voltage regulators, harness")]):
    x = L + i * (816 + 32)
    pic(s, MEDIA / f, x, y, 816, 400)
    text(s, x, y + 412, 816, 34, cap, 24, MUTED)
stat_row(s, y + 500, [("8 printed parts", "Hung from two hose clamps seated in grooves"),
                      ("Motion sensor in the clamp", "Rigid to the barrel, so it reads what the bore is doing"),
                      ("Lens locked against recoil", "A printed M12 × 0.5 split clamp nut holds focus")])

# 7 trigger
s = new_slide(notes="The trigger set went through the most change. The version I took to the first range day needed 161 milliseconds from command to the gun going off, and its printed servo mount broke after two shots. The mount was then redesigned twice: through-bolted with trapped nuts, then an anti-slide lock on the guard. Rev AH puts the servo body beside the guard with its output shaft inside it, and a printed cam presses the trigger’s lower hook. On the second range day the delay measured 62 to 70 ms, from recoil. Design revisions run G through AH, each logged with its reason.")
y = head(s, "Trigger mechanism", "Four generations in three weeks")
for i, (f, d, cap, mode) in enumerate([("03_first_print.jpg", "Oct 28", "First printed bracket", "cover"),
                                       ("04_trigger_prototype.jpg", "Nov 1", "Servo and arming button prototype", "cover"),
                                       ("07_trigger_clamp.jpg", "Nov 8", "Guard clamp, second generation", "cover"),
                                       ("cad_trigger.jpg", "Nov 18 · rev AH", "Servo beside the guard, cam on the trigger’s lower hook", "contain")]):
    x = L + i * (395 + 28)
    pic(s, MEDIA / f, x, y, 395, 360, mode)
    text(s, x, y + 372, 395, 32, d, 24, ACCENT, bold=True)
    text(s, x, y + 408, 395, 66, cap, 24, MUTED, spacing=1.2)
text(s, L, 1080 - 128 - 60, W, 60, [[("Command to ignition: ", {}), ("161 ms", {"color": DIM}), (" → ", {}),
                                     ("62–70 ms", {"color": ACCENT}), (" with rev AH.", {})]], 44, FG)

# 8 timing
s = new_slide(ALT, notes="This is the chain for a real shot on the second range day. The camera frame is about 55 ms old by the time a decision is made, the servo and trigger take about 66 ms to release the hammer and ignite, and the shot takes about 47 ms to reach a clay at 14 metres. Bars are to scale. The grey bar is the first trigger mechanism: 161 ms on its own. A known delay can be predicted across; an inconsistent one cannot, so the measured 4.9 ms jitter in the trigger is the number that really limits accuracy.")
y = head(s, "Timing budget", "Every millisecond between seeing and hitting") + 16
text(s, L, y, 330, 84, "Rev AH, Nov 19", 26, MUTED, anchor=MSO_ANCHOR.MIDDLE)
x = L + 330
for w_, col, v in [(430, "7FB7E6", "55 ms"), (516, ACCENT, "66 ms"), (368, "F2D06B", "47 ms")]:
    rect(s, x, y, w_, 84, col)
    text(s, x, y, w_, 84, v, 26, BG, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    x += w_
lx = L + 330
for col, k in [("7FB7E6", "Image age at the decision"), (ACCENT, "Servo, trigger and lock time"), ("F2D06B", "Shot flight to 14 m")]:
    lw_ = 40 + len(k) * 12.5
    text(s, lx, y + 112, lw_, 34, [[("■ ", {"color": col}), (k, {})]], 24, MUTED)
    lx += lw_ + 40
y2 = y + 112 + 34 + 28
text(s, L, y2, 330, 84, "First trigger, Nov 17", 26, MUTED, anchor=MSO_ANCHOR.MIDDLE)
rect(s, L + 330, y2, 1259, 84, "5B6673", radius=12)
text(s, L + 330, y2, 1259, 84, "161 ms for the trigger alone", 26, FG, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
stat_row(s, 1080 - 128 - 130, [("≈ 168 ms total", "The lead the machine has to predict across"),
                               ("Trigger is the biggest piece", "And the only one the mechanical design controls"),
                               ("± 5 ms repeatability", "Jitter matters more than the absolute delay")])

# 9 calibration
s = new_slide(notes="Each number the prediction depends on was measured with its own procedure. The boresight one is the nicest piece of metrology: a laser cartridge in the chamber is never perfectly centred, so I rotated it and took the centre of the circle it drew, which cancels the laser’s own error. A check at a second distance agreed to 0.067 degrees, against a shot pattern about 0.75 degrees in radius. Trigger delay is now measured from the recoil spike on every live shot.")
y = head(s, "Calibration", "The machine is only as good as its measurements")
tw = W - 48 - 400
rows = [("Quantity", "Value", "How it was measured"),
        ("Focal length", "± 0.3 %", "A sheet of letter paper at six taped distances, 20 to 45 ft"),
        ("Camera to bore", "−1.89° pitch, −1.67° yaw", "Bore laser, rotated in the chamber to cancel its own wobble"),
        ("Camera to motion sensor clock", "−6.31 ms", "Matching gyro rate against image motion over a 15 s swing"),
        ("Trigger delay", "62–70 ms", "Microphone on the bench, then recoil on the accelerometer in live fire")]
gf = s.shapes.add_table(len(rows), 3, px(L), px(y), px(tw), px(90 * len(rows)))
gf._element.graphic.graphicData.tbl.tblPr.find(qn("a:tableStyleId")).text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"  # no style, no grid
tbl = gf.table
for j, frac in enumerate((0.27, 0.25, 0.48)):
    tbl.columns[j].width = px(tw * frac)
tbl.first_row = False
tbl.horz_banding = False
for i, row in enumerate(rows):
    tbl.rows[i].height = px(60 if i == 0 else 92)
    for j, val in enumerate(row):
        c = tbl.cell(i, j)
        c.fill.background()
        c.margin_left, c.margin_right, c.margin_top, c.margin_bottom = px(0), px(20), px(10), px(10)
        c.vertical_anchor = MSO_ANCHOR.TOP
        p = c.text_frame.paragraphs[0]
        r = p.add_run()
        r.text = val
        r.font.size, r.font.name = Pt(26 * 0.5), BODY_FONT
        r.font.bold = i == 0
        r.font.color.rgb = rgb(ACCENT if i == 0 else SOFT)
text(s, L, y + 60 + 4 * 92 + 28, tw, 40, "Boresight checked at a second distance: within 0.067°.", 26, MUTED)
fx = L + tw + 48
pic(s, MEDIA / "10_boresight.jpg", fx, y, 400, 533)
text(s, fx, y + 545, 400, 66, "Nov 16: focal length and boresight in a hallway", 24, MUTED, spacing=1.2)

# 10 test ladder
s = new_slide(ALT, notes="I did not want the first test of anything to be with a loaded gun. Simulation checked the logic. The bench rig timed the whole chain, photon to trigger break, with no ammunition. The backyard gave real sky and real swing. Only then the range. And every range press saves everything the gun saw, which is how I could later audit exactly what happened on each shot.")
y = head(s, "Test ladder", "Prove each step before adding ammunition")
cw = (W - 96) / 4
for i, (tag, t, b) in enumerate([("1 · Simulation", "Synthetic clays", "Simulated targets flown through a model of the calibrated camera, judged against ground truth."),
                                 ("2 · Bench", "Screen in the loop", "The gun’s camera watches a moving target on a monitor while the real servo fires against a microphone."),
                                 ("3 · Backyard", "Real sky, dry gun", "Tracking birds by hand with the gun unloaded, the morning of the first range day."),
                                 ("4 · Range", "Live fire", "Every press records images and motion data, so each shot can be replayed afterwards.")]):
    card(s, L + i * (cw + 32), y, cw, 330, tag, t, b, bg=CARD_ALT)
pic(s, MEDIA / "hawk_sequence.jpg", L, 1080 - 128 - 44 - 260, W, 260)
text(s, L, 1080 - 128 - 32, W, 32, "The first moving target: a hawk over the yard, followed by hand for 8 seconds", 24, MUTED)

# 11 first range day
s = new_slide(ACCENT, notes="Twenty-three days after measuring the gun. Nine presses of the arming button, seven fire commands, two shots actually left the gun, and both broke their clay. Then the mount failed, which ended the day. The picture is the gun’s own camera at the instant the shot arrived: this one missed centre by 0.62 degrees against a 0.75 degree pattern, so a fringe hit. Honest footnote: the trigger delay in the settings was wrong by 60 ms that day, and both clays broke anyway.")
text(s, L, 128, 820, 32, "First live range day · Nov 17 · day 23", 24, "1A1206", bold=True, caps=True, tracking=3)
text(s, L, 200, 860, 160, "2 shots.", 150, "1A1206", bold=True, font=HEAD_FONT, spacing=1.0)
text(s, L, 370, 860, 160, "2 clays.", 150, "1A1206", bold=True, font=HEAD_FONT, spacing=1.0)
text(s, L, 1080 - 128 - 90, 760, 90, "Engaged at 13 to 30 m. Then the printed servo mount broke.", 32, "33240F", spacing=1.3)
pic(s, MEDIA / "shot_scored.png", 1000, 128, 792, 495, "contain")
text(s, 1000, 128 + 495 + 12, 792, 70, "The gun camera at the moment the shot arrived, 27.8 m. Circle: where the pattern went. Cross: where the clay was.", 24, "33240F", spacing=1.2)

# 12 hit
s = new_slide(notes="This is the last clip of the day and the clearest. Real time first, then four times slower with the clay enlarged in the corner. You can see it intact for about a tenth of a second after the muzzle blast, then it comes apart. The trigger is not in frame, but the red ready light on the front module is lit, meaning armed and live. Tally for the afternoon: the thrower counter reached 16 and at least 5 broke; on the others the gun did not fire, the computer was off, or we missed. One caveat to say out loud: the gun’s own recording of this particular shot was not kept. That afternoon the computer kept losing power at the shot, before it could save. That power problem is on the lessons slide.")
y = head(s, "Second range day · Nov 19", "The machine breaks a clay", w=1100)
text(s, L, y, 1100, 90, "Rebuilt trigger (rev AH). I hold the button and follow the clay; the servo fires on its own.", 32, MUTED, spacing=1.3)
y = steps(s, L, y + 118, 1100, ["Gun held level on the trap’s flight line, button held",
                                "Clay launches and climbs into the camera’s view",
                                "Shot fires about 0.9 s after launch",
                                "Clay breaks about 0.12 s after the muzzle blast"])
text(s, L, y + 10, 1100, 90, [[("At least 5 clays broken", {"color": ACCENT, "bold": True}),
                               (" of about 16 thrown that afternoon. The rest: no shot, computer powered off, or a miss.", {})]], 32, MUTED, spacing=1.3)
text(s, L, y + 128, 1100, 70, "Phone video, real time then 4× slow with a zoom on the clay. The trigger is out of frame; the red light on the front module shows the system armed and live.", 24, DIM, spacing=1.2)
movie(s, HERE / "hit_IMG_0585.mp4", HERE / "hit_IMG_0585_poster.jpg", 1329, 128, 463, 824)

# 13 gun view
s = new_slide(ALT, notes="Here the gun’s recording did survive. Phone on the left, the gun’s own view on the right with what it was tracking drawn on top. When the frame turns red, that is the command to the servo; the gun actually goes off 66 ms later, which is the delay from the timing slide showing up in real data. The two videos are synchronised by the sound of the shot and the recoil spike, and the pairing is confirmed by the spacing between two consecutive shots matching to 5 ms.")
y = head(s, "Second range day · Nov 19", "What the gun sees when it fires")
v = HERE / "synced_160357_IMG_0565.mp4"
movie(s, v, frame_of(v, 0.33), L, y, 1300, 602)
text(s, L + 1300 + 36, y, W - 1336, 330,
     [[("Left: ", {"bold": True, "color": FG}), ("phone video.", {})],
      [("Right: ", {"bold": True, "color": FG}), ("the gun’s own camera.", {})],
      [("Yellow rings: ", {"color": "F2D06B"}), ("objects it is tracking.", {})],
      [("Red frame: ", {"color": ACCENT}), ("the fire command. Recoil follows 66 ms later.", {})]], 26, MUTED, spacing=1.6)
text(s, L, 1080 - 64 - 66, W, 66, "Shot at 24 m. The two videos are lined up on the bang: this shot and the one before it are 24.17 s apart on the phone and 24.177 s apart in the gun’s data.", 24, DIM, spacing=1.2)

# 14 lessons
s = new_slide(notes="The honest slide. Printed PLA against 12-gauge recoil: the first mount lasted two shots. On day two the mechanism held, but the computer rebooted at almost every shot. Its supply flags were clean before each cut, so it is not a brown-out; it is a contact opening, and that is a bench job with the gun unloaded. The delay error is the lesson I would repeat to anyone: I had the right measurement and discounted it. And the double: the data shows no second command was sent, the servo had been told to return, and the cause is still open between the servo signal at shutdown, the supply bouncing, and my own finger.")
y = head(s, "What broke", "Recoil found every weak point")
cw, ch = (W - 32) / 2, 300
for i, (tag, t, b) in enumerate([("Structure", "Servo mount cracked after two shots", "Redesigned twice: through-bolted with trapped nuts, then an anti-slide lock on the guard."),
                                 ("Power", "Computer loses power at the shot", "Clean cuts with no voltage sag: something in the battery path opens under recoil. Most shots on day two were never recorded."),
                                 ("Measurement", "Trigger delay was wrong by 60 ms", "A bench microphone read 161 ms. I trusted a video estimate of 95 ms instead. The microphone was right."),
                                 ("Actuation", "One double fire", "A second shot 0.36 s after the first with no command. Servo shutdown sequence changed; root cause not proven.")]):
    card(s, L + (i % 2) * (cw + 32), y + (i // 2) * (ch + 32), cw, ch, tag, t, b)

# 15 safety
s = new_slide(ALT, notes="The machine never fires on its own initiative. The arming button is a dead-man switch: normally open with a pull-up, so any wiring fault fails to the safe state. I am upfront about the gap: all five layers are software, and the double fire is exactly why a hardware cut in the servo supply belongs on the next-steps list.")
y = head(s, "Safety", "A person holds the button for every shot")
lw = W - 56 - 520
B = {"bold": True, "color": FG}
steps(s, L, y, lw, [[("Dry run by default. ", B), ("Without an explicit live setting the servo never moves.", {})],
                    [("Button interlock at the actuator. ", B), ("No button, no trigger press.", {})],
                    [("Release means rest. ", B), ("A cut wire or loose connector reads as not armed.", {})],
                    [("One shot per press.", B)],
                    [("Watchdog. ", B), ("No fresh image for 250 ms and the servo returns to rest.", {})]], gap=28)
rx = L + lw + 56
card(s, rx, y, 520, 250, "Known gap", None, "These are software layers. There is no hardware cut in the servo supply yet; unplugging the battery is what makes it safe.", bg=CARD_ALT)
card(s, rx, y + 274, 520, 210, "Where it runs", None, "Clay targets on a range, under range rules. All bench work with the gun unloaded.", bg=CARD_ALT)

# 16 next
s = new_slide(notes="Three things, in order. The power path first, because without it I cannot even collect data. Then the hardware interlock. Then volume: the trigger delay calibration rests on three live shots, and I do not yet have an honest hit percentage. Close on the CAD turntable and take questions.")
y = head(s, "Next steps", "From “it works” to “it works every time”")
cw = (W - 64) / 3
for i, (tag, t, b) in enumerate([("1 · Power", "Recoil-proof the supply", "Find the contact that opens: tap test on the bench with the ready light as the indicator."),
                                 ("2 · Safety", "Hardware cut for the servo", "Put the arming button in the servo’s supply, not only in software."),
                                 ("3 · Data", "More shots", "Three live shots set the trigger delay. More are needed to pin down its repeatability and a real hit rate.")]):
    card(s, L + i * (cw + 32), y, cw, 290, tag, t, b)
v = HERE / "cad_turntable.mp4"
movie(s, v, frame_of(v, 0.0), L, 1080 - 128 - 340, W, 340, bg="30352F")

prs.save(OUT)
print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, {len(prs.slides)} slides)")
