# TBOT software — architecture contract

Runs on a Raspberry Pi 3 Model B v1.2 (BCM2837, 4x Cortex-A53 1.2 GHz, 1 GB) now and on a Pi 5 + AI HAT+
later; every module also runs on a laptop for prototyping (webcam, recorded sessions, synthetic scenes).
Python >= 3.11, numpy >= 1.24 (1.x and 2.x), OpenCV >= 4.6 (4.x and 5.x). Pi packages come from apt
(`python3-opencv python3-numpy python3-picamera2 python3-lgpio python3-smbus2`), laptop from pip.
No other runtime dependencies. Never import `picamera2`, `lgpio`, `smbus2`/`smbus` at module top level:
import them inside the class/function that needs them so the laptop path never touches them.

Compatibility rules (both environments are tested):
- numpy: no `np.float`/`np.int`/`np.bool`, no `np.trapz`/`np.trapezoid` (use own sum), no `np.round_`,
  no `copy=` keyword tricks; dtype explicit where it matters (`np.float64`, `np.uint8`).
- OpenCV: only APIs present in 4.6 and 5.0: `cv2.fisheye.*`, `cv2.findChessboardCornersSB`,
  `cv2.findChessboardCorners`, `cv2.cornerSubPix`, `cv2.calibrateCamera`, `cv2.undistortPoints`,
  `cv2.projectPoints`, `cv2.Rodrigues`, `cv2.connectedComponentsWithStats`, `cv2.resize`, `cv2.blur`,
  `cv2.boxFilter`, `cv2.GaussianBlur`, `cv2.absdiff`, `cv2.subtract`, `cv2.threshold`, `cv2.inRange`,
  `cv2.cvtColor`, `cv2.imencode`, `cv2.imwrite`, `cv2.imread`, drawing functions,
  `cv2.calcOpticalFlowPyrLK`, `cv2.goodFeaturesToTrack`, `cv2.estimateAffinePartial2D`, `cv2.VideoCapture`,
  `cv2.imshow`/`waitKey`/`namedWindow`/`createTrackbar` (GUI; optional, guarded). NO `cv2.aruco`.
- Every timestamp is an `int` in nanoseconds on `time.monotonic_ns()`'s clock (or the simulated clock that
  stands in for it, see Clock). Durations in config are ms or s as their key suffix says; internal math SI.

## Hardware facts the code depends on (sources: hardware/design/layout.md, hardware/design/BOM.md, innomaker manual V1.4)
- Camera: innomaker CAM-MIPIOV9281 V2 — OmniVision OV9281 **MONOCHROME** global shutter, 1280x800, 3 um
  pixels, 1/4", 2-lane CSI, RAW8/RAW10, modes 1280x800 / 1280x720 / 640x400 (binned), up to ~144 fps at
  1280x800 and 210-309 fps at 640x400 (Pi 3B achieved rate unverified). Kernel overlay `dtoverlay=ov9281`,
  libcamera/Picamera2. Lens: M12 f = 2.8 mm, F2.2, "FOV D148 / H118", TV distortion < -17 %.
  NOTE: f = 2.8 mm on a 3.84 mm-wide sensor gives HFOV ~69 deg (pinhole) / ~79 deg (equidistant), not 118:
  the vendor FOV is for a larger image circle. Calibration decides; the default prior is
  fx = fy = 2.8 mm / 3 um = 933.3 px at 1280x800 (466.7 px at 640x400), equidistant fisheye, D = 0.
  Mono sensor => no hue: orange is detected as local contrast (+ optional optical filter, docs/OPTICS.md).
  Colour sources (laptop webcam) can additionally use HSV.
- Camera pose (gun frame, mm): lens axis along gun +X at (Y 0, Z 17); lens tip X 675.6; PCB back X 652;
  board mounted ROTATED 180 deg about the lens axis (J1 up) => raw image is upside down; sources rotate it.
  Bore axis Y 0, Z 49 (32 mm above the lens axis); muzzle X 841; barrel OD 24.0 @ X 500, 20.5 @ X 841
  (linear); the barrel underside is visible at the top of the image and must be masked.
- IMU: MPU-6050 on a GY-521 on pod_cap, I2C bus 1 (GPIO2/3), addr 0x68. Axis map (layout.md):
  v_gun = R_gun_imu @ v_imu, R_gun_imu = [[0,0,-1],[0,-1,0],[-1,0,0]].
- Arming switch: momentary NO pushbutton on trig_clamp_R, to GND, on a GPIO with pull-up. Pressed (LOW) =
  armed. Open circuit = NOT armed (fail-safe). INVARIANT: the trigger actuator may only move toward fire
  while the switch is held; release => actuator returns to rest immediately.
- Trigger actuators (one fitted at a time, both supported):
  * MG996R servo (default): hardware PWM GPIO18 (PWM0, `dtoverlay=pwm,pin=18,func=2`, sysfs
    /sys/class/pwm/pwmchip0/pwm0), 50 Hz frame. Lever rest +17 deg, fire -17 deg about the servo axis
    (34 deg of travel); predicted rest-to-break ~89 ms no-load / ~115 ms loaded at 4.8 V [INFERENCE].
    Servo pulses: rest_us / fire_us in config; direction must be confirmed on the bench.
  * Actuonix PQ12-100-12-S: 12 V DC motor with end limit switches, NO position feedback; needs an H-bridge
    (not on the BOM yet) on 2 GPIOs (extend / retract). 6-10 mm/s; ~6.7 mm free travel + break => predicted
    0.7-1.1 s command-to-break [INFERENCE]. Fire = extend for extend_ms, then retract for retract_ms.
- Default BCM pins: arming 17, servo PWM 18, PQ12 extend 23 / retract 24, latency LED 27,
  break-sense input 22 (bench fixture: microswitch closes to GND at trigger break), I2C 2/3.

## Frames and conventions
- GUN frame G (layout.md): +X along the bore toward the muzzle, +Y gun-left, +Z up (rib side). Origin =
  trigger face contact point. Config positions are in mm in this frame; code works in metres.
- CAMERA frame C (OpenCV): +x image right, +y image down, +z optical axis forward, origin at the optical
  centre. Upright image (after the source's 180 deg rotation): image right = gun -Y, image down = gun -Z.
  Nominal R_gun_cam = [[0,0,1],[-1,0,0],[0,-1,0]] (columns are the camera axes expressed in G).
  The boresight calibration multiplies a small correction: R_gun_cam = R_corr @ R_nominal, with
  R_corr = Rz(yaw) @ Ry(pitch) @ Rx(roll) about the GUN axes (angles = mount.cam_rpy_deg or boresight.json).
- IMU chip frame I: MPU-6050 axes. v_G = R_gun_imu @ v_I.
- WORLD frame W: gravity aligned, +Z up, right-handed; X = horizontal projection of gun +X at the moment
  the attitude estimator initialised (heading 0). R_wg(t) maps gun vectors into W. With no IMU, R_wg = I
  (the gun is treated as fixed, W == G); everything downstream still works.
- Angles: azimuth az = atan2(d_y, d_x) (positive toward gun-left), elevation el = asin(d_z), radians.
- Pixels: (u, v) = (column, row) in the UPRIGHT image at the RUNTIME resolution, distorted coordinates,
  pixel centres at integers.

## Clock
`tbot.types.Clock` protocol: `now_ns() -> int`. `MonotonicClock` wraps `time.monotonic_ns`. `SimClock`
(simulated time, for synthetic/replay sources): `set(t_ns)` jumps simulated time and restarts a
wall-clock stopwatch; `now_ns()` = last set value + real elapsed ns since that set (so real processing
latency is reproduced on simulated timestamps); `advance(dt_ns)`. Every FrameSource exposes `.clock`;
the app and solver read "now" only from it.

## Package layout and module APIs (owner slice in brackets)
All of these exist exactly as named; signatures are binding. Private helpers are free.

### tbot/types.py, tbot/config.py, tbot/cli.py, tbot/camera/__init__.py, tbot/camera/base.py [contract — do not edit]
- `Frame`, `ImuSample`, `Detection`, `Track`, `FireDecision`, `Clock`, `MonotonicClock`, `SimClock`,
  `ImuSource` protocol (start/stop/read).
- `SourceBase` (camera/base.py): optional base class for sources (context manager, `clock`, `dropped`, `imu()`).
- `Config` dataclass tree loaded from `config/default.toml` (+ `--config` override files + `--set k=v`),
  `Calibration` (all calib JSONs + mask), `LatencyBudget`, `load_config()`, `load_calibration()`,
  `add_common_args(parser)`, `load_from_args(args) -> (Config, Calibration)`, `software_root()`,
  `resolve_path(p)`, `save_json(path, obj)`.
- `FrameSource` protocol (camera/base.py) and `open_source(spec, cfg, calib) -> FrameSource`
  (camera/__init__.py). Source specs: `picam`, `cv:<index>`, `video:<path>`, `replay:<session_dir>`,
  `synthetic` or `synthetic:<scenario>`.

### tbot/optics.py [optics]
```
class CameraModel:
    width: int; height: int; K: np.ndarray (3,3) float64; D: np.ndarray float64; model: str  # "fisheye" (D len 4, cv2.fisheye) | "pinhole" (D len 5, standard)
    @classmethod nominal(cls, cfg: Config, width: int, height: int) -> CameraModel        # from cfg.optics prior
    @classmethod from_calibration(cls, intr: dict | None, cfg: Config, width: int, height: int) -> CameraModel
        # intr = intrinsics.json dict or None (=> nominal). Rescales K when intr["width"/"height"] differ
        # from the runtime size (same aspect / binning: scale fx, fy, cx, cy by the size ratio, about pixel
        # centres: c' = (c + 0.5) * s - 0.5). Raises ValueError if the aspect ratio differs > 1 %.
    def pixel_to_ray(self, uv: np.ndarray) -> np.ndarray     # (N,2) -> (N,3) unit rays, camera frame
    def ray_to_pixel(self, rays: np.ndarray) -> np.ndarray   # (N,3) -> (N,2); rows with z <= 1e-9 -> nan
    def ifov(self, uv: np.ndarray) -> np.ndarray             # (N,2) -> (N,) local angular size of 1 px (rad), mean of u/v
class MountModel:
    R_gun_cam: np.ndarray (3,3); cam_pos_g: np.ndarray (3,) m; muzzle_pos_g: (3,) m
    bore_dir_g: np.ndarray (3,) unit — mean SHOT direction: (1,0,0) rotated by boresight.json "poi_offset_rad"
        [yaw (+ = toward gun-left +Y), pitch (+ = up)] when present (patterning correction), else (1,0,0)
    @classmethod from_config(cls, cfg: Config, boresight: dict | None) -> MountModel
    def cam_to_gun(self, v_c: np.ndarray) -> np.ndarray      # (N,3) or (3,) directions
    def gun_to_cam(self, v_g: np.ndarray) -> np.ndarray
    def bore_point_g(self, range_m: float) -> np.ndarray     # muzzle + range * bore_dir (3,)
    def bore_pixel(self, cam: CameraModel, range_m: float) -> tuple[float, float]  # where the bore line at range_m appears (parallax)
def barrel_mask(cam: CameraModel, mount: MountModel, cfg: Config) -> np.ndarray
    # uint8 (H,W): 255 = usable sky, 0 = barrel (projected cylinder X lens_tip..muzzle, OD per cfg.mount, dilated by margin)
def combined_mask(cam, mount, cfg, calib: Calibration) -> np.ndarray   # barrel_mask AND calib.mask (resized NEAREST) if present
def range_from_size(diameter_px: float, ifov_rad: float, clay_diameter_m: float, psf_px: float) -> float
    # d_eff = sqrt(max(d^2 - psf^2, 0.25)); range = clay_d / (d_eff * ifov)
def rot_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray   # Rz(yaw) @ Ry(pitch) @ Rx(roll), radians
def dir_to_azel(d: np.ndarray) -> tuple[np.ndarray | float, np.ndarray | float]
def azel_to_dir(az, el) -> np.ndarray
```

### tbot/fire/ballistics.py [optics]
```
class ShotModel:
    def __init__(self, cfg: Config)                 # cfg.shot; integrates point-mass sphere drag (Mach-dependent Cd), precomputes tables to cfg.shot.max_range_m
    def time_of_flight(self, range_m) -> float | np.ndarray   # s, from the muzzle
    def velocity(self, range_m) -> float | np.ndarray          # m/s
    def drop(self, range_m, elevation_rad: float = 0.0) -> float | np.ndarray   # m gravity drop below the bore line
    def pattern_radius(self, range_m) -> float | np.ndarray    # m, effective core radius (cfg.shot pattern model)
```

### tbot/camera/base.py [contract]
```
class FrameSource(Protocol):
    width: int; height: int; is_color: bool; nominal_fps: float; clock: Clock; name: str
    def start(self) -> None
    def stop(self) -> None
    def read(self, timeout_s: float = 1.0) -> Frame | None     # newest unread frame, oldest dropped; None = timeout or end of stream
    def imu(self) -> "ImuSource | None"                       # sources that carry their own IMU stream (replay, synthetic); else None
    dropped: int                                              # frames dropped because the consumer was slow
```
Frames are upright (`camera.rotate_180` applied by the source when it is true and the source is the real
camera; replay/synthetic data is stored upright).

### tbot/camera/picam.py, cv_source.py, replay.py; tbot/app/record.py [io]
- `PicamSource(cfg, calib)`: Picamera2, sensor mode closest to camera.width x height (640x400 binned by
  default), fps, manual exposure/gain (or AE), YUV420 main stream -> Y plane as mono uint8, cv2.rotate 180
  when cfg.camera.rotate_180. Capture thread keeps only the newest request (`dropped` counts the rest).
  `t_exposure_ns = SensorTimestamp - ExposureTime*1000/2 + calib latency camera timestamp offset`;
  `t_capture_ns = monotonic_ns()` when the request is dequeued; exposure_us/gain from metadata.
- `CvSource(cfg, spec)`: cv2.VideoCapture for `cv:<index>` (live) and `video:<path>` (file; timestamps from
  CAP_PROP_POS_MSEC or seq/fps on a SimClock). `cfg.camera.emulate_mono`: "none" (BGR colour frames),
  "luma" (gray), "red" (R channel = mono sensor behind an orange/red long-pass filter).
- Session format (shared by record.py writer and replay.py reader): directory with
  `meta.json` {"version":1,"width","height","channels","dtype":"uint8","fps","source","config":{...},"calibration":{...},"created",
  "frames","complete"} ("complete" false until the writer closes; readers trust min(frames.csv rows, frames.bin size)),
  `frames.bin` (raw concatenated frames, C order), `frames.csv` (seq,t_exposure_ns,t_capture_ns,exposure_us,gain),
  `imu.csv` (t_ns,gx,gy,gz,ax,ay,az,temp_c; gun frame, rad/s, m/s^2), `events.jsonl` (one JSON object per line, each has "t_ns" and "type").
- `SessionWriter(path, width, height, channels, fps, meta_extra)`: `add_frame(frame)`, `add_imu(samples)`,
  `add_event(dict)`, `close()`; writes on a background thread, never blocks the caller > 1 ms (bounded queue,
  counts drops). `RingRecorder(cfg, width, height, channels, fps)`: RAM ring of the last
  cfg.record.preroll_s seconds (capped by cfg.record.max_ram_mb); `push(frame)`, `push_imu(samples)`,
  `event(dict)`, `trigger(t_ns)` -> after postroll_s dumps preroll+postroll to a new session dir under
  cfg.record.dir (named `YYYYmmdd-HHMMSS-<tag>`), on a background thread. `main(argv)` = `tbot record`:
  continuous recording from any source for N seconds (or until Ctrl-C), optional decimation.
- `ReplaySource(path, realtime: bool = False)`: np.memmap frames, SimClock, `.imu()` returns an ImuSource
  replaying imu.csv in step with frame time; `.events` list.

### tbot/sim/scenario.py, tbot/camera/synthetic.py [sim]
```
@dataclass class Scenario: name, duration_s, fps, clay_pos_w(t_s)->(3,) m (relative to the camera), clay_visible(t_s)->bool,
    R_wg(t_s)->(3,3), omega_g(t_s)->(3,) rad/s, launch_t_s, ... ground truth accessors
def make_scenario(name: str, cfg: Config, seed: int) -> Scenario     # names in SCENARIOS
SCENARIOS: dict[str, str]   # name -> description; at least: "skeet_crossing", "skeet_incoming", "skeet_outgoing", "trap_straightaway", "static_gun", "no_clay"
class SyntheticSource(FrameSource):
    def __init__(self, cfg: Config, calib: Calibration, scenario: str | None = None, seed: int | None = None)
    def truth(self, t_ns: int) -> dict   # {"clay_dir_w","clay_range_m","clay_pos_w","R_wg","visible","clay_px": (u,v)|None}
```
Rendering uses `CameraModel.from_calibration(calib.intrinsics, ...)` + `MountModel` + `barrel_mask`
(barrel drawn dark), physically sized clay (cfg.mount.clay_diameter_m, tilted disc), exposure motion
blur, sky gradient + clouds + sensor noise, mono (contrast cfg.sim.clay_contrast) or colour (orange on blue).
Clay physics: launch speed/angle per scenario, gravity, quadratic drag; clay positions are world-fixed
(the camera position is the world origin; the gun only rotates). IMU samples (gun frame, with
cfg.sim.gyro_noise) at cfg.imu.sample_rate_hz from the scenario's swing. Simulated timestamps start at
1_000_000_000 ns. Deterministic for a given seed.

### tbot/vision/detector.py, tbot/tools/calib_detector.py [detector]
```
class ClayDetector:
    def __init__(self, cfg: Config, mask: np.ndarray | None = None)
    def set_mask(self, mask: np.ndarray | None) -> None
    def detect(self, frame: Frame) -> list[Detection]          # sorted by score desc, at most cfg.detector.max_detections
    timings_ms: dict[str, float]                               # last call: "bg", "threshold", "components", "total"
    debug: dict[str, np.ndarray]                               # filled only when cfg.detector.debug: "contrast", "binary"
    noise_sigma: float                                         # current robust noise sigma (DN), quantisation-corrected MAD
```
Mono: background = box blur on a cfg.detector.bg_downscale-downscaled image, upsampled; signed contrast
= image - background; robust noise sigma (MAD of the contrast, refreshed every noise_every frames);
threshold max(k_sigma*sigma, min_contrast) per cfg.detector.polarity; mask; connected components;
filter area/aspect; centroid contrast-weighted; score from SNR + shape. Colour frames
(cfg.detector.mode auto/color): AND/OR the HSV orange mask (hsv_lo/hsv_hi) with the contrast mask per
cfg.detector.color_combine. `tbot calib-detector`: interactive trackbar tuner over a source/session
(shows contrast/binary/detections, writes calib/detector.json with the tuned DetectorConfig fields), and
`--auto-mask <session>`: static-pixel mask from a swinging-gun recording -> calib/mask.png.

### tbot/vision/tracker.py, tbot/fire/solver.py [tracker]
```
class ClayTracker:
    def __init__(self, cfg: Config, cam: CameraModel, mount: MountModel)
    def update(self, t_ns: int, detections: list[Detection], R_wg: np.ndarray) -> list[Track]   # returns live tracks
    def predict(self, track: Track, t_ns: int) -> tuple[float, float, float]   # (az, el, sigma_rad) at t_ns
    def predict_dir(self, track: Track, t_ns: int) -> np.ndarray               # (3,) unit, world
    def best(self) -> Track | None            # confirmed track with most hits (ties: newest)
    def reset(self) -> None
    tracks: list[Track]
class FireSolver:
    def __init__(self, cfg: Config, shot: ShotModel, mount: MountModel, latency: LatencyBudget)
    def evaluate(self, now_ns: int, tracker: ClayTracker, attitude: "AttitudeLike", armed: bool,
                 frame_period_s: float) -> FireDecision
```
Tracker: per-track constant-acceleration Kalman filter on (az, el) in W, measurement = detection
pixel -> ray (CameraModel) -> gun (MountModel) -> world (R_wg at the detection's t_ns), azimuth unwrapped;
gating in sigma with a floor (cfg.tracker.gate_min_deg); nearest-neighbour association; confirm/delete by
hits/misses; range = Gaussian fusion of cfg.tracker range prior with size-based ranges (range_from_size with
cfg.detector.psf_px; Detection.diameter_px is the OBSERVED 4-sigma major axis = object convolved with the system
blur, unthresholded) filtered over the track, sigma floored at size_range_bias_frac*range (frame-correlated
size errors). AttitudeLike = anything with `R_wg(t_ns) -> (3,3)` and
`omega_g(t_ns) -> (3,)` (AttitudeEstimator or a fixed identity).
Solver: t_exit = now + latency.fire_delay_s(actuator type) ; bore at t_exit = R_wg(now) extrapolated with
omega_g(now) over the delay; clay at t_exit + ToF(range) from the track; miss angle between the clay
direction as seen from the muzzle (camera->muzzle parallax, shot drop included) and the predicted bore;
sigma = combined 1-sigma angular uncertainty sqrt(sig_track^2 + (w_clay*jitter)^2 + (w_clay*dtof_dr*sigma_r)^2);
pattern = shot.pattern_radius(dist)/dist; p_hit = P(|e + miss| < pattern) for an isotropic 2-D Gaussian e with sigma
(fixed polar quadrature over the pattern disc, no scipy). Fire when armed, track confirmed with
>= cfg.fire.min_track_hits, range within [min_range_m, max_range_m], elevation >= min_elevation_deg,
p_hit >= cfg.fire.min_hit_probability AND p_hit(now) >= p_hit(now + k*frame_period) for k = 1..horizon_frames
(i.e. now is the best moment). `FireDecision.reason` always says why not when fire is False; when armed is False
the reason contains "not armed" (consumers match with `in`, never equality).

### tbot/imu/mpu6050.py, tbot/imu/attitude.py, tbot/tools/calib_imu.py [imu]
```
# ImuSource protocol lives in tbot/types.py (import it from there)
class Mpu6050(ImuSource): __init__(self, cfg: Config, calib: Calibration)   # FIFO mode, background reader thread
class NullImu(ImuSource)
def open_imu(cfg: Config, calib: Calibration, source: FrameSource) -> ImuSource
    # source.imu() if it has one, else Mpu6050 when cfg.imu.enabled and on a Pi, else NullImu
class AttitudeEstimator:
    def __init__(self, cfg: Config)
    def update(self, samples: list[ImuSample]) -> None
    def R_wg(self, t_ns: int) -> np.ndarray     # interpolated in history, extrapolated (constant omega) past the last sample
    def omega_g(self, t_ns: int) -> np.ndarray  # gyro rate (gun frame) nearest t_ns
    ready: bool                                  # gravity initialised (or no IMU: True with identity)
class FixedAttitude: R_wg -> I, omega_g -> 0, ready True
```
Calibration tool `tbot calib-imu {bias,axes,sync}`: static gyro/accel bias (+ temperature), guided axis
check against R_gun_imu (muzzle up / roll left / swing left prompts), camera-IMU time offset by
cross-correlating gyro rate with image rotation rate (optical flow + CameraModel) -> calib/imu.json.

### tbot/hw/gpio.py, arming.py, actuator.py, tbot/tools/latency_actuator.py [hw]
```
class Gpio(Protocol):
    def setup_output(self, pin: int, level: int = 0) -> None; def write(self, pin: int, level: int) -> None
    def setup_input(self, pin: int, pull: str = "up") -> None; def read(self, pin: int) -> int
    def watch(self, pin: int, edge: str, callback: Callable[[int, int, int], None]) -> None   # cb(pin, level, t_ns monotonic)
    def close(self) -> None
class LgpioGpio(Gpio); class SimGpio(Gpio)   # SimGpio.inject(pin, level, t_ns=None) drives inputs; .writes log
def open_gpio(cfg: Config) -> Gpio            # cfg.gpio.backend: "lgpio" | "sim" | "auto" (lgpio on a Pi else sim)
class HardwarePwm: __init__(chip, channel, frame_hz); set_pulse_us(us); disable(); (sysfs) ; SimPwm same API
class ArmingSwitch:
    def __init__(self, gpio: Gpio, cfg: Config)
    armed: bool (property, debounced, fail-safe); press_count: int; shots_this_press: int
    def on_change(self, cb: Callable[[bool, int], None]) -> None    # cb(armed, t_ns)
    def try_consume_shot(self) -> bool       # True (and counts) if armed and shots_this_press < cfg.fire.max_shots_per_arm
class Actuator(Protocol):
    kind: str; busy: bool; last_fire_ns: int | None
    def rest(self) -> None; def fire(self, t_ns: int) -> bool    # refuses (False) unless arming.armed; non-blocking
    def abort(self) -> None; def close(self) -> None
def make_actuator(cfg: Config, gpio: Gpio, arming: ArmingSwitch) -> Actuator   # cfg.actuator.type: servo|pq12|sim|none
```
Every actuator subscribes to arming.on_change and calls abort() on disarm; abort drives to rest within
one control step. `tbot latency-actuator`: N fire/rest cycles (arming must be HELD by the operator; the
tool never bypasses the interlock), break-sense input edge timestamps -> cmd->break and cmd->rest stats ->
merged into calib/latency.json under actuator.<type>.

### tbot/tools/calib_intrinsics.py, calib_boresight.py, make_board.py [calib]
- `tbot board`: chessboard PNG/PDF-ready image (cfg-free; --cols --rows --square-mm --dpi).
- `tbot calib-intrinsics`: live capture (auto-accept diverse views) or `--from-dir`/`--session` images ->
  cv2.fisheye.calibrate (fallback pinhole with --model pinhole) -> calib/intrinsics.json (+ per-view errors,
  coverage map, FOV report).
- `tbot calib-boresight`: aim at a distant target with a bore laser / known POI; click (GUI) or
  `--pixel u,v` + `--range-m` -> small rotation R_corr -> calib/boresight.json; `--poi-offset-cm dy,dz@range_m`
  from a patterning board (pattern centre relative to point of aim; +dy gun-left, +dz up) -> "poi_offset_rad".

### tbot/app/run.py, stream.py, overlay.py, tbot/tools/sim_eval.py [app]
- `Pipeline(cfg, calib, source, gpio=None, actuator=None, arming=None, imu=None)`: `.step() -> dict | None`
  processes one frame: read -> imu drain -> attitude -> detect -> track -> solve -> actuator (only when
  cfg.fire.live) -> telemetry dict (timings per stage, fps, armed, decision, tracks) ; `.run()` loop with
  watchdog (no frame for cfg.fire.watchdog_ms => actuator.abort()), SIGINT clean shutdown, ring recorder
  hooks, debug stream publish. `main(argv)` = `tbot run`.
- `DebugServer(cfg)`: ThreadingHTTPServer on cfg.stream.host:port: `/` page, `/stream.mjpg`,
  `/snapshot.jpg`, `/telemetry` (JSON); `publish(bgr, telemetry)`, rate-limited to cfg.stream.fps.
- `overlay.draw(image, *, detections, tracks, decision, mask, bore_px, telemetry, project_w=None) -> BGR image`
  (`project_w(dir_w) -> (u, v) | None` projects a world direction into the current frame; more keyword-only
  params may be added with defaults).
- `tbot sim`: closed loop over synthetic scenarios x seeds with a SimActuator -> detection rate, track
  latency, fire count, predicted-miss stats, hit rate from the shot model -> table + JSON.

### tbot/tools/latency_camera.py, latency_report.py, bench.py [latency]
- `tbot latency-camera`: LED on cfg.gpio.latency_led_pin in view of the camera, random toggles, first lit
  frame per toggle -> exposure->userspace latency stats and camera timestamp offset (partial-exposure
  brightness interpolation, global shutter) -> calib/latency.json "camera".
- `tbot latency-report`: print the full LatencyBudget (config defaults vs measured), per actuator.
- `tbot bench`: per-stage timing of detector/tracker/solver/overlay on synthetic frames at the configured
  resolution (+ optional live source), percentiles, verdict vs the frame period; saves calib/bench.json.
- Shared helper `tbot/stats.py: summarize(values) -> {"mean","std","p50","p95","min","max","n"}` [contract].

## Calibration files (paths.calib_dir, default `software/calib/`)
- intrinsics.json: {"model","width","height","K":[[..]],"D":[..],"rms_px","n_views","created","notes"}
- boresight.json: {"rpy_deg":[r,p,y],"range_m","bore_pixel":[u,v],"width","height","poi_offset_rad":[yaw,pitch],"created"}
- imu.json: {"gyro_bias_rad_s":[3],"accel_bias_m_s2":[3],"accel_scale":[3],"R_gun_imu":[[..]],"time_offset_ns",
  "temp_c","created"} (biases in the IMU CHIP frame; time_offset_ns added to IMU timestamps)
- latency.json: {"camera":{"timestamp_offset_ns","exposure_to_user_ms":{stats}},
  "actuator":{"servo":{"cmd_to_break_ms":{stats},"cmd_to_rest_ms":{stats}},"pq12":{...},"sim":{...}},"created"}
  where {stats} = {"mean","std","p50","p95","min","max","n"}.
- detector.json: {<DetectorConfig field>: value, ...} — applied over cfg.detector.
- mask.png: uint8, 255 = usable; any size (resized NEAREST to the runtime size).

## Safety model (binding for every slice)
1. `cfg.fire.live = false` by default: the pipeline logs would-fire decisions and never drives the actuator.
2. The actuator layer enforces the arming interlock itself (fire() refuses unless armed; disarm => abort()).
3. One shot per press by default (cfg.fire.max_shots_per_arm).
4. Watchdog: no processed frame for cfg.fire.watchdog_ms => abort to rest.
5. Hardware recommendation (docs): put the arming switch in series with the actuator driver enable/power
   as well, so a software fault cannot fire without the button held.
