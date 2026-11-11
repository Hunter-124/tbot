/*
 * CalRig B firmware - motorised yaw table, Arduino Uno + CNC Shield V3 (X socket).
 * Interface contract: software/docs/CALRIG_PROTOCOL.md (pins, serial protocol, safety rules).
 *
 * NEVER connect this board to anything that fires or actuates the gun. The rig swings an UNLOADED gun.
 *
 * Architecture
 *   Timer1 ISR (20 kHz)   fixed-rate DDA: a 32-bit phase accumulator adds `isr_inc` per tick, a carry is one
 *                         microstep. STEP is raised on the carry tick and lowered on the next tick (50 us
 *                         pulse). Position (int32) is only ever changed here. Step timing never depends on loop().
 *   PCINT0 ISR (D9)       hall edge -> micros() + position, queued for the `H` line, arms HOME latch.
 *   Control tick (1 kHz)  runs from loop(): e-stop, watchdog, motion planner (velocity slew limiter fed by
 *                         trapezoid / jog / enveloped sine / home seek), writes the DDA increment.
 *   loop()                polls the e-stop every pass, non-blocking line parser, ring-buffered serial output
 *                         (never blocks on a full UART buffer).
 */
#include <Arduino.h>
#include <avr/interrupt.h>
#include <avr/io.h>
#include <avr/pgmspace.h>
#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// =====================================================================================================
// Compile-time options (override with -D... in platformio.ini build_flags)
// =====================================================================================================
#ifndef MICROSTEPS
#define MICROSTEPS 16          // MUST match the driver MS1..MS3 jumpers (reported by INFO)
#endif
#ifndef DIR_INVERT
#define DIR_INVERT 0           // 1 flips the DIR pin so that + steps = yaw CCW seen from above
#endif
#ifndef ESTOP_NC
#define ESTOP_NC 1             // 1: normally-closed button to GND, HIGH (open/broken wire) = stop
#endif                         // 0: normally-open button to GND, LOW = stop
#ifndef VMAX_CAP_DEG
#define VMAX_CAP_DEG 90.0f     // compiled-in speed cap (deg/s of table)
#endif
#ifndef AMAX_CAP_DEG
#define AMAX_CAP_DEG 60.0f     // compiled-in acceleration cap (deg/s^2)
#endif
#ifndef VMAX_DEF_DEG
#define VMAX_DEF_DEG 45.0f     // power-up speed
#endif
#ifndef AMAX_DEF_DEG
#define AMAX_DEF_DEG 20.0f     // power-up acceleration
#endif
#ifndef TELEM_HZ
#define TELEM_HZ 200           // T line rate while STREAM 1; must divide 1000
#endif
#ifndef WATCHDOG_MS
#define WATCHDOG_MS 1000       // no received line for this long while moving -> STOP + E WATCHDOG
#endif
#ifndef HARD_LIM_DEG
#define HARD_LIM_DEG 100.0f    // +-hard limit, never exceeded
#endif
#ifndef SOFT_LIM_DEG
#define SOFT_LIM_DEG 90.0f     // power-up soft limits
#endif
#define FW_VERSION "1.0"

// =====================================================================================================
// Derived constants
// =====================================================================================================
#define STEPS_PER_REV   (200L * MICROSTEPS)
#define USTEPS_PER_DEG  (200.0f * (float)MICROSTEPS / 360.0f)

static constexpr int32_t deg_to_us(float d) { return (int32_t)(d * USTEPS_PER_DEG + 0.5f); }

static constexpr int32_t VMAX_CAP = deg_to_us(VMAX_CAP_DEG);    // microsteps/s
static constexpr int32_t AMAX_CAP = deg_to_us(AMAX_CAP_DEG);    // microsteps/s^2
static constexpr int32_t VMAX_DEF = deg_to_us(VMAX_DEF_DEG);
static constexpr int32_t AMAX_DEF = deg_to_us(AMAX_DEF_DEG);
static constexpr int32_t HARD_LIM = deg_to_us(HARD_LIM_DEG);
static constexpr int32_t SOFT_LIM = deg_to_us(SOFT_LIM_DEG);

static const uint32_t ISR_HZ = 20000UL;                 // DDA tick; max step rate = ISR_HZ / 2
static const uint16_t TICK_US = 1000;                   // control tick, 1 kHz
static const float    DT = 0.001f;
static const uint8_t  TELEM_DIV = 1000 / TELEM_HZ;
static const float    INC_PER_SPS = 4294967296.0f / (float)ISR_HZ;   // DDA increment per (step/s)
static const float    BRAKE = 0.9f;                     // braking uses 90 % of amax so the slew limiter can follow

static_assert(1000 % TELEM_HZ == 0, "TELEM_HZ must divide 1000");
static_assert(VMAX_CAP * 4 <= (int32_t)ISR_HZ, "VMAX_CAP too fast for the 20 kHz DDA (max step rate ISR_HZ/4)");
static_assert(HARD_LIM >= SOFT_LIM, "hard limit must be >= soft limit");
static_assert(VMAX_DEF <= VMAX_CAP && AMAX_DEF <= AMAX_CAP, "defaults exceed caps");
static_assert(MICROSTEPS == 1 || MICROSTEPS == 2 || MICROSTEPS == 4 || MICROSTEPS == 8 || MICROSTEPS == 16 ||
              MICROSTEPS == 32 || MICROSTEPS == 64, "MICROSTEPS must be a power of two <= 64");

// =====================================================================================================
// Pins (Uno ATmega328P ports)  STEP D2=PD2, DIR D5=PD5, EN D8=PB0, HOME D9=PB1 (PCINT1), ESTOP A0=PC0
// =====================================================================================================
#define STEP_BIT  _BV(2)
#define DIR_BIT   _BV(5)
#define EN_BIT    _BV(0)
#define HOME_BIT  _BV(1)
#define ESTOP_BIT _BV(0)

#define STEP_LOW()   (PORTD &= ~STEP_BIT)
#define STEP_HIGH()  (PORTD |= STEP_BIT)
#define DRIVER_OFF() (PORTB |= EN_BIT)        // EN is active LOW
#define DRIVER_ON()  (PORTB &= ~EN_BIT)

static inline bool estop_pin_asserted() {
#if ESTOP_NC
  return (PINC & ESTOP_BIT) != 0;
#else
  return (PINC & ESTOP_BIT) == 0;
#endif
}
static inline bool hall_active() { return (PINB & HOME_BIT) == 0; }   // LOW = magnet present

// =====================================================================================================
// ISR-shared state (position only ever written by the Timer1 ISR, plus ZERO/HOME re-zero with interrupts off)
// =====================================================================================================
static volatile int32_t  isr_pos;
static volatile uint32_t isr_acc;        // DDA phase: progress toward the next step in direction isr_dir
static volatile uint32_t isr_inc;        // DDA increment, 0 = stopped
static volatile int8_t   isr_dir = 1;    // last commanded direction (+1/-1), kept while stopped
static volatile int32_t  isr_lim_lo, isr_lim_hi;   // effective limits: a step that would cross is not issued
static volatile bool     isr_blocked;    // set when a step was suppressed by a limit
static int8_t            isr_dir_applied;          // ISR-private: DIR pin state (0 = unknown)

// hall capture
struct HallEvt { uint32_t t; int32_t pos; uint8_t level; };
static const uint8_t HQ_N = 8;
static volatile HallEvt hq[HQ_N];
static volatile uint8_t hq_head, hq_tail;
static volatile bool    hall_last;
static volatile bool    home_armed, home_hit;
static volatile uint32_t home_t;
static volatile int32_t  home_pos;

ISR(TIMER1_COMPA_vect) {
  STEP_LOW();                                  // end of the previous pulse
  uint32_t inc = isr_inc;
  if (!inc) return;
  int8_t d = isr_dir;
  if (d != isr_dir_applied) {                  // DIR change: set it now, step from the next tick (setup time)
    if ((d > 0) != (DIR_INVERT != 0)) PORTD |= DIR_BIT; else PORTD &= ~DIR_BIT;
    isr_dir_applied = d;
    return;
  }
  uint32_t a = isr_acc;
  uint32_t n = a + inc;
  isr_acc = n;
  if (n >= a) return;                          // no carry
  int32_t p = isr_pos;
  if (d > 0) {
    if (p >= isr_lim_hi) { isr_blocked = true; return; }
  } else {
    if (p <= isr_lim_lo) { isr_blocked = true; return; }
  }
  isr_pos = p + d;
  STEP_HIGH();
}

ISR(PCINT0_vect) {                             // only PCINT1 (D9) is unmasked
  bool lvl = hall_active();
  if (lvl == hall_last) return;
  hall_last = lvl;
  uint32_t t = micros();
  int32_t  p = isr_pos;                        // Timer1 ISR cannot preempt us: coherent
  uint8_t  h = hq_head;
  uint8_t  nh = (h + 1) & (HQ_N - 1);
  if (nh != hq_tail) {
    hq[h].t = t; hq[h].pos = p; hq[h].level = lvl;
    hq_head = nh;
  }
  if (lvl && home_armed) {
    home_t = t; home_pos = p;
    home_armed = false;
    home_hit = true;
  }
}

static inline void snapshot(int32_t &p, uint32_t &acc, int8_t &dir, uint32_t &t) {
  uint8_t sreg = SREG;
  cli();
  p = isr_pos; acc = isr_acc; dir = isr_dir; t = micros();
  SREG = sreg;
}

// =====================================================================================================
// Serial output: ring buffer drained from loop() only into free UART space -> never blocks the control tick
// =====================================================================================================
static const uint16_t TX_N = 512;
static char     txb[TX_N];
static uint16_t tx_head, tx_tail, tx_line_start;
static bool     tx_ovf, tx_skip;

static inline uint16_t tx_free() { return (uint16_t)((tx_tail - tx_head - 1) & (TX_N - 1)); }

// Start a line. `reserve` extra bytes must remain free afterwards (keeps room for replies when streaming).
static void tx_begin(uint8_t maxlen, uint16_t reserve) {
  tx_line_start = tx_head;
  tx_ovf = false;
  tx_skip = tx_free() < (uint16_t)(maxlen + reserve);
}
static inline void tx_c(char c) {
  if (tx_skip || tx_ovf) return;
  uint16_t nh = (tx_head + 1) & (TX_N - 1);
  if (nh == tx_tail) { tx_ovf = true; return; }
  txb[tx_head] = c;
  tx_head = nh;
}
static void tx_s(const char *s) { while (*s) tx_c(*s++); }
static void tx_P(const char *p) { char c; while ((c = pgm_read_byte(p++))) tx_c(c); }
static void tx_end() {
  if (tx_skip || tx_ovf) tx_head = tx_line_start; else tx_c('\n');
  if (tx_ovf) tx_head = tx_line_start;
  tx_skip = false; tx_ovf = false;
}
static void tx_u16(uint16_t v, bool pad) {      // 4 digits when pad, else minimal
  char b[5]; uint8_t n = 0;
  do { b[n++] = '0' + (v % 10); v /= 10; } while (v);
  if (pad) while (n < 4) b[n++] = '0';
  while (n) tx_c(b[--n]);
}
static void tx_u(uint32_t v) {                  // decimal; avoids repeated 32-bit division
  if (v < 65536UL) { tx_u16((uint16_t)v, false); return; }
  uint32_t a = v / 10000UL;
  uint16_t lo = (uint16_t)(v - a * 10000UL);
  if (a < 65536UL) { tx_u16((uint16_t)a, false); tx_u16(lo, true); return; }
  uint32_t b = a / 10000UL;
  uint16_t mid = (uint16_t)(a - b * 10000UL);
  tx_u16((uint16_t)b, false); tx_u16(mid, true); tx_u16(lo, true);
}
static void tx_i(int32_t v) {
  if (v < 0) { tx_c('-'); tx_u((uint32_t)0 - (uint32_t)v); } else tx_u((uint32_t)v);
}
static void tx_pump() {
  int n = Serial.availableForWrite();
  while (n > 0 && tx_tail != tx_head) {
    Serial.write((uint8_t)txb[tx_tail]);
    tx_tail = (tx_tail + 1) & (TX_N - 1);
    n--;
  }
}

// =====================================================================================================
// Controller state
// =====================================================================================================
enum Mode : uint8_t { M_IDLE, M_MOVE, M_VEL, M_SINE, M_HOME, M_STOP, M_HOME_STOP };

static Mode     mode = M_IDLE;
static bool     estop_latched;
static bool     en;                      // driver enabled (EN pin low)
static bool     stream_on;
static float    v;                       // commanded velocity, microsteps/s (signed)
static float    vmax = (float)VMAX_DEF, amax = (float)AMAX_DEF;
static int32_t  soft_lo = -SOFT_LIM, soft_hi = SOFT_LIM;
static int32_t  mv_target;
static uint8_t  mv_trim;                 // trim attempts used
static float    vel_cmd;
static int8_t   home_dirn;
static float    home_v;
static bool     home_pending;            // edge latched, re-zero not yet applied
static int32_t  home_off;                // step count at the edge (pre re-zero)
static uint32_t home_edge_t;
static uint32_t last_rx_ms;
static bool     wd_fired;
static uint32_t rx_line_us;
static uint8_t  telem_cnt;

// sine
static uint8_t  s_phase;                 // 0 = braking to rest, 1 = running
static int32_t  s_x0, s_amp;
static uint32_t s_P, s_T, s_k;           // period / total / tick counter in 1 ms ticks
static float    s_amp_f, s_omega;

// ----- events -----------------------------------------------------------------------------------------
static void ev_begin(uint32_t t, const char *code_P) {
  tx_begin(48, 0);
  tx_P(PSTR("E ")); tx_u(t); tx_c(' '); tx_P(code_P);
}
static void ev_simple(const char *code_P) { ev_begin(micros(), code_P); tx_end(); }

// ----- driver / DDA helpers ---------------------------------------------------------------------------
static void set_velocity(float vel) {
  float av = fabsf(vel);
  if (av > (float)VMAX_CAP) av = (float)VMAX_CAP;
  uint32_t inc = (uint32_t)(av * INC_PER_SPS);
  int8_t d = vel < 0 ? -1 : 1;
  uint8_t sreg = SREG; cli();
  if (inc) isr_dir = d;
  isr_inc = inc;
  SREG = sreg;
}
static void zero_motion() {
  uint8_t sreg = SREG; cli();
  isr_inc = 0;
  SREG = sreg;
  v = 0;
}
static void set_limits(int32_t lo, int32_t hi) {
  if (lo < -HARD_LIM) lo = -HARD_LIM;
  if (hi > HARD_LIM) hi = HARD_LIM;
  uint8_t sreg = SREG; cli();
  isr_lim_lo = lo; isr_lim_hi = hi;
  SREG = sreg;
}

// Apply a latched home edge: that edge becomes position 0. Safe at any moment (pos shifted atomically).
static void home_commit() {
  if (!home_pending) return;
  home_pending = false;
  uint8_t sreg = SREG; cli();
  isr_pos -= home_off;
  SREG = sreg;
  mv_target -= home_off;
  ev_begin(home_edge_t, PSTR("HOMED")); tx_c(' '); tx_i(home_off); tx_end();
}

static void cancel_home_arm() {
  uint8_t sreg = SREG; cli();
  home_armed = false; home_hit = false;
  SREG = sreg;
}

static void estop_trip(const char *reason_P) {
  if (estop_latched) return;
  DRIVER_OFF();                                // first: hardware disable, before anything else
  zero_motion();
  estop_latched = true;
  en = false;
  mode = M_IDLE;
  cancel_home_arm();
  home_commit();
  ev_begin(micros(), PSTR("ESTOP")); tx_c(' '); tx_P(reason_P); tx_end();
}

// ----- replies ----------------------------------------------------------------------------------------
static void reply_ok(const char *cmd_P) { tx_begin(32, 0); tx_P(PSTR("OK ")); tx_P(cmd_P); tx_end(); }
static void reply_err(const char *cmd_P, const char *code_P, const char *text_P) {
  tx_begin(64, 0);
  tx_P(PSTR("ERR ")); tx_P(cmd_P); tx_c(' '); tx_P(code_P); tx_c(' '); tx_P(text_P);
  tx_end();
}

// ----- planner ----------------------------------------------------------------------------------------
static inline float slew(float cur, float want, float dv) {
  if (want > cur + dv) return cur + dv;
  if (want < cur - dv) return cur - dv;
  return want;
}
static inline float braking_speed(float dist, float a) {        // max speed from which `a` stops within dist
  if (dist <= 0) return 0;
  return sqrtf(2.0f * BRAKE * a * dist);
}
static inline float sine_window(float c) {                       // c = cos(pi*u); returns sqrt(1-c^2) >= 0
  float s2 = 1.0f - c * c;
  return s2 > 0 ? sqrtf(s2) : 0.0f;
}

static void begin_motion() {            // common prologue of a new motion command (replaces any current one)
  home_commit();
  cancel_home_arm();
  wd_fired = false;
}

static void finish_idle() {
  zero_motion();
  mode = M_IDLE;
}

static void start_stop() {
  if (mode == M_IDLE || mode == M_STOP || mode == M_HOME_STOP) return;
  cancel_home_arm();
  mode = M_STOP;
}

static void tick() {
  int32_t pos; uint32_t acc; int8_t dir; uint32_t now;
  snapshot(pos, acc, dir, now);
  float frac = (float)acc * (1.0f / 4294967296.0f);
  float x = (float)pos + (dir > 0 ? frac : -frac);        // continuous position estimate (microsteps)

  // limits in force: hard while seeking home, else soft
  bool homing = (mode == M_HOME || mode == M_HOME_STOP);
  int32_t lo = homing ? -HARD_LIM : soft_lo, hi = homing ? HARD_LIM : soft_hi;
  set_limits(lo, hi);

  float a  = amax;
  float dv = a * DT;
  float vd = 0;
  bool  clip = false;                // desired velocity reduced by a limit guard
  float snapv = 4.0f * dv;

  if (isr_blocked) {                 // a step was suppressed at a limit: motion is over
    isr_blocked = false;
    if (mode == M_VEL || mode == M_SINE || mode == M_HOME) {      // MOVE/STOP: the planner already ends at the limit
      ev_simple(PSTR("LIMIT"));
      cancel_home_arm();
      finish_idle();
      mode = M_IDLE;
    }
  }

  switch (mode) {
    case M_IDLE:
      break;

    case M_STOP:
      if (v == 0) { finish_idle(); ev_simple(PSTR("DONE")); }
      break;

    case M_MOVE: {
      float rem = (float)mv_target - x;
      float ar = fabsf(rem);
      if (ar <= 1.0f && fabsf(v) <= snapv) {                   // arrived within one microstep: trim to exact
        v = 0;
        if (pos == mv_target) { finish_idle(); ev_simple(PSTR("DONE")); break; }
        if (mv_trim < 4) {
          mv_trim++;
          uint8_t sreg = SREG; cli();                          // one single step in the right direction
          isr_dir = (mv_target > pos) ? 1 : -1;
          isr_acc = 0xFFFFFFFFUL;
          isr_inc = 0x100000UL;                                // carries on the next ISR pass, no 2nd step within the tick
          SREG = sreg;
          set_limits(lo, hi);
          return;
        }
        finish_idle(); ev_simple(PSTR("DONE")); break;         // could not trim (blocked): give up quietly
      }
      float sp = braking_speed(ar, a);
      if (sp > vmax) sp = vmax;
      vd = rem >= 0 ? sp : -sp;
      break;
    }

    case M_VEL:
      vd = vel_cmd;
      if (vd == 0 && v == 0) { finish_idle(); ev_simple(PSTR("DONE")); }
      break;

    case M_HOME:
      vd = (float)home_dirn * home_v;
      break;

    case M_HOME_STOP:
      if (v == 0) {
        finish_idle();
        home_commit();
        ev_simple(PSTR("DONE"));
      }
      break;

    case M_SINE:
      if (s_phase == 0) {                                      // brake to rest first, centre = rest position
        vd = 0;
        if (v == 0) {
          s_x0 = pos;
          if (s_x0 - s_amp < soft_lo || s_x0 + s_amp > soft_hi) {
            ev_simple(PSTR("LIMIT")); finish_idle(); break;
          }
          s_k = 0; s_phase = 1;
        }
        break;
      } else {
        float Ps = (float)s_P * 0.001f;
        float th = ((float)(s_k % s_P) + 0.5f) / (float)s_P;   // cycle fraction at the middle of this tick
        float s = sinf(6.2831853f * th);
        float c = sine_window(s);                              // |cos|
        if (th > 0.25f && th < 0.75f) c = -c;
        float in = 1, din = 0, out = 1, dout = 0;
        if (s_k < s_P) {
          float ci = cosf(3.14159265f * ((float)s_k + 0.5f) / (float)s_P);
          in = 0.5f * (1.0f - ci); din = 0.5f * sine_window(ci) * 3.14159265f / Ps;
        }
        uint32_t left = s_T - s_k;
        if (left <= s_P) {
          float co = cosf(3.14159265f * ((float)left - 0.5f) / (float)s_P);
          out = 0.5f * (1.0f - co); dout = -0.5f * sine_window(co) * 3.14159265f / Ps;
        }
        float env = in * out, denv = din * out + in * dout;
        float vr = s_amp_f * (denv * s + env * s_omega * c);
        vd = vr;                                               // open loop: the DDA integrates it exactly
        if (++s_k >= s_T) {                                    // finished: settle exactly on the centre
          mv_target = s_x0; mv_trim = 0; mode = M_MOVE;
        }
      }
      break;
  }

  // soft/hard limit guard on the desired velocity (applies to every mode)
  {
    float vp = braking_speed((float)hi - x, a);
    float vn = braking_speed(x - (float)lo, a);
    if (vd > vp) { vd = vp; clip = true; }
    if (vd < -vn) { vd = -vn; clip = true; }
  }

  if (mode == M_HOME && home_hit) {                            // edge latched by the PCINT ISR
    uint8_t sreg = SREG; cli();
    home_hit = false;
    SREG = sreg;
    home_off = home_pos; home_edge_t = home_t; home_pending = true;
    mode = M_HOME_STOP;
    vd = 0;
  }

  float lim = a;
  if (mode == M_SINE) { lim = 1.25f * a; if (lim > (float)AMAX_CAP) lim = (float)AMAX_CAP; }
  v = slew(v, vd, lim * DT);

  if (clip && (mode == M_VEL || mode == M_HOME) && fabsf(v) <= snapv && fabsf(vd) <= snapv) {
    v = 0;
    zero_motion();
    cancel_home_arm();
    mode = M_IDLE;
    ev_simple(PSTR("LIMIT"));
  }

  if (en && !estop_latched) set_velocity(v); else zero_motion();

  // telemetry
  if (stream_on && ++telem_cnt >= TELEM_DIV) {
    telem_cnt = 0;
    tx_begin(40, 240);
    tx_P(PSTR("T ")); tx_u(now); tx_c(' '); tx_i(pos); tx_c(' '); tx_i((int32_t)(v >= 0 ? v + 0.5f : v - 0.5f));
    tx_end();
  }
}

// =====================================================================================================
// Command parser
// =====================================================================================================
static const uint8_t LINE_MAX = 48;
static char    line[LINE_MAX + 1];
static uint8_t line_len;
static bool    line_over;

static bool parse_i32(const char *s, int32_t &out) {
  if (!*s) return false;
  char *end;
  errno = 0;
  long r = strtol(s, &end, 10);
  if (*end || errno) return false;
  out = (int32_t)r;
  return true;
}

static bool motion_ready(const char *cmd_P) {
  if (estop_latched || !en) {
    reply_err(cmd_P, PSTR("NOT_READY"), estop_latched ? PSTR("e-stop latched") : PSTR("driver disabled"));
    return false;
  }
  return true;
}

static const char *state_name_P() {
  if (estop_latched) return PSTR("ESTOP");
  switch (mode) {
    case M_IDLE: return PSTR("IDLE");
    case M_VEL: return PSTR("VEL");
    case M_SINE: return PSTR("SINE");
    case M_HOME: case M_HOME_STOP: return PSTR("HOME");
    default: return PSTR("MOVE");               // MOVE and the STOP deceleration
  }
}

static void handle_line(char *s) {
  char *argv[5]; uint8_t argc = 0;
  while (*s) {
    while (*s == ' ' || *s == '\t') s++;
    if (!*s) break;
    if (argc == 5) { reply_err(PSTR("?"), PSTR("ARG"), PSTR("too many arguments")); return; }
    argv[argc++] = s;
    while (*s && *s != ' ' && *s != '\t') { if (*s >= 'a' && *s <= 'z') *s -= 32; s++; }
    if (*s) *s++ = 0;
  }
  if (!argc) return;
  const char *c = argv[0];
#define IS(name) (strcmp_P(c, PSTR(name)) == 0)
#define NEED(n, name) if (argc != (n)) { reply_err(PSTR(name), PSTR("ARG"), PSTR("wrong argument count")); return; }
  int32_t a1, a2, a3;

  if (IS("PING")) {
    NEED(2, "PING");
    for (const char *p = argv[1]; *p; p++) if (*p < '0' || *p > '9' || p - argv[1] > 10) { reply_err(PSTR("PING"), PSTR("ARG"), PSTR("seq must be digits")); return; }
    tx_begin(40, 0); tx_P(PSTR("PONG ")); tx_s(argv[1]); tx_c(' '); tx_u(rx_line_us); tx_end();
  } else if (IS("INFO")) {
    NEED(1, "INFO");
    tx_begin(160, 0);
    tx_P(PSTR("OK INFO fw=calrig-uno/" FW_VERSION " usteps=")); tx_i(MICROSTEPS);
    tx_P(PSTR(" steps_rev=200 ratio=1 vmax_cap=")); tx_i(VMAX_CAP);
    tx_P(PSTR(" amax_cap=")); tx_i(AMAX_CAP);
    tx_P(PSTR(" hard_lim=")); tx_i(HARD_LIM);
    tx_P(PSTR(" telem_hz=")); tx_i(TELEM_HZ);
    tx_end();
  } else if (IS("EN")) {
    NEED(2, "EN");
    if (!parse_i32(argv[1], a1) || (a1 != 0 && a1 != 1)) { reply_err(PSTR("EN"), PSTR("ARG"), PSTR("expected 0 or 1")); return; }
    if (a1) {
      if (estop_latched) { reply_err(PSTR("EN"), PSTR("ESTOP"), PSTR("e-stop latched")); return; }
      DRIVER_ON(); en = true;
    } else {
      DRIVER_OFF(); en = false;
      begin_motion(); zero_motion(); mode = M_IDLE;
    }
    tx_begin(16, 0); tx_P(PSTR("OK EN ")); tx_i(a1); tx_end();
  } else if (IS("ESTOP")) {
    NEED(1, "ESTOP");
    estop_trip(PSTR("CMD"));
    reply_ok(PSTR("ESTOP"));
  } else if (IS("RESET")) {
    NEED(1, "RESET");
    if (estop_pin_asserted()) { reply_err(PSTR("RESET"), PSTR("ESTOP_PIN"), PSTR("e-stop input still asserted")); return; }
    estop_latched = false;                      // driver stays disabled until EN 1
    reply_ok(PSTR("RESET"));
  } else if (IS("CFG")) {
    NEED(3, "CFG");
    if (!parse_i32(argv[1], a1) || !parse_i32(argv[2], a2) || a1 < 1 || a2 < 1) { reply_err(PSTR("CFG"), PSTR("ARG"), PSTR("expected positive integers")); return; }
    if (a1 > VMAX_CAP) a1 = VMAX_CAP;
    if (a2 > AMAX_CAP) a2 = AMAX_CAP;
    vmax = (float)a1; amax = (float)a2;
    tx_begin(48, 0); tx_P(PSTR("OK CFG vmax=")); tx_i(a1); tx_P(PSTR(" amax=")); tx_i(a2); tx_end();
  } else if (IS("LIM")) {
    NEED(3, "LIM");
    if (!parse_i32(argv[1], a1) || !parse_i32(argv[2], a2) || a1 > a2) { reply_err(PSTR("LIM"), PSTR("ARG"), PSTR("expected min <= max")); return; }
    if (a1 < -HARD_LIM) a1 = -HARD_LIM;
    if (a2 > HARD_LIM) a2 = HARD_LIM;
    if (a1 > HARD_LIM) a1 = HARD_LIM;
    if (a2 < -HARD_LIM) a2 = -HARD_LIM;
    soft_lo = a1; soft_hi = a2;
    tx_begin(48, 0); tx_P(PSTR("OK LIM min=")); tx_i(a1); tx_P(PSTR(" max=")); tx_i(a2); tx_end();
  } else if (IS("ZERO")) {
    NEED(1, "ZERO");
    if (mode != M_IDLE || v != 0) { reply_err(PSTR("ZERO"), PSTR("MOVING"), PSTR("motion active")); return; }
    uint8_t sreg = SREG; cli();
    isr_pos = 0; isr_acc = 0;
    SREG = sreg;
    reply_ok(PSTR("ZERO"));
  } else if (IS("MOVE")) {
    NEED(2, "MOVE");
    if (!parse_i32(argv[1], a1)) { reply_err(PSTR("MOVE"), PSTR("ARG"), PSTR("expected integer target")); return; }
    if (!motion_ready(PSTR("MOVE"))) return;
    if (a1 < soft_lo || a1 > soft_hi) { reply_err(PSTR("MOVE"), PSTR("LIMIT"), PSTR("target outside soft limits")); return; }
    begin_motion();
    mv_target = a1; mv_trim = 0; mode = M_MOVE;
    reply_ok(PSTR("MOVE"));
  } else if (IS("VEL")) {
    NEED(2, "VEL");
    if (!parse_i32(argv[1], a1)) { reply_err(PSTR("VEL"), PSTR("ARG"), PSTR("expected integer velocity")); return; }
    if (!motion_ready(PSTR("VEL"))) return;
    if (a1 > (int32_t)vmax) a1 = (int32_t)vmax;
    if (a1 < -(int32_t)vmax) a1 = -(int32_t)vmax;
    begin_motion();
    vel_cmd = (float)a1; mode = M_VEL;
    reply_ok(PSTR("VEL"));
  } else if (IS("SINE")) {
    NEED(4, "SINE");
    if (!parse_i32(argv[1], a1) || !parse_i32(argv[2], a2) || !parse_i32(argv[3], a3) || a1 < 1 || a2 < 20 || a3 < 1) {
      reply_err(PSTR("SINE"), PSTR("ARG"), PSTR("expected amp>=1 period_ms>=20 cycles>=1")); return;
    }
    if (!motion_ready(PSTR("SINE"))) return;
    if ((uint64_t)a2 * (uint64_t)a3 > 0x7FFFFFFFULL) { reply_err(PSTR("SINE"), PSTR("ARG"), PSTR("run too long")); return; }
    float om = 6.2831853f / ((float)a2 * 0.001f);
    float amp = (float)a1;
    if (amp * om * om > amax) { reply_err(PSTR("SINE"), PSTR("ACCEL"), PSTR("amp*(2pi/T)^2 exceeds amax")); return; }
    if (amp * om > vmax) { reply_err(PSTR("SINE"), PSTR("SPEED"), PSTR("amp*2pi/T exceeds vmax")); return; }
    int32_t p0; uint32_t acc0; int8_t d0; uint32_t t0;
    snapshot(p0, acc0, d0, t0);
    if (p0 - a1 < soft_lo || p0 + a1 > soft_hi) { reply_err(PSTR("SINE"), PSTR("LIMIT"), PSTR("swing exceeds soft limits")); return; }
    begin_motion();
    s_amp = a1; s_amp_f = amp; s_omega = om; s_P = (uint32_t)a2; s_T = (uint32_t)a2 * (uint32_t)a3;
    s_k = 0; s_phase = 0; s_x0 = p0;
    mode = M_SINE;
    reply_ok(PSTR("SINE"));
  } else if (IS("HOME")) {
    NEED(3, "HOME");
    if (!parse_i32(argv[1], a1) || !parse_i32(argv[2], a2) || (a1 != 1 && a1 != -1) || a2 < 1) {
      reply_err(PSTR("HOME"), PSTR("ARG"), PSTR("expected dir=+-1 v>=1")); return;
    }
    if (!motion_ready(PSTR("HOME"))) return;
    if (hall_active()) { reply_err(PSTR("HOME"), PSTR("ACTIVE"), PSTR("sensor already active, move off the magnet")); return; }
    if (a2 > VMAX_CAP) a2 = VMAX_CAP;
    begin_motion();
    home_dirn = (int8_t)a1; home_v = (float)a2;
    uint8_t sreg = SREG; cli();
    home_hit = false; home_armed = true;
    SREG = sreg;
    mode = M_HOME;
    reply_ok(PSTR("HOME"));
  } else if (IS("STOP")) {
    NEED(1, "STOP");
    start_stop();
    reply_ok(PSTR("STOP"));
  } else if (IS("STREAM")) {
    NEED(2, "STREAM");
    if (!parse_i32(argv[1], a1) || (a1 != 0 && a1 != 1)) { reply_err(PSTR("STREAM"), PSTR("ARG"), PSTR("expected 0 or 1")); return; }
    stream_on = a1; telem_cnt = 0;
    tx_begin(24, 0); tx_P(PSTR("OK STREAM ")); tx_i(a1); tx_end();
  } else if (IS("HB")) {
    NEED(1, "HB");
    reply_ok(PSTR("HB"));
  } else if (IS("STAT")) {
    NEED(1, "STAT");
    int32_t p; uint32_t acc; int8_t d; uint32_t t;
    snapshot(p, acc, d, t);
    tx_begin(120, 0);
    tx_P(PSTR("OK STAT t=")); tx_u(t);
    tx_P(PSTR(" pos=")); tx_i(p);
    tx_P(PSTR(" vel=")); tx_i((int32_t)(v >= 0 ? v + 0.5f : v - 0.5f));
    tx_P(PSTR(" state=")); tx_P(state_name_P());
    tx_P(PSTR(" en=")); tx_c(en ? '1' : '0');
    tx_P(PSTR(" home=")); tx_c(hall_active() ? '1' : '0');
    tx_P(PSTR(" estop_pin=")); tx_c(estop_pin_asserted() ? '1' : '0');
    tx_end();
  } else {
    reply_err(PSTR("?"), PSTR("UNKNOWN"), PSTR("unknown command"));
  }
#undef IS
#undef NEED
}

static void serial_rx() {
  uint8_t budget = 64;
  while (budget-- && Serial.available() > 0) {
    int ch = Serial.read();
    if (ch == '\n') {
      rx_line_us = micros();
      last_rx_ms = millis();
      wd_fired = false;
      if (line_over) {
        reply_err(PSTR("?"), PSTR("OVERFLOW"), PSTR("line too long"));
      } else if (line_len) {
        line[line_len] = 0;
        handle_line(line);
      }
      line_len = 0; line_over = false;
    } else if (ch == '\r' || ch < 0) {
      // ignore
    } else if (line_len < LINE_MAX) {
      line[line_len++] = (char)ch;
    } else {
      line_over = true;
    }
  }
}

static void drain_hall() {
  while (hq_tail != hq_head) {
    uint8_t t = hq_tail;
    uint32_t ht; int32_t hp; uint8_t hl;
    {
      uint8_t sreg = SREG; cli();
      ht = hq[t].t; hp = hq[t].pos; hl = hq[t].level;
      SREG = sreg;
    }
    tx_begin(40, 0);
    tx_P(PSTR("H ")); tx_u(ht); tx_c(' '); tx_i(hp); tx_c(' '); tx_c(hl ? '1' : '0');
    tx_end();
    hq_tail = (t + 1) & (HQ_N - 1);
  }
}

// =====================================================================================================
// Arduino entry points
// =====================================================================================================
static uint32_t next_tick_us;

void setup() {
  // driver off FIRST, then everything else
  PORTB |= EN_BIT; DDRB |= EN_BIT;
  PORTD &= ~(STEP_BIT | DIR_BIT); DDRD |= STEP_BIT | DIR_BIT;
  DDRB &= ~HOME_BIT; PORTB |= HOME_BIT;                 // INPUT_PULLUP
  DDRC &= ~ESTOP_BIT; PORTC |= ESTOP_BIT;               // INPUT_PULLUP

  set_limits(-SOFT_LIM, SOFT_LIM);
  hall_last = hall_active();

  Serial.begin(115200);

  cli();
  TCCR1A = 0;
  TCCR1B = _BV(WGM12) | _BV(CS10);                      // CTC, no prescaler
  OCR1A = (uint16_t)(F_CPU / ISR_HZ - 1);
  TCNT1 = 0;
  TIMSK1 = _BV(OCIE1A);
  PCICR |= _BV(PCIE0);
  PCMSK0 |= _BV(PCINT1);                                // D9
  sei();

  last_rx_ms = millis();
  ev_simple(PSTR("BOOT"));
  if (estop_pin_asserted()) estop_trip(PSTR("PIN"));
  next_tick_us = micros() + TICK_US;
}

void loop() {
  if (!estop_latched && estop_pin_asserted()) estop_trip(PSTR("PIN"));   // polled on every pass (and each tick)

  serial_rx();

  uint32_t now = micros();
  if ((int32_t)(now - next_tick_us) >= 0) {
    next_tick_us += TICK_US;
    if ((int32_t)(now - next_tick_us) >= 0) next_tick_us = now + TICK_US;   // fell behind: resync
    if (!estop_latched && estop_pin_asserted()) estop_trip(PSTR("PIN"));
    if (!estop_latched && mode != M_IDLE && mode != M_STOP && mode != M_HOME_STOP && !wd_fired &&
        (uint32_t)(millis() - last_rx_ms) > WATCHDOG_MS) {
      wd_fired = true;
      start_stop();
      ev_simple(PSTR("WATCHDOG"));
    }
    tick();
  }

  drain_hall();
  tx_pump();
}
