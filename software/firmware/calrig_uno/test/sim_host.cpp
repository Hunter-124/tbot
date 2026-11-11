// Host-side closed-loop simulation of calrig_uno.ino: Timer1 ISR @20 kHz, 1 us clock, simulated hall magnet,
// e-stop input and UART. Build+run:  g++ -std=c++17 -O1 -Istub sim_host.cpp -o /tmp/sim_host && /tmp/sim_host
#include <Arduino.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <unistd.h>
#include <sys/wait.h>

volatile uint8_t PORTB, DDRB, PINB, PORTC, DDRC, PINC, PORTD, DDRD, PIND, SREG;
volatile uint8_t TCCR1A, TCCR1B, TIMSK1, PCICR, PCMSK0;
volatile uint16_t OCR1A, TCNT1;
uint32_t sim_us;
SerialStub Serial;

#include "../calrig_uno.ino"

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("  FAIL %s:%d: %s  ", __FILE__, __LINE__, #c); printf(__VA_ARGS__); printf("\n"); } } while (0)

struct Sim {
  int32_t mag_lo = 100000, mag_hi = 100001;        // hall magnet window in microsteps (none by default)
  bool estop_btn = false;                          // true = button pressed (NC contact open)
  std::string out; size_t out_read = 0;
  std::vector<double> t, pos, vs;                      // 1 ms samples
  std::vector<std::string> events;
  int steps_up = 0, steps_dn = 0; bool step_pin_prev = false; uint32_t last_step_us = 0; double min_step_dt = 1e9;
  Sim() {
    PINC = 0; PINB = HOME_BIT;                     // hall inactive = HIGH
    setup();
  }
  void drive_inputs() {
    bool asserted = estop_btn;
    // NC button: closed (not pressed) -> pin LOW; pressed/open -> pullup HIGH
    if (ESTOP_NC) PINC = asserted ? ESTOP_BIT : 0; else PINC = asserted ? 0 : ESTOP_BIT;
    bool act = pos_now() >= mag_lo && pos_now() <= mag_hi;
    uint8_t want = act ? 0 : HOME_BIT;
    if ((PINB & HOME_BIT) != want) { PINB = (PINB & ~HOME_BIT) | want; if (PCMSK0 & _BV(PCINT1)) ISR_PCINT0_vect(); }
  }
  int32_t pos_now() { return isr_pos; }
  void send(const std::string &s) { for (char c : s) Serial.rx.push_back((uint8_t)c); Serial.rx.push_back('\n'); run_ms(1); }
  void run_ms(int ms) {
    for (int i = 0; i < ms * 20; i++) {
      sim_us += 50;
      bool before = (PORTD & STEP_BIT) != 0;
      drive_inputs();
      ISR_TIMER1_COMPA_vect();
      bool after = (PORTD & STEP_BIT) != 0;
      if (!before && after) { uint32_t dt = sim_us - last_step_us; if (last_step_us && dt < min_step_dt) min_step_dt = dt; last_step_us = sim_us; }
      Serial.tx_space = 64;                         // UART drains faster than we fill in this model
      loop();
      if ((i % 20) == 19) { vs.push_back(v); t.push_back(sim_us / 1e6); pos.push_back((double)isr_pos + (isr_dir > 0 ? 1 : -1) * (double)isr_acc / 4294967296.0); }
    }
    std::string o = Serial.tx; Serial.tx.clear(); out += o;
  }
  double vacc(size_t a) { double m = 0; for (size_t i = a + 1; i < vs.size(); i++) m = std::max(m, fabs(vs[i] - vs[i - 1]) / 1e-3); return m; }
  std::string new_out() { std::string r = out.substr(out_read); out_read = out.size(); return r; }
  bool has(const std::string &needle, size_t from = 0) { return out.find(needle, from) != std::string::npos; }
  // peak |velocity| and |acceleration| (finite differences over 20 ms windows) between sample indexes
  void kin(size_t a, size_t b, double &vpk, double &apk) {
    vpk = apk = 0; const int w = 20;
    for (size_t i = a + w; i < b && i < pos.size(); i++) {
      double v = (pos[i] - pos[i - w]) / (w * 1e-3), vp = (i >= a + 2 * w) ? (pos[i - w] - pos[i - 2 * w]) / (w * 1e-3) : 0;
      vpk = std::max(vpk, fabs(v));
      if (i >= a + 2 * w) apk = std::max(apk, fabs(v - vp) / (w * 1e-3));
    }
  }
};

static std::string lines_with(const std::string &s, const char *pfx) {
  std::istringstream is(s); std::string l, r;
  while (std::getline(is, l)) if (l.rfind(pfx, 0) == 0) r += l + "\n";
  return r;
}

int main(int argc, char **argv) {
  int pick = argc > 1 ? atoi(argv[1]) : -1;
  if (pick < 0) { int bad = 0; for (int i = 0; i < 9; i++) { pid_t p = fork(); if (p == 0) { char b[8]; snprintf(b, 8, "%d", i); execl(argv[0], argv[0], b, (char *)0); _exit(99); } int st; waitpid(p, &st, 0); if (!WIFEXITED(st) || WEXITSTATUS(st)) bad++; } printf(bad ? "\n%d SCENARIOS FAILED\n" : "\nALL SCENARIOS OK\n", bad); return bad ? 1 : 0; }

  
  const double amax_d = AMAX_DEF, vmax_d = VMAX_DEF;
  printf("USTEPS/deg=%.4f vmax=%d amax=%d cap v=%d a=%d hard=%d\n", USTEPS_PER_DEG, (int)VMAX_DEF, (int)AMAX_DEF, (int)VMAX_CAP, (int)AMAX_CAP, (int)HARD_LIM);
  if (pick == 0 || pick == 99) {
    printf("[boot/info/protocol]\n");
    Sim s; s.run_ms(5);
    CHECK(s.has("E ") && s.has(" BOOT"), "no BOOT: %s", s.out.c_str());
    CHECK(PORTB & EN_BIT, "driver must boot disabled");
    s.send("PING 42"); s.send("INFO"); s.send("STAT"); s.send("FOO"); s.send("MOVE 10"); s.send("EN 2"); s.send("stream 0");
    s.send(std::string(100, 'A')); s.send("HB");
    s.run_ms(50);
    printf("%s", s.new_out().c_str());
    CHECK(s.has("PONG 42 "), "pong");
    CHECK(s.has("ERR MOVE NOT_READY"), "not ready");
    CHECK(s.has("ERR ? UNKNOWN"), "unknown");
    CHECK(s.has("ERR ? OVERFLOW"), "overflow");
    CHECK(s.has("OK HB"), "hb after overflow");
    CHECK(s.has("ERR EN ARG"), "en arg");
  }
  if (pick == 1 || pick == 99) {
    printf("[trapezoid MOVE 800, then MOVE 0]\n");
    Sim s; s.run_ms(5); s.send("EN 1"); s.send("MOVE 800"); s.run_ms(10);
    size_t i0 = s.t.size();
    for (int k = 0; k < 40 && !s.has("DONE"); k++) { s.send("HB"); s.run_ms(250); }
    CHECK(s.has(" DONE"), "no DONE");
    CHECK(isr_pos == 800, "pos %d", (int)isr_pos);
    double vp, ap; s.kin(i0 - 10, s.t.size(), vp, ap);
    printf("  peak v=%.1f (vmax %.0f) peak a=%.1f (amax %.1f) time=%.2fs min step gap %.0f us\n", vp, vmax_d, ap, amax_d, s.t.back(), s.min_step_dt);
    CHECK(vp <= vmax_d * 1.03 && vp > vmax_d * 0.9, "vpeak");
    CHECK(ap <= amax_d * 1.1, "apeak");
    // replace in flight: go back to -400 starting mid cruise
    s.send("MOVE 0"); s.run_ms(300); s.send("MOVE -400"); size_t i1 = s.t.size();
    for (int k = 0; k < 60 && !(isr_pos == -400 && mode == M_IDLE); k++) { s.send("HB"); s.run_ms(250); }
    CHECK(isr_pos == -400, "pos2 %d", (int)isr_pos);
    s.kin(i1 - 330, s.t.size(), vp, ap);
    printf("  replace: peak a=%.1f\n", ap);
    CHECK(ap <= amax_d * 1.1, "apeak2");
    s.send("MOVE 900"); s.send("MOVE 5"); s.send("STAT"); s.run_ms(20);
    CHECK(s.has("ERR MOVE LIMIT"), "limit refusal");
    // short moves land exactly
    for (int target : {-399, -397, -390, -300}) { s.send("MOVE " + std::to_string(target)); s.run_ms(10); for (int k = 0; k < 30 && mode != M_IDLE; k++) { s.send("HB"); s.run_ms(100);} s.run_ms(20); CHECK(isr_pos == target, "short move %d -> %d", target, (int)isr_pos); }
  }
  if (pick == 2 || pick == 99) {
    printf("[VEL to soft limit, STOP, LIM]\n");
    Sim s; s.run_ms(5); s.send("EN 1"); s.send("VEL 400"); size_t i0 = s.t.size();
    for (int k = 0; k < 100 && !s.has(" LIMIT"); k++) { s.send("HB"); s.run_ms(100); }
    s.run_ms(50);
    CHECK(s.has(" LIMIT"), "no LIMIT");
    printf("  pos at limit %d  soft_hi %d  mode=%d\n", (int)isr_pos, (int)soft_hi, (int)mode);
    CHECK(isr_pos <= soft_hi && isr_pos >= soft_hi - 2, "stopped at limit");
    double vp, ap; s.kin(i0, s.t.size(), vp, ap); printf("  peak a=%.1f\n", ap);
    CHECK(ap <= amax_d * 1.1, "apeak vel");
    s.send("VEL -400"); s.run_ms(2000); s.send("STOP"); s.send("HB");
    for (int k = 0; k < 20 && mode != M_IDLE; k++) { s.send("HB"); s.run_ms(100); }
    CHECK(mode == M_IDLE && v == 0, "stopped");
    s.send("LIM -50 50"); s.send("ZERO"); s.send("LIM 100 -100"); s.send("LIM -5000 5000"); s.send("CFG 99999 99999"); s.send("CFG 100 50"); s.run_ms(30);
    printf("%s", lines_with(s.new_out(), "OK LIM").c_str());
    CHECK(s.has("OK LIM min=-889 max=889") || s.has("OK LIM min=-888 max=888"), "lim clamp");
    CHECK(s.has("OK CFG vmax=800 amax=533"), "cfg clamp");
    CHECK(s.has("ERR LIM ARG"), "lim arg");
    CHECK(vmax == 100 && amax == 50, "cfg applied");
  }
  if (pick == 3 || pick == 99) {
    printf("[SINE]\n");
    Sim s; s.run_ms(5); s.send("EN 1"); s.send("CFG 400 178"); s.send("MOVE 100"); for (int k = 0; k < 40 && mode != M_IDLE; k++) { s.send("HB"); s.run_ms(100); }
    s.send("SINE 400 4000 3"); s.send("SINE 400 1000 3"); s.send("SINE 900 20000 3"); s.send("SINE 1 3000 1"); s.run_ms(20);
    printf("%s", lines_with(s.new_out(), "ERR").c_str());
    CHECK(s.has("ERR SINE ACCEL") && s.has("ERR SINE LIMIT"), "sine checks");
    s.send("SINE 400 10000 3"); size_t i0 = s.t.size();
    for (int k = 0; k < 400 && mode != M_IDLE; k++) { s.send("HB"); s.run_ms(100); }
    double lo = 1e9, hi = -1e9; for (size_t i = i0; i < s.pos.size(); i++) { lo = std::min(lo, s.pos[i]); hi = std::max(hi, s.pos[i]); }
    double vp, ap; s.kin(i0, s.t.size(), vp, ap); ap = s.vacc(i0);
    printf("  swing [%.1f,%.1f] (ref 100+-400 -> [-300,500]); final pos %d; vpk=%.1f (ref %.1f) apk=%.1f (ref amp*w^2=%.1f)\n", lo, hi, (int)isr_pos, vp, 400 * 2 * M_PI / 10.0, ap, 400 * pow(2 * M_PI / 10.0, 2));
    CHECK(isr_pos == 100, "final pos");
    CHECK(hi > 480 && lo < -280 && hi <= 502 && lo >= -302, "amplitude");
    CHECK(ap < amax_d * 1.25, "sine accel");
    CHECK(s.has(" DONE"), "sine DONE");
    // sine replacing a running VEL
    s.send("VEL 300"); s.run_ms(600); s.send("SINE 50 3000 2"); s.run_ms(30);
    for (int k = 0; k < 200 && mode != M_IDLE; k++) { s.send("HB"); s.run_ms(100); }
    printf("  after VEL->SINE: pos %d mode %d  events: %s\n", (int)isr_pos, (int)mode, lines_with(s.out, "E ").substr(0, 0).c_str());
  }
  if (pick == 4 || pick == 99) {
    printf("[HOME]\n");
    Sim s; s.mag_lo = -300; s.mag_hi = -290; s.run_ms(5); s.send("EN 1"); s.send("MOVE 50"); for (int k = 0; k < 20 && mode != M_IDLE; k++) { s.send("HB"); s.run_ms(100); }
    s.send("HOME -1 80"); size_t i0 = s.t.size();
    for (int k = 0; k < 300 && mode != M_IDLE; k++) { s.send("HB"); s.run_ms(100); }
    std::string o = s.out;
    printf("%s", (lines_with(o, "H ") + lines_with(o, "E ")).c_str());
    CHECK(s.has("HOMED"), "no HOMED");
    CHECK(isr_pos > -80 && isr_pos < 0, "pos after rezero %d", (int)isr_pos);
    CHECK(s.has(" -290") || s.has(" -289") || s.has(" -291"), "offset ~ -290");
    (void)i0;
    s.send("HOME -1 80"); s.run_ms(20);
  }
  if (pick == 7 || pick == 99) {
    printf("[HOME active refusal]\n");
    Sim s2; s2.mag_lo = -5; s2.mag_hi = 5; s2.run_ms(5); s2.send("EN 1"); s2.send("HOME 1 80"); s2.run_ms(30);
    CHECK(s2.has("ERR HOME ACTIVE"), "active refusal");
  }
  if (pick == 8 || pick == 99) {
    printf("[HOME no magnet]\n");
    Sim s3; s3.run_ms(5); s3.send("EN 1"); s3.send("HOME 1 200");
    for (int k = 0; k < 200 && mode != M_IDLE; k++) { s3.send("HB"); s3.run_ms(100); }
    printf("  no magnet: pos %d hard %d: %s", (int)isr_pos, (int)HARD_LIM, lines_with(s3.out, "E ").c_str());
    CHECK(isr_pos <= HARD_LIM && isr_pos >= HARD_LIM - 2, "stopped at hard limit");
    CHECK(s3.has(" LIMIT"), "LIMIT event");
  }
  if (pick == 5 || pick == 99) {
    printf("[ESTOP / WATCHDOG / EN]\n");
    Sim s; s.run_ms(5); s.send("EN 1"); s.send("VEL 300"); s.run_ms(500);
    CHECK(!(PORTB & EN_BIT), "driver enabled");
    s.estop_btn = true;
    // time from pin asserted to EN high: run 50 us at a time
    uint32_t t0 = sim_us; int n = 0; while (!(PORTB & EN_BIT) && n < 100) { s.run_ms(0); sim_us += 50; s.drive_inputs(); ISR_TIMER1_COMPA_vect(); loop(); n++; }
    printf("  e-stop latency: %u us (EN high=%d)\n", sim_us - t0, (PORTB & EN_BIT) != 0);
    CHECK(sim_us - t0 <= 1000, "latency");
    s.run_ms(10); s.send("STAT"); s.send("EN 1"); s.send("VEL 100"); s.send("RESET"); s.run_ms(30);
    std::string o = s.out; printf("%s", (lines_with(o, "E ") + lines_with(o, "ERR") + lines_with(o, "OK STAT")).c_str());
    CHECK(s.has("ERR EN ESTOP") && s.has("ERR VEL NOT_READY") && s.has("ERR RESET ESTOP_PIN"), "estop refusals");
    CHECK(isr_inc == 0, "no stepping after estop");
    s.estop_btn = false; s.send("RESET"); s.send("STAT"); s.send("VEL 100"); s.run_ms(30);
    CHECK(s.has("OK RESET") && (PORTB & EN_BIT), "reset keeps driver off");
    CHECK(s.has("ERR VEL NOT_READY"), "disabled after reset");
    s.send("EN 1"); s.send("VEL 300"); s.run_ms(300);
    int32_t p0 = isr_pos; s.run_ms(1200);       // no lines -> watchdog (VEL 300 for 1.5 s then stop)
    s.run_ms(2000);
    printf("  watchdog: %s", lines_with(s.out, "E ").c_str());
    CHECK(s.has("WATCHDOG"), "watchdog");
    CHECK(mode == M_IDLE && v == 0, "stopped by watchdog");
    (void)p0;
    s.send("ESTOP"); s.run_ms(10); CHECK((PORTB & EN_BIT) && s.has("ESTOP CMD"), "ESTOP command");
  }
  if (pick == 6 || pick == 99) {
    printf("[telemetry]\n");
    Sim s; s.run_ms(5); s.send("EN 1"); s.send("STREAM 1"); s.send("VEL 200"); s.run_ms(1000);
    std::string o = s.out; int nT = 0; for (size_t p = 0; (p = o.find("\nT ", p)) != std::string::npos; p++) nT++;
    printf("  T lines in ~1 s: %d\n", nT);
    CHECK(nT > 180 && nT < 210, "telem rate");
  }
  return fails ? 1 : 0;
}
