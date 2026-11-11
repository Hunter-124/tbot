// Minimal host-side stand-in for Arduino.h + avr-libc, only what calrig_uno.ino uses.
// Used by test/sim_host.cpp to compile and exercise the firmware on a PC.  NOT part of the firmware build.
#pragma once
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <string>
#include <deque>

#define _BV(b) (1u << (b))
#define F_CPU 16000000UL
#define PROGMEM
#define PSTR(x) (x)
#define pgm_read_byte(p) (*(const uint8_t *)(p))
#define strcmp_P strcmp

extern volatile uint8_t PORTB, DDRB, PINB, PORTC, DDRC, PINC, PORTD, DDRD, PIND, SREG;
extern volatile uint8_t TCCR1A, TCCR1B, TIMSK1, PCICR, PCMSK0;
extern volatile uint16_t OCR1A, TCNT1;
enum { WGM12 = 3, CS10 = 0, OCIE1A = 1, PCIE0 = 0, PCINT1 = 1 };
static inline void cli() {}
static inline void sei() {}

extern void ISR_TIMER1_COMPA_vect();
extern void ISR_PCINT0_vect();
#define ISR(vec) void ISR_##vec()

extern uint32_t sim_us;
static inline uint32_t micros() { return sim_us; }
static inline uint32_t millis() { return sim_us / 1000; }

struct SerialStub {
  std::deque<uint8_t> rx;
  std::string tx;
  size_t tx_space = 64;                        // emulate the 64 byte UART buffer being drained by the sim
  void begin(unsigned long) {}
  int available() { return (int)rx.size(); }
  int read() { if (rx.empty()) return -1; int c = rx.front(); rx.pop_front(); return c; }
  int availableForWrite() { return (int)tx_space; }
  size_t write(uint8_t c) { tx.push_back((char)c); if (tx_space) tx_space--; return 1; }
};
extern SerialStub Serial;
void setup();
void loop();
