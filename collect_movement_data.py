"""
Lab 2 - Activity 3: Movement Data Collection
---------------------------------------------
Companion script to the Lab 1 ultrasonic sensor rig. Records short windows
of raw distance readings, reduces each window to a handful of engineered
features, and appends one labeled row per window to movement_data.csv.

Run this ON THE PI (needs RPi.GPIO + the physical sensor). Perform the
movement you're labeling continuously while it counts through the window.

Usage:
    python3 collect_movement_data.py <label> <num_windows>

Example (run once per label, using the categories suggested in the lab sheet):
    python3 collect_movement_data.py not_moving 20
    python3 collect_movement_data.py moving_towards 20
    python3 collect_movement_data.py moving_away 20
    python3 collect_movement_data.py moving_slowly 20

Each run appends to the same CSV, so run it multiple times (once per label)
to build up the full labeled dataset. Aim for roughly equal counts per
label -- an imbalanced dataset (e.g. 80 "not_moving" rows vs 10 of
everything else) will visibly skew accuracy scores later.
"""

import sys
import csv
import time
import os
import numpy as np
import RPi.GPIO as GPIO

PIN_TRIGGER = 7
PIN_ECHO = 11
ECHO_TIMEOUT = 0.05

WINDOW_SIZE = 10          # raw readings per window (= one labeled row)
SAMPLE_INTERVAL = 0.15    # seconds between readings within a window (~6.7 Hz)
OUTPUT_FILE = "movement_data.csv"
FEATURE_NAMES = ["mean_dist", "std_dist", "min_dist", "max_dist", "range_dist", "slope", "delta"]


def setup_gpio():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)
    GPIO.setup(PIN_TRIGGER, GPIO.OUT)
    GPIO.setup(PIN_ECHO, GPIO.IN)
    GPIO.output(PIN_TRIGGER, GPIO.LOW)


def measure_distance():
    """Same sensor routine as the parking/security scripts -- times out
    instead of hanging if the sensor doesn't respond."""
    GPIO.output(PIN_TRIGGER, GPIO.HIGH)
    time.sleep(0.00001)
    GPIO.output(PIN_TRIGGER, GPIO.LOW)

    wait_start = time.time()
    pulse_start = wait_start
    while GPIO.input(PIN_ECHO) == 0:
        pulse_start = time.time()
        if pulse_start - wait_start > ECHO_TIMEOUT:
            return None

    pulse_end = pulse_start
    while GPIO.input(PIN_ECHO) == 1:
        pulse_end = time.time()
        if pulse_end - pulse_start > ECHO_TIMEOUT:
            return None

    duration = pulse_end - pulse_start
    return round(duration * 17150, 2)


def collect_window():
    """Collects WINDOW_SIZE readings (retrying past sensor timeouts) and
    reduces them to a feature vector describing that window."""
    readings = []
    while len(readings) < WINDOW_SIZE:
        dist = measure_distance()
        if dist is not None:
            readings.append(dist)
        time.sleep(SAMPLE_INTERVAL)

    readings = np.array(readings)
    t = np.arange(len(readings))
    slope = np.polyfit(t, readings, 1)[0]  # cm per reading; negative = approaching sensor

    return {
        "mean_dist": round(readings.mean(), 2),
        "std_dist": round(readings.std(), 2),      # low std = steady distance = likely not moving
        "min_dist": round(readings.min(), 2),
        "max_dist": round(readings.max(), 2),
        "range_dist": round(readings.max() - readings.min(), 2),
        "slope": round(float(slope), 3),           # captures speed AND direction of motion
        "delta": round(readings[-1] - readings[0], 2),
    }


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 collect_movement_data.py <label> <num_windows>")
        sys.exit(1)

    label = sys.argv[1]
    num_windows = int(sys.argv[2])

    setup_gpio()
    file_exists = os.path.isfile(OUTPUT_FILE)

    try:
        with open(OUTPUT_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FEATURE_NAMES + ["label"])
            if not file_exists:
                writer.writeheader()

            print(f"Recording {num_windows} windows for label '{label}'.")
            print("Perform the movement continuously -- starting in 3 seconds...")
            time.sleep(3)

            for i in range(num_windows):
                features = collect_window()
                features["label"] = label
                writer.writerow(features)
                f.flush()
                print(f"  [{i + 1}/{num_windows}] {features}")

        print(f"Done. Appended {num_windows} rows to {OUTPUT_FILE}.")
    finally:
        GPIO.cleanup()


if __name__ == "__main__":
    main()
