"""Serial receiver for robot-base odometry stream."""

from __future__ import annotations

import argparse

import serial

from echo import parse_odometry_line


def listen_serial(port: str, baud: int = 115200) -> None:
    """Read and print odometry CSV from ESP32 firmware."""
    with serial.Serial(port, baud, timeout=1) as ser:
        print(f"Listening on {port} @ {baud} baud (Ctrl+C to stop)")
        while True:
            line = ser.readline().decode("utf-8", errors="ignore")
            parsed = parse_odometry_line(line)
            if parsed is None:
                continue
            ts, left, right, heading, servo = parsed
            print(
                f"t={ts}ms  L={left} R={right}  "
                f"heading={heading:.1f}deg  servo={servo:.1f}deg"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="EchoMap odometry receiver")
    parser.add_argument("--port", required=True, help="Serial port, e.g. /dev/cu.usbserial-*")
    parser.add_argument("--baud", type=int, default=115200)
    args = parser.parse_args()

    try:
        listen_serial(args.port, args.baud)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
