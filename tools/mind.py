#!/usr/bin/env python3
"""Send a text seed to esp32-mind and print its on-device continuation."""

import argparse
import time
from pathlib import Path

import serial
from serial.tools import list_ports
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
ESPRESSIF_VID = 0x303A
USB_SERIAL_JTAG_PID = 0x1001
END_MARKER = b"\nMIND END "


def find_port() -> str:
    matches = [
        port.device
        for port in list_ports.comports()
        if port.vid == ESPRESSIF_VID and port.pid == USB_SERIAL_JTAG_PID
    ]
    if not matches:
        raise RuntimeError("no Espressif USB Serial/JTAG device found")
    if len(matches) > 1:
        raise RuntimeError(f"multiple Espressif devices found: {', '.join(matches)}")
    return matches[0]


def open_port(path: str) -> serial.Serial:
    device = serial.Serial()
    device.port = path
    device.baudrate = 115200
    device.timeout = 0.1
    device.write_timeout = 5
    device.dtr = False
    device.rts = False
    device.open()
    return device


def wait_for_pong(device: serial.Serial, timeout: float = 8.0) -> None:
    deadline = time.monotonic() + timeout
    device.reset_input_buffer()
    while time.monotonic() < deadline:
        device.write(b"PING\n")
        attempt_deadline = min(deadline, time.monotonic() + 0.75)
        while time.monotonic() < attempt_deadline:
            line = device.readline()
            if line.startswith(b"MIND PONG"):
                return
        time.sleep(0.1)
    raise TimeoutError("device did not answer PING")


def build_command(token_ids: list[int], max_tokens: int) -> bytes:
    if not token_ids:
        raise ValueError("prompt encoded to zero tokens")
    if len(token_ids) > 96:
        raise ValueError(f"prompt is {len(token_ids)} tokens; device maximum is 96")
    ids = " ".join(str(token_id) for token_id in token_ids)
    return f"GENERATE {max_tokens} {ids}\n".encode()


def read_until_begin(device: serial.Serial, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        line = device.readline()
        if line.startswith(b"MIND BEGIN"):
            return
        if line.startswith(b"MIND ERROR"):
            raise RuntimeError(line.decode(errors="replace").strip())
    raise TimeoutError("device did not start generation")


def read_completion(device: serial.Serial, timeout: float) -> tuple[str, str]:
    deadline = time.monotonic() + timeout
    response = bytearray()
    while time.monotonic() < deadline:
        available = device.in_waiting
        chunk = device.read(available if available else 1)
        if not chunk:
            continue
        response.extend(chunk)
        marker_at = response.find(END_MARKER)
        if marker_at < 0:
            continue

        completion = bytes(response[:marker_at]).decode("utf-8", errors="replace")
        stats = bytearray(response[marker_at + len(END_MARKER) :])
        while b"\n" not in stats and time.monotonic() < deadline:
            stats.extend(device.read(device.in_waiting or 1))
        stats_line = bytes(stats).split(b"\n", 1)[0].decode(errors="replace")
        return completion, f"MIND END {stats_line}"
    raise TimeoutError("generation did not finish")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="+", help="TinyStories seed text")
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--port")
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=ROOT / "data" / "bpe16384.json",
    )
    args = parser.parse_args()

    if not 1 <= args.max_tokens <= 128:
        parser.error("--max-tokens must be between 1 and 128")

    prompt = " ".join(args.prompt)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    token_ids = tokenizer.encode(prompt).ids
    command = build_command(token_ids, args.max_tokens)
    port = args.port or find_port()

    print(f"device: {port}")
    print(f"prompt: {prompt!r} ({len(token_ids)} tokens)")
    with open_port(port) as device:
        wait_for_pong(device)
        device.reset_input_buffer()
        device.write(command)
        read_until_begin(device, timeout=5)
        completion, stats = read_completion(device, timeout=120)

    print(f"\nmind> {completion}")
    print(stats)


if __name__ == "__main__":
    main()
