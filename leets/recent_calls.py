""" this code example is based on leet i saw on leetcode 
933. Number of Recent Calls """
import socket
import struct
import time
from collections import deque
import select


# Constants 
ICMP_ECHO = 8
ICMP_CODE = 0
ICMP_REPLY = 0
TARGET = "8.8.8.8"
TIME_WINDOW = 3.0
PING_INTERVAL = 1.0

MAX_PINGS = 300
timestamps = deque()

# Rrrrrrrrrrraaaawww Soockets
sock = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_ICMP)
sock.setblocking(False)

def checksum(data):
    s = 0
    n = len(data) % 2
    for i in range(0, len(data) - n, 2):
        s += (data[i] << 8) + data[i + 1]
    if n:
        s += data[-1] << 8
    while (s >> 16):
        s = (s & 0xFFFF) + (s >> 16)
    s = ~s & 0xFFFF
    return s

seq = 1
identifier = 0x1234



def build_packet(seq):
    header = struct.pack('!BBHHH', ICMP_ECHO, ICMP_CODE, 0, identifier, seq)
    payload = struct.pack('d', time.time())
    chksum = checksum(header + payload)
    header = struct.pack('!BBHHH', ICMP_ECHO, ICMP_CODE, chksum, identifier, seq)
    return header + payload

def send_ping():
    global seq 
    packet = build_packet(seq)
    sock.sendto(packet, (TARGET, 0))
    seq += 1

def recv_reply(timeout=0.01):
    try:
        ready, _, _ = select.select([sock], [], [], timeout)
        if not ready:
            return

        while True:
            try:
                data, addr = sock.recvfrom(1024)
            except BlockingIOError:
                break  # No more data to read

            if len(data) < 28:
                break  # Corrupted 
            icmp_header = data[20:28] # skipping IPv4 header 
            _type, _code, _chk, _id, _seq = struct.unpack('!BBHHH', icmp_header)
            if _type == ICMP_REPLY and _id == identifier:
                now = time.time()
                timestamps.append(now)
            else:
                break # parsing only the first relevant one
    except Exception as e:
        print(f"recv_reply error: {e}")

def count_recent(): # Question from leetcode regarding 3000 requests
    now = time.time()
    while timestamps and now - timestamps[0] > TIME_WINDOW:
        timestamps.popleft()
    return len(timestamps)

def ping_loop():
    print(f"Pinging {TARGET} (Pure Python raw socket)...")
    while True:
        send_ping()
        time.sleep(PING_INTERVAL)
        for _ in range(10):
            recv_reply(timeout=0.01)
        print(f"Replies in last {TIME_WINDOW:.1f}s: {count_recent()}")


if __name__ == "__main__":
    ping_loop()