class Router:

    def __init__(self, memoryLimit: int):
        self.memory_limit = memoryLimit
        self.packet_queue = deque()
        self.unique_packets = set()
        self.destination_timestamps = defaultdict(list)
        self.processed_packet_index = defaultdict(int)
    def _remove_oldest_packet(self):
        oldest_packet: Packet = self.packet_queue.popleft()
        self.unique_packets.remove(oldest_packet)
        destination = oldest_packet[1]
        self.processed_packet_index[destination] += 1


    def addPacket(self, source: int, destination: int, timestamp: int) -> bool:
        packet: Packet = (source, destination, timestamp)
        if packet in self.unique_packets:
            return False
        if len(self.packet_queue) == self.memory_limit:
            self._remove_oldest_packet()
        self.packet_queue.append(packet)
        self.unique_packets.add(packet)
        self.destination_timestamps[destination].append(timestamp)
        return True
 
        

    def forwardPacket(self) -> List[int]:
        if not self.packet_queue:
            return []
        oldest_packet: Packet = self.packet_queue.popleft()
        self.unique_packets.remove(oldest_packet)
        destination = oldest_packet[1]
        self.processed_packet_index[destination] += 1
        return list(oldest_packet)

        

    def getCount(self, destination: int, startTime: int, endTime: int) -> int:
        if destination not in self.destination_timestamps:
            return 0
        all_timestamps = self.destination_timestamps[destination]
        start_search_index = self.processed_packet_index[destination]
        def binary_search_left(arr, x, start):
            low, high = start, len(arr)
            ans = high
            while low <= high:
                mid = (low + high) // 2
                if mid < len(arr) and arr[mid] >= x:
                    ans = mid
                    high = mid - 1
                else:
                    low = mid + 1
            return ans
        def binary_search_right(arr, x, start):
            low, high = start, len(arr)
            ans = low
            while low <= high:
                mid = (low + high) // 2
                if mid < len(arr) and arr[mid] <= x:
                    ans = mid + 1
                    low = mid + 1
                else:
                    high = mid - 1
            return ans
        lower_bound_index = binary_search_left(all_timestamps, startTime, start_search_index)
        upper_bound_index = binary_search_right(all_timestamps, endTime, start_search_index)
        return upper_bound_index - lower_bound_index

# Your Router object will be instantiated and called as such:
# obj = Router(memoryLimit)
# param_1 = obj.addPacket(source,destination,timestamp)
# param_2 = obj.forwardPacket()
# param_3 = obj.getCount(destination,startTime,endTime)
