import json
import re
from typing import Dict, List

class GPUNode:
    def __init__(self, name: str, gpu_type: str, num_slots: int = 8):
        self.name = name
        self.gpu_type = gpu_type
        self.num_slots = num_slots
        self.slots = [None] * num_slots  # State of nodes (Active, Busy, Excluded)
        self.excluded = False
        
    def allocate_slots(self, slot_indices: List[int], sc_id: int):
        """Occupies specific slots with job ID."""
        for idx in slot_indices:
            if 0 <= idx < self.num_slots: 
                self.slots[idx] = sc_id  # slots[0] = 1001
                
    def free_slots(self, slot_indices: List[int]):
        """Releases specified slots."""
        for idx in slot_indices:
            if 0 <= idx < self.num_slots: 
                self.slots[idx] = None
                
    def get_slot_status(self) -> List[str]:
        """Returns visual status of each slot."""
        if self.excluded: return ['excluded'] * self.num_slots
        return ['busy' if slot is not None else 'active' for slot in self.slots]

class GPUPoolManager:
    def __init__(self, pool_config_path: str):
        with open(pool_config_path, 'r') as f:
            self.config = json.load(f)
        self.nodes: Dict[str, GPUNode] = {}
        self._parse_pool_config()
        
    def _parse_pool_config(self):
        """Parses JSON config to initialize GPUNode objects."""
        for pool_name, pool_data in self.config.items():
            if 'nodes' not in pool_data: continue
            for node_spec in pool_data['nodes']:
                is_excluded = node_spec.startswith('~')
                node_spec = node_spec.lstrip('~')
                
                parts = node_spec.split(',')
                node_range = parts[0].strip()
                gpu_type = parts[1].strip() if len(parts) > 1 else 'Unknown'
                
                for node_name in self._expand_node_names(node_range):
                    if node_name not in self.nodes:
                        self.nodes[node_name] = GPUNode(node_name, gpu_type)
                    if is_excluded: 
                        self.nodes[node_name].excluded = True

    def _expand_node_names(self, node_range: str) -> List[str]:
        """Expands range notation like 'h[1-3]' to ['h1', 'h2', 'h3']."""
        range_match = re.match(r'([a-zA-Z]+)\[(\d+)-(\d+)\]', node_range)
        if range_match:
            prefix, start, end = range_match.group(1), int(range_match.group(2)), int(range_match.group(3))
            return [f"{prefix}{i}" for i in range(start, end + 1)]
        return [node_range.strip()]