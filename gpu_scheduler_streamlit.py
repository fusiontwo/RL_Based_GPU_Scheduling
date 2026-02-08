import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import os
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple

# GPU Node
class GPUNode:
    def __init__(self, name: str, gpu_type: str, num_slots: int = 8):
        '''Initialize a GPU node with given name, type, and number of slots.'''
        self.name = name
        self.gpu_type = gpu_type
        self.num_slots = num_slots
        self.slots = [None] * num_slots
        self.excluded = False
        
    def allocate_slots(self, slot_indices: List[int], sc_id: int):
        '''Allocate specified slots to a job identified by sc_id.'''
        for idx in slot_indices:
            if 0 <= idx < self.num_slots: 
                self.slots[idx] = sc_id
                
    def free_slots(self, slot_indices: List[int]):
        '''Free specified slots.'''
        for idx in slot_indices:
            if 0 <= idx < self.num_slots: 
                self.slots[idx] = None
                
    def get_slot_status(self) -> List[str]:
        '''Get the status of each slot in the node.'''
        if self.excluded: return ['excluded'] * self.num_slots
        return ['busy' if slot is not None else 'active' for slot in self.slots]

class GPUPoolManager:
    def __init__(self, pool_config_path: str):
        '''Initialize the GPU pool manager with configuration from a JSON file.'''
        with open(pool_config_path, 'r') as f:
            self.config = json.load(f)
        self.nodes: Dict[str, GPUNode] = {}
        self._parse_pool_config()
        
    def _parse_pool_config(self):
        '''Parse the pool configuration and initialize GPU nodes.'''
        for pool_name, pool_data in self.config.items():
            if 'nodes' not in pool_data: continue
            for node_spec in pool_data['nodes']:
                is_excluded = node_spec.startswith('~')
                node_spec = node_spec.lstrip('~')
                
                # Parsing like "h[9235-9548], H100"
                parts = node_spec.split(',')
                node_range = parts[0].strip()
                gpu_type = parts[1].strip() if len(parts) > 1 else 'Unknown'
                
                node_names = self._expand_node_names(node_range)

                for node_name in node_names:
                    # Generate or update GPUNode
                    if node_name not in self.nodes:
                        self.nodes[node_name] = GPUNode(node_name, gpu_type)
                    else:
                        # Update GPU type if node already exists
                        self.nodes[node_name].gpu_type = gpu_type
                    
                    # Excluded logic 
                    if is_excluded: 
                        self.nodes[node_name].excluded = True

    def _expand_node_names(self, node_range: str) -> List[str]:
            range_match = re.match(r'([a-zA-Z]+)\[(\d+)-(\d+)\]', node_range)
            if range_match:
                prefix, start, end = range_match.group(1), int(range_match.group(2)), int(range_match.group(3))
                return [f"{prefix}{i}" for i in range(start, end + 1)]
            
            return [node_range.strip()]

# Parsing data & Simulation
def parse_exec_host(exec_host: str) -> List[Tuple[str, List[int]]]:
    result = []
    if pd.isna(exec_host): return result
    hosts = str(exec_host).split('|')
    for host in hosts:
        match = re.match(r'(\w+)\[([\d,]+)\]', host.strip())
        if match:
            node_name = match.group(1)
            slots = [int(s) for s in match.group(2).split(',')]
            result.append((node_name, slots))
    return result

def init_simulator(log_path, config_path):
    '''Initialize the simulator state from log and config files.'''
    if not os.path.exists(log_path) or not os.path.exists(config_path):
        st.error(f"파일을 찾을 수 없습니다.")
        return None

    # Convert the log file into a DataFrame
    try:
        df = pd.read_csv(log_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(log_path, encoding='cp949')

    # Initialize GPU pool manager
    manager = GPUPoolManager(config_path)
    events = []
    for _, row in df.iterrows():
        job_data = row.to_dict()
        events.append((row['SUBMIT_TIME'], 'submit', job_data))
        events.append((row['START_TIME'], 'start', job_data))
        events.append((row['FINISH_TIME'], 'finish', job_data))
    
    events.sort(key=lambda x: (x[0], 0 if x[1]=='submit' else 1))
    
    return {
        'events': events,
        'current_idx': 0,
        'backlog': [],
        'nodes': manager.nodes,
        'current_time': events[0][0] if events else 0,
        'total_events': len(events)
    }

def step_forward(state):
    if state['current_idx'] < state['total_events']:
        ts, etype, data = state['events'][state['current_idx']]
        state['current_time'] = ts
        
        if etype == 'submit':
            state['backlog'].append(data)
        elif etype == 'start':
            state['backlog'] = [j for j in state['backlog'] if j['SC_ID'] != data['SC_ID']]
            assignments = parse_exec_host(data['EXEC_HOST'])
            for node_name, slots in assignments:
                if node_name in state['nodes']:
                    state['nodes'][node_name].allocate_slots(slots, data['SC_ID'])
        elif etype == 'finish':
            assignments = parse_exec_host(data['EXEC_HOST'])
            for node_name, slots in assignments:
                if node_name in state['nodes']:
                    state['nodes'][node_name].free_slots(slots)
        
        state['current_idx'] += 1
        return True
    return False

# UI layout
st.set_page_config(layout="wide", page_title="GPU Scheduler Simulator")

st.sidebar.title("🛠️ Configuration")
log_path = st.sidebar.text_input("Logdata CSV Path", "./RL_based_gpu_scheduling/data/20250914-20251101_logdata__anon.csv")
conf_path = st.sidebar.text_input("Config JSON Path", "./RL_based_gpu_scheduling/data/pool_conf_250912_anon.json")

if st.sidebar.button("🚀 Initialize Simulator"):
    st.session_state.state = init_simulator(log_path, conf_path)
    if st.session_state.state:
        st.sidebar.success("Completed Initialization!")

if 'state' in st.session_state:
    state = st.session_state.state
    
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.subheader(f"🕒 Simulation Tick: `{state['current_time']}`")
        progress = state['current_idx'] / state['total_events']
        st.progress(progress)
    with c2:
        if st.button("▶ Next Step"):
            step_forward(state)
            st.rerun()
    with c3:
        bulk = st.number_input("Bulk Steps", 1, 5000, 100)
        if st.button(f"⏩ Run {bulk} Steps"):
            for _ in range(bulk):
                if not step_forward(state): break
            st.rerun()

    st.divider()

    # Change Layout
    m_col, s_col = st.columns([1, 2]) 
    
    with m_col:
        st.subheader("🖥️ Cluster Live Status")
        nodes_by_type = defaultdict(list)
        for node in state['nodes'].values():
            nodes_by_type[node.gpu_type].append(node)
        
        display_order = ['V100', 'A100', 'H100']
        available_types = [t for t in display_order if t in nodes_by_type]
        available_types += [t for t in sorted(nodes_by_type.keys()) if t not in display_order]
        
        tabs = st.tabs([f"{gt}" for gt in available_types])
        
        # Generate tab contents
        for i, gt in enumerate(available_types):
            with tabs[i]:
                display_nodes = nodes_by_type[gt]
                for node in display_nodes:
                    status = node.get_slot_status()
                    viz = "".join(["🟩" if s=='active' else "🟨" if s=='busy' else "🟥" for s in status])
                    st.write(f"`{node.name:7}` {viz}")

    with s_col:
        st.subheader("📋 Backlog Details")
        if state['backlog']:
            bl_df = pd.DataFrame(state['backlog'])
            bl_df['WAIT_TIME'] = state['current_time'] - bl_df['SUBMIT_TIME']
            
            columns_order = [
                'SC_ID', 'QUEUE', 'SLOTS', 'EXEC_HOST', 
                'SUBMIT_TIME', 'START_TIME', 'FINISH_TIME', 'WAIT_TIME'
            ]
            
            display_cols = [c for c in columns_order if c in bl_df.columns]
            
            st.dataframe(bl_df[display_cols], width='stretch', hide_index=True)
        else:
            st.info("No pending jobs in backlog.")

else:
    st.warning("👈 Click the initialization button on the left sidebar to start the simulation.")