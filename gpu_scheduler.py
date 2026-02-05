import streamlit as st
import pandas as pd
import numpy as np
import json
import re
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple

# --- 기존 클래스 로직 (GPUNode, GPUPoolManager) ---

class GPUNode:
    def __init__(self, name: str, gpu_type: str, num_slots: int = 8):
        self.name = name
        self.gpu_type = gpu_type
        self.num_slots = num_slots
        self.slots = [None] * num_slots
        self.excluded = False
        
    def allocate_slots(self, slot_indices: List[int], sc_id: int):
        for idx in slot_indices:
            if 0 <= idx < self.num_slots: self.slots[idx] = sc_id
                
    def free_slots(self, slot_indices: List[int]):
        for idx in slot_indices:
            if 0 <= idx < self.num_slots: self.slots[idx] = None
                
    def get_slot_status(self) -> List[str]:
        if self.excluded: return ['excluded'] * self.num_slots
        return ['busy' if slot is not None else 'active' for slot in self.slots]

class GPUPoolManager:
    def __init__(self, pool_config_path: str):
        with open(pool_config_path, 'r') as f:
            self.config = json.load(f)
        self.nodes: Dict[str, GPUNode] = {}
        self._parse_pool_config()
        
    def _parse_pool_config(self):
        for pool_name, pool_data in self.config.items():
            if 'nodes' not in pool_data: continue
            for node_spec in pool_data['nodes']:
                is_excluded = node_spec.startswith('~')
                node_spec = node_spec.lstrip('~')
                parts = node_spec.split(',')
                node_range = parts[0].strip()
                gpu_type = parts[1].strip() if len(parts) > 1 else 'Unknown'
                node_names = self._parse_node_range(node_range)
                for node_name in node_names:
                    if node_name not in self.nodes:
                        self.nodes[node_name] = GPUNode(node_name, gpu_type)
                    if is_excluded: self.nodes[node_name].excluded = True
                        
    def _parse_node_range(self, node_range: str) -> List[str]:
        match = re.match(r'(\w+)\[(\d+)-(\d+)\]', node_range)
        if match:
            prefix, start, end = match.group(1), int(match.group(2)), int(match.group(3))
            return [f"{prefix}{i}" for i in range(start, end + 1)]
        return [node_range]

# --- 시뮬레이션 상태 관리 함수 ---

def init_simulator(log_path, config_path):
    df = pd.read_csv(log_path)
    # 컬럼명 보정 (SC_ID 등 확인)
    manager = GPUPoolManager(config_path)
    
    events = []
    for _, row in df.iterrows():
        events.append((row['SUBMIT_TIME'], 'submit', row.to_dict()))
        events.append((row['START_TIME'], 'start', row.to_dict()))
        events.append((row['FINISH_TIME'], 'finish', row.to_dict()))
    events.sort(key=lambda x: x[0])
    
    return {
        'events': events,
        'current_idx': 0,
        'backlog': [],
        'nodes': manager.nodes,
        'current_time': events[0][0] if events else 0
    }

def parse_exec_host(exec_host: str):
    result = []
    if pd.isna(exec_host): return result
    hosts = str(exec_host).split('|')
    for host in hosts:
        match = re.match(r'(\w+)\[([\d,]+)\]', host.strip())
        if match:
            node_name = match.group(1)
            slots = [int(x) for x in match.group(2).split(',')]
            result.append((node_name, slots))
    return result

def step_forward(state):
    if state['current_idx'] < len(state['events']):
        ts, etype, data = state['events'][state['current_idx']]
        state['current_time'] = ts
        
        if etype == 'submit':
            state['backlog'].append(data)
        elif etype == 'start':
            state['backlog'] = [j for j in state['backlog'] if j['SC_ID'] != data['SC_ID']]
            for node_name, slots in parse_exec_host(data['EXEC_HOST']):
                if node_name in state['nodes']:
                    state['nodes'][node_name].allocate_slots(slots, data['SC_ID'])
        elif etype == 'finish':
            for node_name, slots in parse_exec_host(data['EXEC_HOST']):
                if node_name in state['nodes']:
                    state['nodes'][node_name].free_slots(slots)
        
        state['current_idx'] += 1
        return True
    return False

# --- Streamlit UI 구성 ---

st.set_page_config(layout="wide", page_title="GPU Scheduler Simulator")

st.sidebar.title("Configuration")
log_file = st.sidebar.text_input("Log CSV Path", "./RL_based_gpu_scheduling/data/20250914-20251101_logdata__anon.csv")
conf_file = st.sidebar.text_input("Config JSON Path", "./RL_based_gpu_scheduling/data/pool_conf_250912_anon.json")

if st.sidebar.button("Initialize Simulator"):
    st.session_state.state = init_simulator(log_file, conf_file)
    st.success("Simulator Initialized!")

if 'state' in st.session_state:
    state = st.session_state.state
    
    # 상단 컨트롤바
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.write(f"### 🕒 Current Time: `{datetime.fromtimestamp(state['current_time'])}`")
    with c2:
        if st.button("▶ Next Step (1 Event)"):
            step_forward(state)
    with c3:
        step_count = st.number_input("Bulk Steps", min_value=1, value=50)
        if st.button(f"⏩ Run {step_count} Steps"):
            for _ in range(step_count): step_forward(state)

    # 메인 화면 구성
    col_main, col_backlog = st.columns([3, 1])

    with col_main:
        # GPU 타입별 필터링
        gpu_types = sorted(list(set(n.gpu_type for n in state['nodes'].values())))
        tabs = st.tabs(gpu_types)
        
        for i, g_type in enumerate(gpu_types):
            with tabs[i]:
                relevant_nodes = [n for n in state['nodes'].values() if n.gpu_type == g_type][:30] # 성능상 30개 제한
                for node in relevant_nodes:
                    slots = node.get_slot_status()
                    # 이모지를 이용한 시각화 (Active: 🟩, Busy: 🟨, Excluded: 🟥)
                    slot_str = "".join(["🟩" if s=='active' else "🟨" if s=='busy' else "🟥" for s in slots])
                    st.write(f"`{node.name:7}` {slot_str}")

    with col_backlog:
        st.write("### 📋 Backlog")
        if state['backlog']:
            df_back = pd.DataFrame(state['backlog'])[['SC_ID', 'QUEUE', 'SLOTS']].head(20)
            st.table(df_back)
        else:
            st.info("No pending jobs")

else:
    st.info("Please initialize the simulator from the sidebar.")