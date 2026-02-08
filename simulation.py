import pandas as pd
import re
from typing import List, Tuple
from model import GPUPoolManager

def parse_exec_host(exec_host: str) -> List[Tuple[str, List[int]]]:
    """Parses 'node[0,1]' string into (node_name, [slot_indices])."""
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

def init_simulator(log_path: str, config_path: str):
    """Initializes the simulation state and event timeline."""
    try:
        df = pd.read_csv(log_path)
    except Exception:
        df = pd.read_csv(log_path, encoding='cp949')

    manager = GPUPoolManager(config_path)
    events = []
    for _, row in df.iterrows():
        job_data = row.to_dict()
        events.append((row['SUBMIT_TIME'], 'submit', job_data))
        events.append((row['START_TIME'], 'start', job_data))
        events.append((row['FINISH_TIME'], 'finish', job_data))
    
    # Sort by time; prioritize 'submit' on simultaneous timestamps
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
    """Processes the next single event in the timeline."""
    if state['current_idx'] < state['total_events']:
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