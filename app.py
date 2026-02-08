import streamlit as st
import pandas as pd
from collections import defaultdict
from simulation import init_simulator, step_forward

st.set_page_config(layout="wide", page_title="GPU Scheduler Simulator")

# Sidebar Configuration
st.sidebar.title("🛠️ Configuration")
log_path = st.sidebar.text_input("Log CSV Path", "./RL_based_gpu_scheduling/data/20250914-20251101_logdata__anon.csv")
conf_path = st.sidebar.text_input("Config JSON Path", "./RL_based_gpu_scheduling/data/pool_conf_250912_anon.json")

if st.sidebar.button("🚀 Initialize Simulator"):
    st.session_state.state = init_simulator(log_path, conf_path)
    if st.session_state.state:
        st.sidebar.success("Completed Initialization!")

# Main Simulation UI
if 'state' in st.session_state:
    state = st.session_state.state
    
    # Controls
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.subheader(f"🕒 Simulation Tick: `{state['current_time']}`")
        progress = state['current_idx'] / state['total_events'] if state['total_events'] > 0 else 0
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

    # Cluster Status & Backlog
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
        for i, gt in enumerate(available_types):
            with tabs[i]:
                for node in nodes_by_type[gt]:
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