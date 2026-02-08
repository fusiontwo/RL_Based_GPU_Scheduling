import streamlit as st
import pandas as pd
from collections import defaultdict
from simulation import init_simulator, step_forward

st.set_page_config(layout="wide", page_title="GPU Scheduler Simulator")

# Sidebar Configuration
st.sidebar.title("🛠️ Configuration")
log_path = st.sidebar.text_input("Log CSV Path", "./RL_based_gpu_scheduling/data/20250914-20251101_logdata__anon.csv")
conf_path = st.sidebar.text_input("Config JSON Path", "./RL_based_gpu_scheduling/data/pool_conf_250912_anon.json")

if st.sidebar.button("🚀 Initialize"):
    st.session_state.state = init_simulator(log_path, conf_path)

# Main Simulation UI
if 'state' in st.session_state:
    state = st.session_state.state
    
    # Controls
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.subheader(f"🕒 Time: `{state['current_time']}`")
        st.progress(state['current_idx'] / state['total_events'])
    with c2:
        if st.button("▶ Next"):
            step_forward(state)
            st.rerun()
    with c3:
        bulk = st.number_input("Bulk", 1, 5000, 100)
        if st.button(f"⏩ Run {bulk}"):
            for _ in range(bulk):
                if not step_forward(state): break
            st.rerun()

    st.divider()

    # Cluster Status & Backlog
    m_col, s_col = st.columns([1, 2]) 
    
    with m_col:
        st.subheader("🖥️ Node Status")
        nodes_by_type = defaultdict(list)
        for node in state['nodes'].values():
            nodes_by_type[node.gpu_type].append(node)
        
        tabs = st.tabs(list(nodes_by_type.keys()))
        for i, gt in enumerate(nodes_by_type.keys()):
            with tabs[i]:
                for node in nodes_by_type[gt]:
                    status = node.get_slot_status()
                    viz = "".join(["🟩" if s=='active' else "🟨" if s=='busy' else "🟥" for s in status])
                    st.write(f"`{node.name:7}` {viz}")

    with s_col:
        st.subheader("📋 Backlog")
        if state['backlog']:
            bl_df = pd.DataFrame(state['backlog'])
            bl_df['WAIT'] = state['current_time'] - bl_df['SUBMIT_TIME']
            st.dataframe(bl_df[['SC_ID', 'SLOTS', 'WAIT']], use_container_width=True)
        else:
            st.info("Empty")