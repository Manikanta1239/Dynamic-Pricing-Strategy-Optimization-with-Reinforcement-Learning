import streamlit as st
from main import DynamicPricingSystem
import asyncio
import nest_asyncio
import torch.multiprocessing as mp
import sys
import torch
import time
import plotly.express as px
import pandas as pd

# Prevent Streamlit file watcher from breaking on torch internals
sys.modules['torch.classes'].__path__ = []

# Fix event loop and multiprocessing issues
nest_asyncio.apply()
if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn', force=True)

# Configure page
st.set_page_config(
    page_title="Dynamic Pricing RL App", 
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Page configuration
    st.title("🤖 Dynamic Pricing System")
    st.markdown("""
    This application uses Reinforcement Learning to optimize product pricing strategies.
    Follow the steps below to analyze and optimize your pricing.
    """)
    
    # Initialize session state variables
    if 'system' not in st.session_state:
        st.session_state.system = None
    if 'training_progress' not in st.session_state:
        st.session_state.training_progress = 0
    if 'current_episode' not in st.session_state:
        st.session_state.current_episode = 0
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # URL inputs with validation
        amazon_url = st.text_input("Amazon URL", help="Enter the full Amazon product URL")
        flipkart_url = st.text_input("Flipkart URL", help="Enter the full Flipkart product URL")
        
        # Advanced settings expandable section
        with st.expander("🛠️ Advanced Settings"):
            training_episodes = st.slider("Training Episodes", 10, 500, 100, 10)
            learning_rate = st.select_slider(
                "Learning Rate",
                options=[0.0001, 0.001, 0.01, 0.1],
                value=0.001,
                format_func=lambda x: f"{x:.4f}"
            )
            use_scraping = st.checkbox("Enable Web Scraping", value=True)
        
        # Initialize button with loading state
        if st.button("🚀 Initialize System", use_container_width=True):
            urls = {}
            if amazon_url:
                urls['amazon'] = amazon_url
            if flipkart_url:
                urls['flipkart'] = flipkart_url
                
            with st.spinner("Initializing system..."):
                try:
                    st.session_state.system = DynamicPricingSystem(
                        product_urls=urls,
                        use_scraping=use_scraping
                    )
                    st.success("✅ System initialized successfully!")
                except Exception as e:
                    st.error(f"❌ Initialization failed: {str(e)}")
    
    # Main content area
    if st.session_state.system:
        # Progress tracker
        progress_placeholder = st.empty()
        col1, col2, col3, col4, col5 = st.columns(5)
        steps = ["Data", "Environment", "Training", "Evaluation", "Deployment"]
        step_status = {step: "⭕" for step in steps}
        
        # Data collection
        with col1:
            if st.button("1️⃣ Collect Data", use_container_width=True):
                step_status["Data"] = "⏳"
                progress_placeholder.write(f"Progress: {' '.join(step_status.values())}")
                with st.spinner("Collecting data..."):
                    st.session_state.system.collect_data()
                step_status["Data"] = "✅"
                progress_placeholder.write(f"Progress: {' '.join(step_status.values())}")
        
        # Environment preparation
        with col2:
            if st.button("2️⃣ Prepare", use_container_width=True):
                step_status["Environment"] = "⏳"
                progress_placeholder.write(f"Progress: {' '.join(step_status.values())}")
                with st.spinner("Preparing..."):
                    env = st.session_state.system.prepare_environment()
                    st.session_state.system.setup_agents()
                step_status["Environment"] = "✅"
                progress_placeholder.write(f"Progress: {' '.join(step_status.values())}")
        
        # Training section with real-time updates
        with col3:
            if st.button("3️⃣ Train", use_container_width=True):
                step_status["Training"] = "⏳"
                progress_placeholder.write(f"Progress: {' '.join(step_status.values())}")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for episode in range(training_episodes):
                    st.session_state.current_episode = episode
                    progress = (episode + 1) / training_episodes
                    progress_bar.progress(progress)
                    status_text.text(f"Training Episode: {episode + 1}/{training_episodes}")
                    st.session_state.system.train_agent(num_episodes=1)
                    
                step_status["Training"] = "✅"
                progress_placeholder.write(f"Progress: {' '.join(step_status.values())}")
        
        # Evaluation with interactive charts
        with col4:
            if st.button("4️⃣ Evaluate", use_container_width=True):
                step_status["Evaluation"] = "⏳"
                progress_placeholder.write(f"Progress: {' '.join(step_status.values())}")
                
                with st.spinner("Evaluating strategies..."):
                    report = st.session_state.system.evaluate_strategies()
                    
                    # Display interactive results
                    st.subheader("Evaluation Results")
                    
                    # Convert results to DataFrame for plotting
                    results_df = pd.DataFrame(report['summary'])
                    
                    # Interactive plotly charts
                    fig = px.bar(results_df, 
                                title="Strategy Comparison",
                                labels={"value": "Performance", "variable": "Metric"})
                    st.plotly_chart(fig)
                
                step_status["Evaluation"] = "✅"
                progress_placeholder.write(f"Progress: {' '.join(step_status.values())}")
        
        # Deployment with interactive scenarios
        with col5:
            if st.button("5️⃣ Deploy", use_container_width=True):
                step_status["Deployment"] = "⏳"
                progress_placeholder.write(f"Progress: {' '.join(step_status.values())}")
                
                results = st.session_state.system.deploy_optimal_strategy()
                
                # Interactive results viewer
                st.subheader("Deployment Scenarios")
                for i, result in enumerate(results):
                    with st.expander(f"Scenario {i+1}"):
                        st.write(result)
                
                step_status["Deployment"] = "✅"
                progress_placeholder.write(f"Progress: {' '.join(step_status.values())}")
        
        # Visualization section with tabs
        st.subheader("📊 Analysis & Visualizations")
        tab1, tab2, tab3 = st.tabs(["Price Analysis", "Profit Analysis", "Agent Performance"])
        
        with tab1:
            try:
                st.image("screenshots\price_comparison.png", caption="Price Comparison Across Strategies")
            except FileNotFoundError:
                st.info("Price analysis not available yet. Complete the evaluation step first.")
        
        with tab2:
            try:
                st.image("screenshots\profit_comparison.png", caption="Profit Comparison")
            except FileNotFoundError:
                st.info("Profit analysis not available yet. Complete the evaluation step first.")
        
        with tab3:
            for agent in ["PPO", "Static", "Rule-Based"]:
                try:
                    st.image(f"screenshots\demand_vs_price_{agent}.png", caption=f"Demand vs Price – {agent}")
                except FileNotFoundError:
                    st.info(f"{agent} performance visualization not available yet.")

if __name__ == "__main__":
    main()
