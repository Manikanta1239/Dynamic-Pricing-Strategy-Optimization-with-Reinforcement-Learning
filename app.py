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
import logging
import warnings
from absl import app as absl_app
from absl import logging as absl_logging

# Prevent Streamlit file watcher from breaking on torch internals
sys.modules['torch.classes'].__path__ = []

# Fix event loop and multiprocessing issues
nest_asyncio.apply()
if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn', force=True)

# Configure logging and warnings
logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings('ignore')
absl_logging.set_verbosity(absl_logging.WARNING)

# Configure page
st.set_page_config(
    page_title="Dynamic Pricing RL App", 
    layout="wide",
    initial_sidebar_state="expanded"
)

def create_visualizations(results_container, summary_data, report):
    if not summary_data:
        results_container.warning("No evaluation results available.")
        return
    
    try:
        # Convert summary data to proper format if needed
        if isinstance(summary_data, dict):
            # Transform dict to list of tuples format
            formatted_data = []
            for strategy, metrics in summary_data.items():
                if isinstance(metrics, dict):
                    for metric, value in metrics.items():
                        formatted_data.append({
                            'Strategy': strategy,
                            'Metric': metric,
                            'Value': value
                        })
                else:
                    formatted_data.append({
                        'Strategy': strategy,
                        'Metric': 'Performance',
                        'Value': metrics
                    })
            
            results_df = pd.DataFrame(formatted_data)
        else:
            # Assume it's already in DataFrame format or compatible structure
            results_df = pd.DataFrame(summary_data)
        
        # Create tabs for visualizations
        metric_tab, profit_tab = st.tabs(["Performance Metrics", "Profit Analysis"])
        
        with metric_tab:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fig = px.bar(
                    results_df,
                    x='Strategy',
                    y='Value',
                    color='Metric',
                    title="Strategy Performance Comparison",
                    template="plotly_white"
                )
                fig.update_layout(
                    height=400,
                    showlegend=True,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Handle profit analysis
        profit_ranges = report.get('profit_ranges', {})
        if profit_ranges:
            with profit_tab:
                profit_data = pd.DataFrame([
                    {
                        'Strategy': strategy,
                        'Type': ptype,
                        'Value': value
                    }
                    for strategy, (min_val, avg_val, max_val) in profit_ranges.items()
                    for ptype, value in [
                        ('Min', min_val),
                        ('Avg', avg_val),
                        ('Max', max_val)
                    ]
                ])
                
                fig = px.box(
                    profit_data,
                    x='Strategy',
                    y='Value',
                    title="Profit Distribution by Strategy",
                    template="plotly_white"
                )
                fig.update_layout(
                    height=400,
                    yaxis_title="Profit ($)",
                    showlegend=True,
                    hovermode='x unified'
                )
                fig.update_traces(
                    hovertemplate="$%{y:,.2f}<extra></extra>"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary statistics table
                stats_df = pd.pivot_table(
                    profit_data,
                    values='Value',
                    index='Strategy',
                    columns='Type'
                ).round(2)
                stats_df = stats_df.apply(lambda x: x.map('${:,.2f}'.format))
                st.dataframe(stats_df, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error creating visualizations: {str(e)}")
        st.exception(e)  # This will show the full traceback in development

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
        step_status = {step: "⭕" for step in ["Data", "Environment", "Training", "Evaluation", "Deployment"]}
        
        st.write("## Process Steps")
        col_container = st.container()
        with col_container:
            # Create columns with equal width and proper spacing
            cols = st.columns([1, 1, 1, 1, 1], gap="medium")
            
            # Data collection
            with cols[0]:
                st.markdown("### Step 1")
                if st.button("📥 Collect Data", key="collect_data", use_container_width=True):
                    step_status["Data"] = "⏳"
                    progress_placeholder.write(f"Progress: {' '.join(step_status.values())}")
                    with st.spinner("Collecting data..."):
                        st.session_state.system.collect_data()
                    step_status["Data"] = "✅"
                    progress_placeholder.write(f"Progress: {' '.join(step_status.values())}")
            
            # Environment preparation
            with cols[1]:
                st.markdown("### Step 2")
                if st.button("🔧 Prepare", key="prepare_env", use_container_width=True):
                    step_status["Environment"] = "⏳"
                    progress_placeholder.write(f"Progress: {' '.join(step_status.values())}")
                    with st.spinner("Preparing..."):
                        env = st.session_state.system.prepare_environment()
                        st.session_state.system.setup_agents()
                    step_status["Environment"] = "✅"
                    progress_placeholder.write(f"Progress: {' '.join(step_status.values())}")
            
            # Training section
            with cols[2]:
                st.markdown("### Step 3")
                training_container = st.container()
                
                if st.button("🎯 Train", key="train_agent", use_container_width=True):
                    step_status["Training"] = "⏳"
                    progress_placeholder.write(f"Progress: {' '.join(step_status.values())}")
                    
                    progress_bar = training_container.progress(0)
                    status_text = training_container.empty()
                    metrics_container = training_container.container()
                    
                    # Training loop with metrics
                    for episode in range(training_episodes):
                        st.session_state.current_episode = episode
                        progress = (episode + 1) / training_episodes
                        progress_bar.progress(progress)
                        status_text.text(f"Training Episode: {episode + 1}/{training_episodes}")
                        
                        # Train for one episode
                        training_metrics = st.session_state.system.train_agent(num_episodes=1)
                        
                        # Display metrics if available
                        if training_metrics and episode % 10 == 0:
                            col1, col2 = metrics_container.columns(2)
                            col1.metric("Average Reward", f"{training_metrics.get('avg_reward', 0):.2f}")
                            col2.metric("Loss", f"{training_metrics.get('loss', 0):.4f}")
                    
                    step_status["Training"] = "✅"
                    progress_placeholder.write(f"Progress: {' '.join(step_status.values())}")
            
            # Evaluation
            with cols[3]:
                st.markdown("### Step 4")
                if st.button("📊 Evaluate", key="evaluate", use_container_width=True):
                    step_status["Evaluation"] = "⏳"
                    progress_placeholder.write(f"Progress: {' '.join(step_status.values())}")
                    
                    with st.spinner("Evaluating strategies..."):
                        try:
                            report = st.session_state.system.evaluate_strategies()
                            results_container = st.container()
                            
                            with results_container:
                                st.subheader("Evaluation Results")
                                summary_data = report.get('summary', {})
                                create_visualizations(results_container, summary_data, report)
                                
                                # Display best strategy with metrics
                                best_agent = report.get('best_agent')
                                if best_agent and best_agent in report.get('profit_ranges', {}):
                                    st.success(f"🏆 Best Strategy: {best_agent}")
                                    avg_profit = report['profit_ranges'][best_agent][1]
                                    st.metric(
                                        "Expected Average Profit",
                                        f"${avg_profit:,.2f}",
                                        delta=f"${avg_profit - report['profit_ranges'].get('Static', [0,0,0])[1]:,.2f} vs Static"
                                    )
                        
                        except Exception as e:
                            st.error(f"Evaluation failed: {str(e)}")
                    
                    step_status["Evaluation"] = "✅"
                    progress_placeholder.write(f"Progress: {' '.join(step_status.values())}")
            
            # Deployment
            with cols[4]:
                st.markdown("### Step 5")
                if st.button("🚀 Deploy", key="deploy", use_container_width=True):
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
                st.image("price_comparison.png", caption="Price Comparison Across Strategies")
            except FileNotFoundError:
                st.info("Price analysis not available yet. Complete the evaluation step first.")
        
        with tab2:
            try:
                st.image("profit_comparison.png", caption="Profit Comparison")
            except FileNotFoundError:
                st.info("Profit analysis not available yet. Complete the evaluation step first.")
        
        with tab3:
            for agent in ["PPO", "Static", "Rule-Based"]:
                try:
                    st.image(f"demand_vs_price_{agent}.png", caption=f"Demand vs Price – {agent}")
                except FileNotFoundError:
                    st.info(f"{agent} performance visualization not available yet.")

if __name__ == "__main__":
    main()
