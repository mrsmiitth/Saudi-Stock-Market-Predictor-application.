import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import json
import base64
from datetime import datetime, timedelta
import time
import plotly.graph_objects as go
from saudi_stock_predictor import StockPredictor, StockModelTrainer

# Page configuration
st.set_page_config(
    page_title="Saudi Stock Market Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        margin-bottom: 1rem;
    }
    .card {
        border-radius: 5px;
        background-color: #f9f9f9;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
    }
    .positive {
        color: #28a745;
    }
    .negative {
        color: #dc3545;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 1rem;
        color: #6c757d;
    }
</style>
""", unsafe_allow_html=True)

# Function to load market info
@st.cache_resource
def load_market_data():
    try:
        predictor = StockPredictor()
        return predictor.market_info
    except Exception as e:
        st.error(f"Error loading market data: {str(e)}")
        return pd.DataFrame(columns=["Symbol", "Name", "Sector"])

# Function to load predictions
@st.cache_data
def load_predictions():
    try:
        predictions_file = "predictions/market_predictions_summary.csv"
        if os.path.exists(predictions_file):
            return pd.read_csv(predictions_file)
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading predictions: {str(e)}")
        return pd.DataFrame()

# Function to load market report
@st.cache_data
def load_market_report():
    try:
        report_file = "reports/market_report.md"
        if os.path.exists(report_file):
            with open(report_file, "r", encoding="utf-8") as f:
                return f.read()
        return "No market report found. Please generate a report first."
    except Exception as e:
        return f"Error loading market report: {str(e)}"

# Function to check if model exists
def model_exists(symbol):
    model_file = f"models/{symbol.replace('.', '_')}_model.pth"
    return os.path.exists(model_file)

# Function to check if data exists
def data_exists(symbol):
    data_file = f"data/{symbol.replace('.', '_')}.csv"
    return os.path.exists(data_file)

# Main header
st.markdown('<h1 class="main-header">Saudi Stock Market Predictor 📊</h1>', unsafe_allow_html=True)

# Load market data
market_info = load_market_data()
predictions = load_predictions()

# Sidebar
with st.sidebar:
    st.header("Navigation")
    
    # View mode selection
    view_mode = st.radio(
        "Select View:",
        ["Dashboard", "Stock Analysis", "Sector Analysis", "Market Report"]
    )
    
    # Stock selection (for Stock Analysis mode)
    if view_mode == "Stock Analysis":
        stock_options = [(row["Symbol"], row["Name"]) for _, row in market_info.iterrows()]
        selected_stock = st.selectbox(
            "Select Stock:",
            options=stock_options,
            format_func=lambda x: f"{x[1]} ({x[0]})"
        )
        
        analyze_stock = st.button("Analyze Stock")
    
    # Sector selection (for Sector Analysis mode)
    elif view_mode == "Sector Analysis":
        sectors = market_info["Sector"].unique().tolist()
        selected_sector = st.selectbox("Select Sector:", sectors)
        
        analyze_sector = st.button("Analyze Sector")
    
    # Advanced options
    with st.expander("Advanced Settings"):
        update_data = st.button("Update Market Data")
        if update_data:
            with st.spinner("Updating market data..."):
                try:
                    predictor = StockPredictor()
                    predictor.download_historical_data(force_update=True)
                    predictor.prepare_market_features()
                    st.success("Market data updated successfully")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error updating market data: {str(e)}")

# Main content based on view mode
if view_mode == "Dashboard":
    # Dashboard view
    st.markdown('<h2 class="sub-header">Market Dashboard</h2>', unsafe_allow_html=True)
    
    # Market statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="metric-label">Total Stocks</div>
            <div class="metric-value">{len(market_info)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        sectors_count = len(market_info["Sector"].unique())
        st.markdown(f"""
        <div class="card">
            <div class="metric-label">Sectors</div>
            <div class="metric-value">{sectors_count}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        models_count = len([f for f in os.listdir("models") if f.endswith("_model.pth")]) if os.path.exists("models") else 0
        st.markdown(f"""
        <div class="card">
            <div class="metric-label">Trained Models</div>
            <div class="metric-value">{models_count}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Prediction summary
    if not predictions.empty:
        st.markdown('<h3>Top Stocks by Predicted Growth</h3>', unsafe_allow_html=True)
        
        # Top stocks table
        top_stocks = predictions.sort_values("Growth_Pct", ascending=False).head(10)
        
        # Format the table display
        display_df = top_stocks[["Symbol", "Name", "Sector", "Current_Price", "Future_Price", "Growth_Pct"]].copy()
        display_df["Current_Price"] = display_df["Current_Price"].map("${:.2f}".format)
        display_df["Future_Price"] = display_df["Future_Price"].map("${:.2f}".format)
        display_df["Growth_Pct"] = display_df["Growth_Pct"].map("{:.2f}%".format)
        
        display_df.columns = ["Symbol", "Name", "Sector", "Current Price", "Predicted Price", "Expected Growth"]
        
        st.dataframe(display_df)
        
        # Visualization of top 5 stocks by growth
        st.markdown('<h3>Predicted Growth Comparison (Top 5 Stocks)</h3>', unsafe_allow_html=True)
        
        top5 = top_stocks.head(5)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=top5["Name"],
            y=top5["Growth_Pct"],
            marker_color=['rgba(55, 83, 109, 0.7)', 'rgba(26, 118, 255, 0.7)', 
                         'rgba(55, 128, 191, 0.7)', 'rgba(0, 175, 181, 0.7)', 
                         'rgba(77, 175, 124, 0.7)'],
            text=[f"{x:.1f}%" for x in top5["Growth_Pct"]],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Top 5 Stocks by Expected Growth",
            xaxis_title="Company",
            yaxis_title="Expected Growth (%)",
            yaxis=dict(ticksuffix="%"),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Sector analysis
        st.markdown('<h3>Sector Analysis</h3>', unsafe_allow_html=True)
        
        sector_analysis = predictions.groupby("Sector").agg({
            "Growth_Pct": ["mean", "min", "max", "count"]
        }).reset_index()
        
        sector_analysis.columns = ["Sector", "Average Growth", "Minimum Growth", "Maximum Growth", "Stocks Count"]
        sector_analysis = sector_analysis.sort_values("Average Growth", ascending=False)
        
        # Format percentages
        sector_analysis["Average Growth"] = sector_analysis["Average Growth"].map("{:.2f}%".format)
        sector_analysis["Minimum Growth"] = sector_analysis["Minimum Growth"].map("{:.2f}%".format)
        sector_analysis["Maximum Growth"] = sector_analysis["Maximum Growth"].map("{:.2f}%".format)
        
        st.dataframe(sector_analysis)
    else:
        st.info("No predictions available. Run the market analysis to generate predictions.")
        
        if st.button("Run Market Analysis"):
            with st.spinner("Analyzing market... This may take some time."):
                try:
                    predictor = StockPredictor()
                    results = predictor.train_all_models(epochs=20)  # Reduced epochs for speed
                    predictions = predictor.predict_all_stocks()
                    predictor.generate_market_report()
                    st.success("Market analysis completed successfully!")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error during market analysis: {str(e)}")

elif view_mode == "Stock Analysis":
    if 'selected_stock' in locals():
        symbol, name = selected_stock
        
        st.markdown(f'<h2 class="sub-header">Stock Analysis: {name} ({symbol})</h2>', unsafe_allow_html=True)
        
        # Check data and model availability
        has_data = data_exists(symbol)
        has_model = model_exists(symbol)
        
        # Display status
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="card">
                <div class="metric-label">Data Status</div>
                <div class="metric-value">{'✅ Available' if has_data else '❌ Not Available'}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="card">
                <div class="metric-label">Model Status</div>
                <div class="metric-value">{'✅ Trained' if has_model else '❌ Not Trained'}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Analysis actions
        if analyze_stock or (has_model and 'analyze_stock' not in locals()):
            with st.spinner(f"Analyzing {name}..."):
                try:
                    # Download data if needed
                    if not has_data:
                        st.text("Downloading stock data...")
                        predictor = StockPredictor()
                        predictor.download_historical_data(symbols=[symbol])
                    
                    # Train model if needed
                    if not has_model:
                        st.text("Training prediction model...")
                        trainer = StockModelTrainer(symbol)
                        results = trainer.train_model(epochs=30)
                        
                        # Display training results
                        st.subheader("Model Training Results")
                        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                        with metrics_col1:
                            st.metric("MAE", f"{results['mae']:.2f}")
                        with metrics_col2:
                            st.metric("RMSE", f"{results['rmse']:.2f}")
                        with metrics_col3:
                            st.metric("MAPE", f"{results['mape']:.2f}%")
                    
                    # Make predictions
                    st.text("Generating price predictions...")
                    predictor = StockPredictor(symbol)
                    future = predictor.predict_future(days=180)
                    
                    # Display prediction results
                    st.subheader("Price Predictions")
                    
                    # Extract key metrics
                    current_price = future['Current_Price']
                    future_price = future['Prices'][-1]
                    growth = ((future_price / current_price) - 1) * 100
                    
                    # Display metrics
                    pred_col1, pred_col2, pred_col3 = st.columns(3)
                    with pred_col1:
                        st.metric("Current Price", f"${current_price:.2f}")
                    with pred_col2:
                        st.metric("Predicted Price (6 months)", f"${future_price:.2f}")
                    with pred_col3:
                        st.metric("Expected Growth", f"{growth:.2f}%", 
                                delta=f"{growth:.2f}%" if growth >= 0 else f"{growth:.2f}%")
                    
                    # Display chart
                    st.subheader("Price Forecast Chart")
                    chart_path = f"charts/{symbol.replace('.', '_')}_future.png"
                    if os.path.exists(chart_path):
                        st.image(chart_path)
                    else:
                        st.warning("Chart not available")
                    
                    # Detailed data
                    with st.expander("View Detailed Prediction Data"):
                        df = future['DataFrame']
                        st.dataframe(df)
                        
                        # Download button
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download as CSV",
                            data=csv,
                            file_name=f"{symbol}_predictions.csv",
                            mime="text/csv"
                        )
                
                except Exception as e:
                    st.error(f"Error during stock analysis: {str(e)}")
        else:
            # If analysis not started
            st.info("Click 'Analyze Stock' to start the analysis")

elif view_mode == "Sector Analysis":
    if 'selected_sector' in locals():
        st.markdown(f'<h2 class="sub-header">Sector Analysis: {selected_sector}</h2>', unsafe_allow_html=True)
        
        # Get sector stocks
        sector_stocks = market_info[market_info["Sector"] == selected_sector]
        
        st.subheader(f"Stocks in {selected_sector} Sector")
        st.dataframe(sector_stocks[["Symbol", "Name"]])
        
        # Check if sector analysis exists
        sector_results_path = f"reports/{selected_sector}_training_results.csv"
        sector_results_exist = os.path.exists(sector_results_path)
        
        if sector_results_exist:
            # Display sector analysis results
            st.subheader("Sector Analysis Results")
            
            # Load training results
            sector_results = pd.read_csv(sector_results_path)
            
            # Display average accuracy metrics
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            with metrics_col1:
                st.metric("Average MAE", f"{sector_results['MAE'].mean():.2f}")
            with metrics_col2:
                st.metric("Average RMSE", f"{sector_results['RMSE'].mean():.2f}")
            with metrics_col3:
                st.metric("Average MAPE", f"{sector_results['MAPE'].mean():.2f}%")
            
            # Display best performing models
            st.subheader("Best Performing Models by Accuracy")
            best_models = sector_results.sort_values("MAPE").head(5)
            
            # Add stock names
            best_models = best_models.merge(market_info[["Symbol", "Name"]], on="Symbol")
            
            st.dataframe(best_models[["Symbol", "Name", "MAE", "RMSE", "MAPE"]])
            
            # Display sector predictions if available
            if not predictions.empty:
                sector_predictions = predictions[predictions["Sector"] == selected_sector]
                
                if not sector_predictions.empty:
                    st.subheader("Sector Stock Predictions")
                    
                    # Get top stocks by expected growth
                    top_sector_stocks = sector_predictions.sort_values("Growth_Pct", ascending=False)
                    
                    # Format display dataframe
                    display_df = top_sector_stocks[["Symbol", "Name", "Current_Price", "Future_Price", "Growth_Pct"]].copy()
                    display_df["Current_Price"] = display_df["Current_Price"].map("${:.2f}".format)
                    display_df["Future_Price"] = display_df["Future_Price"].map("${:.2f}".format)
                    display_df["Growth_Pct"] = display_df["Growth_Pct"].map("{:.2f}%".format)
                    
                    display_df.columns = ["Symbol", "Name", "Current Price", "Predicted Price", "Expected Growth"]
                    
                    st.dataframe(display_df)
                    
                    # Visualization of sector stocks comparison
                    st.subheader("Expected Growth Comparison")
                    
                    # Limit to top 10 stocks for better visualization
                    stocks_to_plot = top_sector_stocks.head(10) if len(top_sector_stocks) > 10 else top_sector_stocks
                    
                    # Create bar chart using Plotly
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=stocks_to_plot["Name"],
                        y=stocks_to_plot["Growth_Pct"],
                        marker_color=['green' if x >= 0 else 'red' for x in stocks_to_plot["Growth_Pct"]],
                        text=[f"{x:.1f}%" for x in stocks_to_plot["Growth_Pct"]],
                        textposition='auto'
                    ))
                    
                    fig.update_layout(
                        title=f"Expected Growth for {selected_sector} Sector Stocks",
                        xaxis_title="Company",
                        yaxis_title="Expected Growth (%)",
                        yaxis=dict(ticksuffix="%"),
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # Button to start sector analysis
        if analyze_sector or (sector_results_exist and 'analyze_sector' not in locals()):
            if not sector_results_exist:
                with st.spinner(f"Analyzing {selected_sector} sector... This may take some time."):
                    try:
                        predictor = StockPredictor()
                        predictor.download_historical_data(symbols=sector_stocks["Symbol"].tolist())
                        predictor.prepare_market_features()
                        results = predictor.train_models_for_sector(selected_sector, epochs=30)
                        st.success(f"Analysis for {selected_sector} sector completed successfully!")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error during sector analysis: {str(e)}")
        else:
            # If analysis not started and results don't exist
            if not sector_results_exist:
                st.info("Click 'Analyze Sector' to start the analysis")

elif view_mode == "Market Report":
    st.markdown('<h2 class="sub-header">Market Report</h2>', unsafe_allow_html=True)
    
    # Load report
    report_content = load_market_report()
    
    # Display report
    st.markdown(report_content)
    
    # If report not available
    if "No market report found" in report_content:
        if st.button("Generate Market Report"):
            with st.spinner("Generating market report..."):
                try:
                    predictor = StockPredictor()
                    report = predictor.generate_market_report()
                    st.success("Market report generated successfully!")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error generating market report: {str(e)}")

# Footer
st.markdown("---")
st.markdown("📊 Saudi Stock Market Predictor | Built with Python, PyTorch, and Streamlit")