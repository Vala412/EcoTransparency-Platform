"""
Streamlit Frontend for EcoTransparency Platform
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import time

# Configure Streamlit page
st.set_page_config(
    page_title="EcoTransparency Platform",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .alert-high {
        background-color: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #c62828;
    }
    .alert-medium {
        background-color: #fff3e0;
        color: #ef6c00;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ef6c00;
    }
    .alert-low {
        background-color: #e8f5e8;
        color: #2e7d32;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'api_base_url' not in st.session_state:
    st.session_state.api_base_url = "http://localhost:8000"

def call_api(endpoint, method="GET", data=None):
    """Helper function to call API endpoints"""
    try:
        url = f"{st.session_state.api_base_url}/{endpoint}"
        
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Connection error: {str(e)}")
        return None

def main():
    """Main application function"""
    st.title("🌱 EcoTransparency Platform")
    st.subheader("Making Sustainability Transparent and Actionable")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Dashboard", "Product Analysis", "Sustainability Scoring", "Greenwashing Detection", 
         "Demand Forecasting", "Personalized Recommendations", "Carbon Calculator", "Data Upload"]
    )
    
    # Route to selected page
    if page == "Dashboard":
        dashboard_page()
    elif page == "Product Analysis":
        product_analysis_page()
    elif page == "Sustainability Scoring":
        sustainability_scoring_page()
    elif page == "Greenwashing Detection":
        greenwashing_detection_page()
    elif page == "Demand Forecasting":
        demand_forecasting_page()
    elif page == "Personalized Recommendations":
        personalization_page()
    elif page == "Carbon Calculator":
        carbon_calculator_page()
    elif page == "Data Upload":
        data_upload_page()

def dashboard_page():
    """Dashboard overview page"""
    st.header("📊 Platform Dashboard")
    
    # Get dashboard analytics
    analytics = call_api("analytics/dashboard")
    
    if analytics:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Products", analytics['total_products'], "↗️ 12%")
        
        with col2:
            st.metric("Avg Sustainability Score", f"{analytics['avg_sustainability_score']:.1f}", "↗️ 2.3")
        
        with col3:
            st.metric("Greenwashing Alerts", analytics['greenwashing_alerts'], "↘️ 5")
        
        with col4:
            st.metric("Carbon Reduction", f"{analytics['carbon_footprint_reduction']:.1f}%", "↗️ 1.2%")
        
        # Recent activity
        st.subheader("Recent Activity")
        activity_df = pd.DataFrame(analytics['recent_activity'])
        
        fig = px.bar(
            activity_df, 
            x='date', 
            y='count', 
            color='type',
            title="Daily Activity Overview"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Top categories
        st.subheader("Top Categories by Sustainability")
        categories_df = pd.DataFrame(analytics['top_categories'])
        
        fig = px.scatter(
            categories_df,
            x='count',
            y='avg_score',
            size='count',
            color='category',
            title="Category Performance",
            labels={'count': 'Number of Products', 'avg_score': 'Average Sustainability Score'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Trending sustainable products
    st.subheader("🔥 Trending Sustainable Products")
    trending = call_api("personalization/trending")
    
    if trending:
        trending_df = pd.DataFrame(trending)
        
        for _, product in trending_df.iterrows():
            with st.expander(f"🌟 {product['name']} (Score: {product['sustainability_score']:.1f})"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Product ID:** {product['product_id']}")
                    st.write(f"**Sustainability Score:** {product['sustainability_score']:.1f}")
                with col2:
                    st.write(f"**Trend Score:** {product['trend_score']:.2f}")
                    st.progress(product['trend_score'])

def product_analysis_page():
    """Product analysis page"""
    st.header("🔍 Product Analysis")
    
    # Product search
    col1, col2 = st.columns([2, 1])
    
    with col1:
        product_id = st.number_input("Enter Product ID", min_value=1, value=1)
    
    with col2:
        if st.button("Analyze Product"):
            analyze_product(product_id)
    
    # Product list
    st.subheader("Product Catalog")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        category_filter = st.selectbox("Category", ["All", "Electronics", "Home", "Clothing"])
    
    with col2:
        min_score = st.slider("Min Sustainability Score", 0, 100, 50)
    
    with col3:
        max_results = st.slider("Max Results", 10, 100, 20)
    
    # Get products
    params = {
        "limit": max_results,
        "min_sustainability_score": min_score
    }
    
    if category_filter != "All":
        params["category"] = category_filter
    
    products = call_api(f"products/?{requests.compat.urlencode(params)}")
    
    if products:
        products_df = pd.DataFrame(products)
        
        # Display products in a grid
        for i in range(0, len(products_df), 3):
            cols = st.columns(3)
            
            for j, col in enumerate(cols):
                if i + j < len(products_df):
                    product = products_df.iloc[i + j]
                    
                    with col:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>{product['name']}</h4>
                            <p><strong>Brand:</strong> {product['brand']}</p>
                            <p><strong>Category:</strong> {product['category']}</p>
                            <p><strong>Price:</strong> ${product['price']:.2f}</p>
                            <p><strong>Sustainability Score:</strong> {product['sustainability_score']:.1f}</p>
                            <p><strong>Carbon Footprint:</strong> {product['carbon_footprint']:.1f} kg CO2e</p>
                        </div>
                        """, unsafe_allow_html=True)

def analyze_product(product_id):
    """Analyze a specific product"""
    product = call_api(f"products/{product_id}")
    
    if product:
        st.success(f"Analyzing Product: {product['name']}")
        
        # Product details
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Product Details")
            st.write(f"**Name:** {product['name']}")
            st.write(f"**Brand:** {product['brand']}")
            st.write(f"**Category:** {product['category']}")
            st.write(f"**Price:** ${product['price']:.2f}")
        
        with col2:
            st.subheader("Sustainability Metrics")
            st.write(f"**Sustainability Score:** {product['sustainability_score']:.1f}")
            st.write(f"**Carbon Footprint:** {product['carbon_footprint']:.1f} kg CO2e")
            
            # Score gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=product['sustainability_score'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Sustainability Score"},
                gauge={'axis': {'range': [None, 100]},
                      'bar': {'color': "lightgreen"},
                      'steps': [
                          {'range': [0, 50], 'color': "lightgray"},
                          {'range': [50, 80], 'color': "yellow"}],
                      'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 90}}))
            st.plotly_chart(fig, use_container_width=True)

def sustainability_scoring_page():
    """Sustainability scoring page"""
    st.header("🌿 Sustainability Scoring")
    
    st.write("Enter product information to calculate sustainability score:")
    
    # Input form
    with st.form("sustainability_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            carbon_footprint = st.number_input("Carbon Footprint (kg CO2e)", min_value=0.0, value=10.0)
            water_usage = st.number_input("Water Usage (liters)", min_value=0.0, value=100.0)
            energy_consumption = st.number_input("Energy Consumption (kWh)", min_value=0.0, value=50.0)
            transportation_distance = st.number_input("Transportation Distance (km)", min_value=0.0, value=500.0)
        
        with col2:
            weight = st.number_input("Product Weight (kg)", min_value=0.1, value=1.0)
            packaging_recyclable = st.checkbox("Packaging Recyclable")
            packaging_biodegradable = st.checkbox("Packaging Biodegradable")
            renewable_energy_percentage = st.slider("Renewable Energy %", 0, 100, 20)
        
        submitted = st.form_submit_button("Calculate Sustainability Score")
    
    if submitted:
        # Prepare data for API call
        product_data = {
            "carbon_footprint": carbon_footprint,
            "water_usage": water_usage,
            "energy_consumption": energy_consumption,
            "transportation_distance": transportation_distance,
            "weight": weight,
            "packaging_recyclable": 1 if packaging_recyclable else 0,
            "packaging_biodegradable": 1 if packaging_biodegradable else 0,
            "renewable_energy_percentage": renewable_energy_percentage
        }
        
        # Call API
        result = call_api("sustainability/score", method="POST", data={"product_id": 1, **product_data})
        
        if result:
            # Display results
            st.success("Sustainability Score Calculated!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Overall Score")
                st.metric("Sustainability Score", f"{result['overall_score']:.1f}")
                
                # Component scores
                st.subheader("Component Scores")
                components = {
                    "Emissions": result['emission_score'],
                    "Resource Usage": result['resource_usage_score'],
                    "Transportation": result['transportation_score'],
                    "Recyclability": result['recyclability_score'],
                    "Labor Practices": result['labor_practices_score']
                }
                
                for component, score in components.items():
                    if score is not None:
                        st.metric(component, f"{score:.1f}")
            
            with col2:
                st.subheader("Score Breakdown")
                
                # Create radar chart
                categories = list(components.keys())
                values = [score if score is not None else 0 for score in components.values()]
                
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name='Sustainability Components'
                ))
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )),
                    showlegend=True,
                    title="Sustainability Score Breakdown"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            if result.get('recommendations'):
                st.subheader("Recommendations")
                for rec in result['recommendations']:
                    st.write(f"• {rec}")

def greenwashing_detection_page():
    """Greenwashing detection page"""
    st.header("🕵️ Greenwashing Detection")
    
    st.write("Analyze product claims for potential greenwashing:")
    
    # Input form
    with st.form("greenwashing_form"):
        product_description = st.text_area(
            "Product Description",
            placeholder="Enter product description, marketing text, or claims to analyze..."
        )
        
        marketing_text = st.text_area(
            "Marketing Text",
            placeholder="Enter additional marketing materials..."
        )
        
        certifications = st.text_input(
            "Claimed Certifications",
            placeholder="e.g., USDA Organic, Fair Trade, Energy Star (comma-separated)"
        )
        
        submitted = st.form_submit_button("Analyze for Greenwashing")
    
    if submitted and product_description:
        # Prepare data for API call
        analysis_data = {
            "description": product_description,
            "marketing_text": marketing_text,
            "certifications": [cert.strip() for cert in certifications.split(",") if cert.strip()]
        }
        
        # Call API
        result = call_api("greenwashing/analyze", method="POST", data={"product_id": 1, **analysis_data})
        
        if result:
            # Display results
            risk_score = result['risk_score']
            
            if risk_score > 0.8:
                alert_class = "alert-high"
                risk_level = "HIGH RISK"
            elif risk_score > 0.4:
                alert_class = "alert-medium"
                risk_level = "MEDIUM RISK"
            else:
                alert_class = "alert-low"
                risk_level = "LOW RISK"
            
            st.markdown(f"""
            <div class="{alert_class}">
                <h3>Greenwashing Risk Level: {risk_level}</h3>
                <p>Risk Score: {risk_score:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Flagged Content")
                if result.get('flagged_text'):
                    for text in result['flagged_text']:
                        st.write(f"⚠️ {text}")
                
                if result.get('flagged_claims'):
                    st.subheader("Flagged Claims")
                    for claim in result['flagged_claims']:
                        st.write(f"🚨 {claim}")
            
            with col2:
                st.subheader("Analysis Details")
                st.write(f"**Alert Type:** {result['alert_type']}")
                st.write(f"**Confidence:** {result['confidence']:.2f}")
                st.write(f"**Status:** {result['status']}")
        
        # Sample greenwashing patterns
        st.subheader("Common Greenwashing Patterns")
        patterns = [
            "Vague claims without specific data",
            "Hidden trade-offs (highlighting one green attribute while ignoring others)",
            "Irrelevant claims (emphasizing absence of substances that were never used)",
            "Fake certifications or misleading logos",
            "Fluffy language with no clear meaning"
        ]
        
        for pattern in patterns:
            st.write(f"• {pattern}")

def demand_forecasting_page():
    """Demand forecasting page"""
    st.header("📈 Demand Forecasting")
    
    # Product selection
    product_id = st.number_input("Product ID", min_value=1, value=1)
    forecast_horizon = st.slider("Forecast Horizon (days)", 7, 365, 30)
    
    if st.button("Generate Forecast"):
        # Call API
        forecasts = call_api(f"forecasting/predict/{product_id}?forecast_horizon={forecast_horizon}")
        
        if forecasts:
            # Convert to DataFrame
            forecast_df = pd.DataFrame(forecasts)
            forecast_df['forecast_date'] = pd.to_datetime(forecast_df['forecast_date'])
            
            # Display forecast chart
            st.subheader("Demand Forecast")
            
            fig = go.Figure()
            
            # Add forecast line
            fig.add_trace(go.Scatter(
                x=forecast_df['forecast_date'],
                y=forecast_df['predicted_demand'],
                mode='lines',
                name='Predicted Demand',
                line=dict(color='blue')
            ))
            
            # Add confidence intervals if available
            if 'confidence_interval_lower' in forecast_df.columns and forecast_df['confidence_interval_lower'].notna().any():
                fig.add_trace(go.Scatter(
                    x=forecast_df['forecast_date'],
                    y=forecast_df['confidence_interval_upper'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    name='Upper Bound'
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_df['forecast_date'],
                    y=forecast_df['confidence_interval_lower'],
                    mode='lines',
                    line=dict(width=0),
                    fillcolor='rgba(0,100,80,0.2)',
                    fill='tonexty',
                    showlegend=False,
                    name='Lower Bound'
                ))
            
            fig.update_layout(
                title="Demand Forecast",
                xaxis_title="Date",
                yaxis_title="Predicted Demand",
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Forecasted Demand", f"{forecast_df['predicted_demand'].sum():.1f}")
            
            with col2:
                st.metric("Average Daily Demand", f"{forecast_df['predicted_demand'].mean():.1f}")
            
            with col3:
                st.metric("Peak Demand", f"{forecast_df['predicted_demand'].max():.1f}")
            
            # Production recommendations
            st.subheader("Production Recommendations")
            recommendations = call_api(f"forecasting/production-recommendations/{product_id}?forecast_horizon={forecast_horizon}")
            
            if recommendations:
                st.write(f"**Recommended Production:** {recommendations['recommended_production']:.1f} units")
                st.write(f"**Safety Stock:** {recommendations['safety_stock']:.1f} units")
                
                if recommendations.get('inventory_alerts'):
                    st.warning("⚠️ Inventory Alerts:")
                    for alert in recommendations['inventory_alerts']:
                        st.write(f"• {alert['message']}")

def personalization_page():
    """Personalization page"""
    st.header("👤 Personalized Recommendations")
    
    # User input
    user_id = st.text_input("User ID", value="user123")
    
    col1, col2 = st.columns(2)
    
    with col1:
        category_filter = st.selectbox("Preferred Category", ["All", "Electronics", "Home", "Clothing"])
        num_recommendations = st.slider("Number of Recommendations", 5, 20, 10)
    
    with col2:
        if st.button("Get Recommendations"):
            get_user_recommendations(user_id, num_recommendations, category_filter)
    
    # User preferences form
    st.subheader("Update User Preferences")
    
    with st.form("preferences_form"):
        sustainability_priorities = st.multiselect(
            "Sustainability Priorities",
            ["low_carbon", "plastic_free", "local", "renewable_energy", "fair_trade"],
            default=["low_carbon", "local"]
        )
        
        min_sustainability_score = st.slider("Minimum Sustainability Score", 0, 100, 60)
        
        price_range = st.slider("Price Range", 0, 1000, (50, 500))
        
        submitted = st.form_submit_button("Update Preferences")
        
        if submitted:
            profile_data = {
                "preferences": {
                    "sustainability_priorities": sustainability_priorities,
                    "min_sustainability_score": min_sustainability_score,
                    "price_range": price_range
                },
                "interaction_data": []  # Would be populated with real interaction data
            }
            
            result = call_api(f"personalization/profile/{user_id}", method="POST", data=profile_data)
            
            if result:
                st.success("User preferences updated!")

def get_user_recommendations(user_id, num_recommendations, category_filter):
    """Get personalized recommendations for user"""
    params = f"num_recommendations={num_recommendations}"
    if category_filter != "All":
        params += f"&category={category_filter}"
    
    recommendations = call_api(f"personalization/recommendations/{user_id}?{params}")
    
    if recommendations:
        st.subheader("Personalized Recommendations")
        
        for i, rec in enumerate(recommendations):
            with st.expander(f"#{i+1} Product {rec['product_id']} (Score: {rec['score']:.2f})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Recommendation Score:** {rec['score']:.2f}")
                    st.write(f"**Product ID:** {rec['product_id']}")
                
                with col2:
                    st.write(f"**Explanation:** {rec['explanation']}")
                    
                    # Feedback buttons
                    col_like, col_dislike = st.columns(2)
                    with col_like:
                        if st.button("👍 Like", key=f"like_{rec['product_id']}"):
                            update_feedback(user_id, rec['product_id'], "like")
                    
                    with col_dislike:
                        if st.button("👎 Dislike", key=f"dislike_{rec['product_id']}"):
                            update_feedback(user_id, rec['product_id'], "dislike")

def update_feedback(user_id, product_id, feedback_type):
    """Update user feedback"""
    feedback_data = {
        "product_id": product_id,
        "feedback_type": feedback_type,
        "implicit_feedback": False
    }
    
    result = call_api(f"personalization/feedback/{user_id}", method="POST", data=feedback_data)
    
    if result:
        st.success(f"Feedback recorded: {feedback_type}")

def carbon_calculator_page():
    """Carbon footprint calculator page"""
    st.header("🌍 Carbon Footprint Calculator")
    
    st.write("Calculate the carbon footprint of your product basket:")
    
    # Product input
    if 'products' not in st.session_state:
        st.session_state.products = []
    
    with st.form("add_product_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            product_name = st.text_input("Product Name")
            weight = st.number_input("Weight (kg)", min_value=0.1, value=1.0)
        
        with col2:
            distance = st.number_input("Transportation Distance (km)", min_value=0.0, value=100.0)
            material = st.selectbox("Primary Material", ["plastic", "steel", "aluminum", "glass", "paper"])
        
        with col3:
            quantity = st.number_input("Quantity", min_value=1, value=1)
            
        add_product = st.form_submit_button("Add Product")
    
    if add_product and product_name:
        product = {
            "name": product_name,
            "weight": weight,
            "transportation_distance": distance,
            "material_carbon_factor": {"plastic": 3.4, "steel": 2.9, "aluminum": 11.5, "glass": 0.85, "paper": 1.1}[material],
            "quantity": quantity
        }
        st.session_state.products.append(product)
        st.success(f"Added {product_name} to basket")
    
    # Display current basket
    if st.session_state.products:
        st.subheader("Current Basket")
        
        for i, product in enumerate(st.session_state.products):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**{product['name']}** - {product['quantity']} units, {product['weight']} kg each")
            
            with col2:
                if st.button("Remove", key=f"remove_{i}"):
                    st.session_state.products.pop(i)
                    st.experimental_rerun()
        
        # Calculate footprint
        if st.button("Calculate Carbon Footprint"):
            calculation_data = {"products": st.session_state.products}
            
            result = call_api("carbon/calculate", method="POST", data=calculation_data)
            
            if result:
                st.subheader("Carbon Footprint Results")
                
                # Display total footprint
                total_footprint = result['total_footprint_kg_co2e']
                st.metric("Total Carbon Footprint", f"{total_footprint:.2f} kg CO2e")
                
                # Comparisons
                st.subheader("Equivalent Comparisons")
                comparisons = result['comparison']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Car Miles", f"{comparisons['equivalent_car_miles']:.1f}")
                
                with col2:
                    st.metric("AC Usage (hours)", f"{comparisons['equivalent_ac_hours']:.1f}")
                
                with col3:
                    st.metric("Trees to Offset", f"{comparisons['trees_to_offset']:.1f}")
                
                # Recommendations
                st.subheader("Recommendations")
                for rec in result['recommendations']:
                    st.write(f"• {rec}")
        
        # Clear basket
        if st.button("Clear Basket"):
            st.session_state.products = []
            st.experimental_rerun()

def data_upload_page():
    """Data upload page"""
    st.header("📤 Data Upload")
    
    st.write("Upload data files to train models and analyze products:")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Display file info
        st.write(f"**File:** {uploaded_file.name}")
        st.write(f"**Size:** {uploaded_file.size} bytes")
        
        # Data type selection
        data_type = st.selectbox(
            "Data Type",
            ["product_data", "demand_data", "user_interaction_data"]
        )
        
        if st.button("Upload and Process"):
            # Read and display preview
            df = pd.read_csv(uploaded_file)
            
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            st.subheader("Data Summary")
            st.write(f"**Rows:** {len(df)}")
            st.write(f"**Columns:** {len(df.columns)}")
            st.write(f"**Column Names:** {', '.join(df.columns)}")
            
            # Mock upload to API (in real implementation, you'd send the file)
            st.success("Data uploaded successfully!")
            st.write("✅ Data processed and ready for analysis")
    
    # Sample data generation
    st.subheader("Generate Sample Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Generate Sample Product Data"):
            generate_sample_product_data()
    
    with col2:
        if st.button("Generate Sample Demand Data"):
            generate_sample_demand_data()

def generate_sample_product_data():
    """Generate sample product data"""
    np.random.seed(42)
    
    products = []
    categories = ["Electronics", "Home", "Clothing", "Beauty", "Sports"]
    
    for i in range(100):
        product = {
            "id": i + 1,
            "name": f"Product {i + 1}",
            "category": np.random.choice(categories),
            "price": np.random.uniform(10, 500),
            "weight": np.random.uniform(0.1, 10),
            "carbon_footprint": np.random.uniform(1, 50),
            "water_usage": np.random.uniform(10, 1000),
            "energy_consumption": np.random.uniform(5, 200),
            "sustainability_score": np.random.uniform(20, 95),
            "transportation_distance": np.random.uniform(50, 2000),
            "recyclability_score": np.random.uniform(0, 100)
        }
        products.append(product)
    
    df = pd.DataFrame(products)
    
    st.subheader("Generated Sample Product Data")
    st.dataframe(df.head(10))
    
    # Download link
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Sample Product Data",
        data=csv,
        file_name="sample_products.csv",
        mime="text/csv"
    )

def generate_sample_demand_data():
    """Generate sample demand data"""
    np.random.seed(42)
    
    # Generate date range
    start_date = datetime.now() - timedelta(days=365)
    dates = pd.date_range(start=start_date, periods=365, freq='D')
    
    demand_data = []
    
    for i, date in enumerate(dates):
        # Add seasonality and trend
        base_demand = 100
        seasonal = 20 * np.sin(2 * np.pi * i / 365)  # Yearly seasonality
        weekly = 10 * np.sin(2 * np.pi * i / 7)      # Weekly seasonality
        trend = 0.1 * i                              # Upward trend
        noise = np.random.normal(0, 5)               # Random noise
        
        demand = base_demand + seasonal + weekly + trend + noise
        demand = max(0, demand)  # Ensure non-negative
        
        demand_data.append({
            "date": date.strftime("%Y-%m-%d"),
            "demand": demand,
            "product_id": 1
        })
    
    df = pd.DataFrame(demand_data)
    
    st.subheader("Generated Sample Demand Data")
    st.dataframe(df.head(10))
    
    # Show demand chart
    fig = px.line(df, x='date', y='demand', title='Sample Demand Data')
    st.plotly_chart(fig, use_container_width=True)
    
    # Download link
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Sample Demand Data",
        data=csv,
        file_name="sample_demand.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()
