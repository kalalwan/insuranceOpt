import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pulp import *

# Set page config
st.set_page_config(page_title="Premium Pricing Optimizer", layout="wide")

# Title
st.title("Case Study: New Business Pricing Optimizer")

# Top navigation bar
st.markdown("""
<style>
.stButton button {
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    home_button = st.button("Home")
with col2:
    model_button = st.button("Model")
with col3:
    results_button = st.button("Results")
with col4:
    insights_button = st.button("Insights")

# Determine which page to show
if home_button:
    page = "Home"
elif model_button:
    page = "Model"
elif results_button:
    page = "Results"
elif insights_button:
    page = "Insights"
else:
    page = "Home"  # Default page

st.markdown("---")  # Add a separator line

# Function to reset parameters
def reset_parameters():
    st.session_state.TP = 100000
    st.session_state.demand_elasticity = 1.0
    st.session_state.retention_sensitivity = 1.0
    st.session_state.competitiveness_sensitivity = 1.0
    st.session_state.base_operating_ratio = 0.85
    st.session_state.w1 = 0.4
    st.session_state.w2 = 0.3
    st.session_state.w3 = 0.2
    st.session_state.w4 = 0.1

# Sidebar for input parameters
st.sidebar.header("Input Parameters")

# Add reset button
st.sidebar.button("Reset Parameters", on_click=reset_parameters)

# Technical Premium
TP = st.sidebar.number_input("Technical Premium ($)", min_value=1000, max_value=1000000, value=st.session_state.get('TP', 100000), help="The base premium calculated to achieve a 92% combined ratio.")

# Elasticity and sensitivity parameters
demand_elasticity = st.sidebar.slider("Demand Elasticity", 0.5, 2.0, st.session_state.get('demand_elasticity', 1.0), help="How sensitive sales are to price changes.")
retention_sensitivity = st.sidebar.slider("Retention Sensitivity", 0.1, 2.0, st.session_state.get('retention_sensitivity', 1.0), help="How sensitive customer retention is to price increases.")
competitiveness_sensitivity = st.sidebar.slider("Competitiveness Sensitivity", 0.1, 2.0, st.session_state.get('competitiveness_sensitivity', 1.0), help="How sensitive market competitiveness is to price changes.")

# Base operating ratio
base_operating_ratio = st.sidebar.slider("Base Operating Ratio", 0.7, 0.9, st.session_state.get('base_operating_ratio', 0.85), help="The starting point for operational efficiency.")

# Objective weights
st.sidebar.subheader("Objective Weights")
w1 = st.sidebar.slider("Profit Weight", 0.0, 1.0, st.session_state.get('w1', 0.4), help="Importance of profit in the optimization.")
w2 = st.sidebar.slider("Retention Weight", 0.0, 1.0, st.session_state.get('w2', 0.3), help="Importance of customer retention in the optimization.")
w3 = st.sidebar.slider("Competitiveness Weight", 0.0, 1.0, st.session_state.get('w3', 0.2), help="Importance of market competitiveness in the optimization.")
w4 = st.sidebar.slider("Operating Ratio Weight", 0.0, 1.0, st.session_state.get('w4', 0.1), help="Importance of operating ratio in the optimization.")

# Optimization function
def optimize_pricing(TP, demand_elasticity, retention_sensitivity, competitiveness_sensitivity, 
                     base_operating_ratio, w1, w2, w3, w4):
    # Create the LP problem
    prob = LpProblem("Insurance_Pricing_Optimization", LpMaximize)

    # Decision variable
    x = LpVariable("Flex", lowBound=-0.15, upBound=0.20)

    # Piecewise linear approximation
    segments = 10
    breakpoints = np.linspace(-0.15, 0.20, segments + 1)
    lambdas = LpVariable.dicts("lambda", range(segments), lowBound=0, upBound=1)

    # Helper functions
    def ExpectedSales(x):
        return 1 - demand_elasticity * x

    def Profit(x):
        return TP * (1 + x) * ExpectedSales(x) * (1 - (base_operating_ratio / (1 + x)))

    def Retention(x):
        return 1 - retention_sensitivity * max(0, x)

    def Competitiveness(x):
        return 1 - competitiveness_sensitivity * abs(x)

    def OperatingRatio(x):
        return base_operating_ratio / (1 + x)

    # Objective function using piecewise linear approximation
    prob += w1 * lpSum([lambdas[i] * Profit(breakpoints[i]) for i in range(segments)]) + \
            w2 * lpSum([lambdas[i] * Retention(breakpoints[i]) for i in range(segments)]) + \
            w3 * lpSum([lambdas[i] * Competitiveness(breakpoints[i]) for i in range(segments)]) - \
            w4 * lpSum([lambdas[i] * OperatingRatio(breakpoints[i]) for i in range(segments)]), "Total_Score"

    # Constraints
    prob += lpSum(lambdas) == 1, "Sum_of_Lambdas"
    prob += x == lpSum([lambdas[i] * breakpoints[i] for i in range(segments)]), "Flex_Definition"
    prob += lpSum([lambdas[i] * OperatingRatio(breakpoints[i]) for i in range(segments)]) <= 0.92, "Max_Combined_Ratio"
    prob += lpSum([lambdas[i] * Competitiveness(breakpoints[i]) for i in range(segments)]) >= 0.80, "Min_Competitiveness"

    # Solve the problem
    prob.solve()

    # Extract results
    flex = value(x)
    final_premium = TP * (1 + flex)
    profit = sum([value(lambdas[i]) * Profit(breakpoints[i]) for i in range(segments)])
    retention = sum([value(lambdas[i]) * Retention(breakpoints[i]) for i in range(segments)])
    competitiveness = sum([value(lambdas[i]) * Competitiveness(breakpoints[i]) for i in range(segments)])
    operating_ratio = sum([value(lambdas[i]) * OperatingRatio(breakpoints[i]) for i in range(segments)])

    return {
        "Flex": flex,
        "Final Premium": final_premium,
        "Profit": profit,
        "Retention": retention,
        "Competitiveness": competitiveness,
        "Operating Ratio": operating_ratio,
        "Problem": prob
    }

# Run optimization
results = optimize_pricing(TP, demand_elasticity, retention_sensitivity, competitiveness_sensitivity, 
                           base_operating_ratio, w1, w2, w3, w4)

# Content based on navigation
if page == "Home":
    st.header("Welcome to the Insurance Premium Pricing Optimizer")
    st.write("""
    This tool helps optimize pricing strategies for new business accounts. 
    Use the navigation bar above to explore different sections and the sidebar to adjust parameters.
    """)

elif page == "Model":
    st.header("Linear Programming Model")
    
    with st.expander("View General Linear Program Formulation"):
        st.latex(r"""
        \begin{align*}
        \text{Maximize: } & w_1 \cdot \text{Profit}(x) + w_2 \cdot \text{Retention}(x) + w_3 \cdot \text{Competitiveness}(x) - w_4 \cdot \text{OperatingRatio}(x) \\[10pt]
        \text{Subject to:} & \\
        & -0.15 \leq x \leq 0.20 \quad \text{(Flex bounds)} \\
        & \text{OperatingRatio}(x) \leq 0.92 \quad \text{(Maximum combined ratio)} \\
        & \text{Competitiveness}(x) \geq 0.80 \quad \text{(Minimum competitiveness)} \\[10pt]
        \text{Where:} & \\
        & x \text{ is the flex (decision variable)} \\
        & \text{Profit}(x) = TP \cdot (1 + x) \cdot \text{ExpectedSales}(x) \cdot (1 - \text{OperatingRatio}(x)) \\
        & \text{Retention}(x) = 1 - \alpha \cdot \max(0, x) \\
        & \text{Competitiveness}(x) = 1 - \beta \cdot |x| \\
        & \text{OperatingRatio}(x) = \frac{\text{BaseOperatingRatio}}{1 + x} \\
        & \text{ExpectedSales}(x) = 1 - \gamma \cdot x
        \end{align*}
        """)

elif page == "Results":
    st.header("Optimization Results")
    
    # Summary dashboard
    st.subheader("Summary Dashboard")
    col1, col2, col3 = st.columns(3)
    col1.metric("Optimal Flex", f"{results['Flex']:.2%}")
    col2.metric("Final Premium", f"${results['Final Premium']:,.2f}")
    col3.metric("Profit", f"${results['Profit']:,.2f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Retention Rate", f"{results['Retention']:.2%}")
    col5.metric("Competitiveness", f"{results['Competitiveness']:.2%}")
    col6.metric("Operating Ratio", f"{results['Operating Ratio']:.2%}")

    # Visualizations
    st.subheader("Visualizations")

    # Create data for visualizations
    flex_range = np.linspace(-0.15, 0.20, 100)
    profit_values = [TP * (1 + x) * (1 - demand_elasticity * x) * (1 - (base_operating_ratio / (1 + x))) for x in flex_range]
    retention_values = [1 - retention_sensitivity * max(0, x) for x in flex_range]
    competitiveness_values = [1 - competitiveness_sensitivity * abs(x) for x in flex_range]
    operating_ratio_values = [base_operating_ratio / (1 + x) for x in flex_range]

    # Combined plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=flex_range, y=profit_values, mode='lines', name='Profit'))
    fig.add_trace(go.Scatter(x=flex_range, y=retention_values, mode='lines', name='Retention'))
    fig.add_trace(go.Scatter(x=flex_range, y=competitiveness_values, mode='lines', name='Competitiveness'))
    fig.add_trace(go.Scatter(x=flex_range, y=operating_ratio_values, mode='lines', name='Operating Ratio'))
    fig.add_vline(x=results['Flex'], line_dash="dash", line_color="red", annotation_text="Optimal Flex")
    fig.update_layout(title="Key Metrics vs Flex", xaxis_title="Flex", yaxis_title="Value")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Insights":
    st.header("Analysis and Insights")
    st.write(f"""
    Based on the optimization results:

    1. The optimal flex is {results['Flex']:.2%}, resulting in a final premium of ${results['Final Premium']:,.2f}.
    2. This pricing strategy is expected to yield a profit of ${results['Profit']:,.2f}.
    3. The estimated retention rate is {results['Retention']:.2%}, with a competitiveness score of {results['Competitiveness']:.2%}.
    4. The operating ratio under this pricing strategy is {results['Operating Ratio']:.2%}.

    Key insights:
    - The model balances profitability with retention and competitiveness.
    - The optimal flex suggests a {'increase' if results['Flex'] > 0 else 'decrease'} in premium from the technical premium.
    - The operating ratio is maintained below the 92% threshold, ensuring overall profitability.

    Adjusting the input parameters allows for scenario testing and sensitivity analysis. This can help in understanding how different market conditions or business priorities might affect the optimal pricing strategy.
    """)

    st.info("""
    Remember, this is a simplified model. In reality, insurance pricing involves many more variables and complex interactions. 
    Always consult with actuarial and financial experts when making real-world pricing decisions.
    """)

# Add a footer
st.markdown("---")
st.markdown("© NB Co-op Presentations Ltd. All rights reserved.")