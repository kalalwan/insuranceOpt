import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import linprog

st.set_page_config(layout="wide")
st.title("3D Insurance Portfolio Optimization")

# Define the three portfolios
portfolios = ['Auto', 'Home', 'Commercial']

# Sidebar for input parameters
st.sidebar.header("Portfolio Parameters")

# Function to create a color gradient
def get_color(i, n):
    return f"rgb({255-int(255*i/n)}, {int(255*i/n)}, 150)"

expected_returns = {}
risks = {}
min_allocations = {}

for i, p in enumerate(portfolios):
    color = get_color(i, len(portfolios))
    st.sidebar.markdown(f"<h3 style='color: {color};'>{p} Portfolio</h3>", unsafe_allow_html=True)
    expected_returns[p] = st.sidebar.slider(f"Expected Return - {p}", 0.0, 0.20, 0.10, 0.01, key=f"return_{p}")
    risks[p] = st.sidebar.slider(f"Risk - {p}", 0.0, 0.15, 0.05, 0.01, key=f"risk_{p}")
    min_allocations[p] = st.sidebar.slider(f"Minimum Allocation - {p}", 0.0, 0.5, 0.1, 0.01, key=f"min_alloc_{p}")
    st.sidebar.markdown("---")

max_risk = st.sidebar.slider("Maximum Portfolio Risk", 0.0, 0.15, 0.07, 0.01)

# Optimization function
def optimize_portfolio(returns, risks, min_allocs, max_risk):
    c = [-r for r in returns]  # Negative because we're maximizing
    A_ub = [risks, [-1, -1, -1]]
    b_ub = [max_risk, -1]  # -1 because sum of allocations should be >= 1
    A_eq = [[1, 1, 1]]
    b_eq = [1]
    bounds = [(min_alloc, 1) for min_alloc in min_allocs]
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    return res.x if res.success else None

# Generate points for the feasible region
def generate_feasible_points(risks, min_allocs, max_risk, n=5000):
    points = []
    for _ in range(n):
        x = np.random.rand(3)
        x = np.maximum(x, min_allocs)
        x /= x.sum()  # Normalize to sum to 1
        if np.dot(risks, x) <= max_risk and all(x[i] >= min_allocs[i] for i in range(3)):
            points.append(x)
    return np.array(points)

# Display objective function and constraints
def display_problem_formulation(returns, risks, min_allocs, max_risk):
    st.subheader("Problem Formulation")
    
    st.write("**Objective Function (to maximize):**")
    obj_func = " + ".join([f"{r:.2f}x_{i+1}" for i, r in enumerate(returns)])
    st.latex(f"\\text{{Maximize }} {obj_func}")
    
    st.write("**Constraints:**")
    risk_constraint = " + ".join([f"{r:.2f}x_{i+1}" for i, r in enumerate(risks)])
    st.latex(f"{risk_constraint} \leq {max_risk:.2f}")
    st.latex("x_1 + x_2 + x_3 = 1")
    for i, min_alloc in enumerate(min_allocs):
        st.latex(f"x_{i+1} \geq {min_alloc:.2f}")
    st.latex("x_1, x_2, x_3 \leq 1")
    
    st.write("Where:")
    for i, p in enumerate(portfolios):
        st.write(f"x_{i+1} = Allocation to {p} portfolio")

# Optimize and visualize
if st.button("Optimize and Visualize"):
    returns = list(expected_returns.values())
    risk_values = list(risks.values())
    min_alloc_values = list(min_allocations.values())
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Display problem formulation
        display_problem_formulation(returns, risk_values, min_alloc_values, max_risk)
    
    optimal_allocation = optimize_portfolio(returns, risk_values, min_alloc_values, max_risk)
    
    if optimal_allocation is not None:
        feasible_points = generate_feasible_points(risk_values, min_alloc_values, max_risk)
        
        with col2:
            # Create the 3D scatter plot
            fig = go.Figure()
            
            # Feasible region as semi-transparent points
            fig.add_trace(go.Scatter3d(
                x=feasible_points[:, 0],
                y=feasible_points[:, 1],
                z=feasible_points[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color='blue',
                    opacity=0.1
                ),
                name='Feasible Region'
            ))
            
            # Optimal point
            fig.add_trace(go.Scatter3d(
                x=[optimal_allocation[0]],
                y=[optimal_allocation[1]],
                z=[optimal_allocation[2]],
                mode='markers',
                marker=dict(size=8, color='red'),
                name='Optimal Allocation'
            ))
            
            # Customize the layout
            fig.update_layout(
                scene = dict(
                    xaxis_title='Auto',
                    yaxis_title='Home',
                    zaxis_title='Commercial',
                    aspectmode='cube'
                ),
                width=600,
                height=600,
                title_text='Portfolio Allocation Visualization'
            )
            
            # Display the plot
            st.plotly_chart(fig)
        
        with col1:
            # Display optimal allocation and expected return
            st.subheader("Optimal Allocation:")
            for p, alloc in zip(portfolios, optimal_allocation):
                st.write(f"{p}: {alloc:.2%}")
            
            expected_return = sum(r * a for r, a in zip(returns, optimal_allocation))
            st.write(f"Expected Return: {expected_return:.2%}")
            st.write(f"Portfolio Risk: {sum(r * a for r, a in zip(risk_values, optimal_allocation)):.2%}")
    else:
        st.error("Optimization failed. Please adjust the parameters and try again.")