#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Kamiti Michael <kamitimichael404@gmail.com>
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.optimize import approx_fprime

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Maize–FAW Mathematical Model",
    page_icon="🌽",
    layout="wide"
)

st.title("Maize Yield Response to Climate Warming and Fall Armyworm")
st.subheader("Trans Nzoia County, Kenya")
st.caption(
    "Deterministic nonlinear ODE model | "
    "Temperature treated as a constant exogenous parameter"
)

# ============================================================
# 2. MODEL PARAMETERS FROM THE FINAL PROJECT
# ============================================================

K = 489056 / 105190 # Empirical upper production benchmark
B0_default = 0.10 * K # Initial maize state
P0_default = 50.0 # Initial FAW model population
t_end = 120 # Simulation duration in days

# Maize temperature-response parameters
Tmin = 13.5
ToptB = 30.5
Tmax = 38.0
rmax = 0.06

gamma = np.log(2) / np.log(
    (Tmax - Tmin) / (ToptB - Tmin)
)

# FAW data used in the project
T_data = np.array([20.0, 25.0, 30.0])
rm_data = np.array([0.068, 0.137, 0.224])
L_data = np.array([57.20, 39.87, 27.18])

mu_data = 1.0 / L_data

# Model conversion and calibration parameters
c_beta = 0.20
alpha25 = 7.79060e-4

# ============================================================
# 3. TEMPERATURE-DEPENDENT FUNCTIONS
# ============================================================

def maize_response(T):
    """Maize response to temperature."""
    if T <= Tmin or T >= Tmax:
        return 0.0

    x = (T - Tmin) / (ToptB - Tmin)
    return 2 * x**gamma - x**(2 * gamma)


def r_maize(T):
    """Temperature-dependent maize growth rate."""
    return rmax * maize_response(T)


def rm_faw(T):
    """Linearly interpolated FAW intrinsic growth rate."""
    return float(np.interp(T, T_data, rm_data))


def mu_faw(T):
    """Linearly interpolated FAW mortality/removal rate."""
    return float(np.interp(T, T_data, mu_data))


def beta_faw(T):
    """Model FAW interaction coefficient."""
    return c_beta * rm_faw(T) / K


def alpha_faw(T):
    """Calibrated maize-loss coefficient scaled by FAW response."""
    return alpha25 * rm_faw(T) / rm_data[1]


def reproduction_number(T):
    """Basic reproduction number at the maize carrying capacity."""
    return beta_faw(T) * K / mu_faw(T)


# ============================================================
# 4. THE TWO-STATE ODE MODEL
# ============================================================

def model(t, y, T):
    B, P = y

    # Protect against tiny negative numerical solver values.
    # This does not change the mathematical model itself.
    B_eval = max(B, 0.0)
    P_eval = max(P, 0.0)

    dB = (
        r_maize(T) * B_eval * (1 - B_eval / K)
        - alpha_faw(T) * P_eval * B_eval
    )

    dP = (
        beta_faw(T) * P_eval * B_eval
        - mu_faw(T) * P_eval
    )

    return [dB, dP]


def simulate(T, B0, P0):
    """Solve the coupled model for a fixed temperature."""
    times = np.linspace(0, t_end, 301)

    solution = solve_ivp(
        model,
        (0, t_end),
        [B0, P0],
        args=(T,),
        t_eval=times,
        rtol=1e-9,
        atol=1e-11
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    return times, solution.y[0], solution.y[1]


def simulate_without_faw(T, B0):
    """Reference maize trajectory with no FAW pressure."""
    times = np.linspace(0, t_end, 301)

    def maize_only(t, y):
        B = max(y[0], 0.0)
        return [r_maize(T) * B * (1 - B / K)]

    solution = solve_ivp(
        maize_only,
        (0, t_end),
        [B0],
        t_eval=times,
        rtol=1e-9,
        atol=1e-11
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    return times, solution.y[0]


# ============================================================
# 5. EQUILIBRIA AND LOCAL STABILITY
# ============================================================

def jacobian(B, P, T):
    """Jacobian matrix of the two-state system."""
    r = r_maize(T)
    a = alpha_faw(T)
    b = beta_faw(T)
    m = mu_faw(T)

    return np.array([
        [r * (1 - 2 * B / K) - a * P, -a * B],
        [b * P, b * B - m]
    ])


def equilibrium_analysis(T):
    r = r_maize(T)
    a = alpha_faw(T)
    b = beta_faw(T)
    m = mu_faw(T)
    R0 = reproduction_number(T)

    # Origin equilibrium
    E0 = (0.0, 0.0)
    eig0 = np.linalg.eigvals(jacobian(*E0, T))

    # Pest-free equilibrium
    E1 = (K, 0.0)
    eig1 = np.linalg.eigvals(jacobian(*E1, T))

    if R0 < 1:
        pest_free_status = (
            "Locally asymptotically stable "
            "for the stated model, if r(T) > 0"
        )
    elif np.isclose(R0, 1.0):
        pest_free_status = "Threshold case; linearisation is inconclusive"
    else:
        pest_free_status = "Unstable to FAW invasion"

    # Coexistence equilibrium, when it is biologically feasible
    coexistence = None
    coexistence_eigs = None
    coexistence_status = "No positive coexistence equilibrium"

    if b > 0 and a > 0:
        B_star = m / b
        P_star = (r / a) * (1 - B_star / K)

        if B_star > 0 and B_star < K and P_star > 0:
            coexistence = (B_star, P_star)
            coexistence_eigs = np.linalg.eigvals(
                jacobian(B_star, P_star, T)
            )

            if np.all(np.real(coexistence_eigs) < 0):
                coexistence_status = "Locally asymptotically stable"
            else:
                coexistence_status = "Not locally asymptotically stable"

    return {
        "R0": R0,
        "origin": E0,
        "origin_eigenvalues": eig0,
        "pest_free": E1,
        "pest_free_eigenvalues": eig1,
        "pest_free_status": pest_free_status,
        "coexistence": coexistence,
        "coexistence_eigenvalues": coexistence_eigs,
        "coexistence_status": coexistence_status
    }


# ============================================================
# 6. AUTOMATIC SCIENTIFIC REMARKS FOR EACH TEMPERATURE
# ============================================================

def temperature_remark(T, R0, B_final, B_no_faw, P_final):
    """Generate a cautious interpretation of a model scenario."""
    crop_response = maize_response(T)
    baseline_response = maize_response(25.0)
    pest_growth = rm_faw(T)
    baseline_pest_growth = rm_faw(25.0)

    remarks = []

    # Direct temperature effect on maize
    if crop_response > baseline_response + 0.03:
        remarks.append(
            "The direct maize temperature response is higher "
            "than at the 25°C baseline."
        )
    elif crop_response < baseline_response - 0.03:
        remarks.append(
            "The direct maize temperature response is lower "
            "than at the 25°C baseline."
        )
    else:
        remarks.append(
            "The direct maize temperature response is close "
            "to that at the 25°C baseline."
        )

    # FAW invasion threshold
    if R0 < 1:
        remarks.append(
            "R₀ < 1: FAW cannot invade the pest-free crop "
            "equilibrium under this model's threshold criterion."
        )
    elif np.isclose(R0, 1.0):
        remarks.append(
            "R₀ is approximately 1: this is a threshold "
            "scenario and should be interpreted cautiously."
        )
    else:
        remarks.append(
            "R₀ > 1: FAW can invade near the pest-free "
            "equilibrium under the model's assumptions."
        )

    # FAW thermal response
    if pest_growth > baseline_pest_growth + 1e-9:
        remarks.append(
            "The model's FAW intrinsic growth-rate input is "
            "higher than at 25°C."
        )
    elif pest_growth < baseline_pest_growth - 1e-9:
        remarks.append(
            "The model's FAW intrinsic growth-rate input is "
            "lower than at 25°C."
        )
    else:
        remarks.append(
            "The FAW intrinsic growth-rate input is at its "
            "25°C reference value."
        )

    # Simulated pest-mediated difference
    loss = 100 * (1 - B_final / max(B_no_faw, 1e-12))

    remarks.append(
        f"The simulated final maize state is {loss:.1f}% "
        "below the no-FAW reference for the same temperature "
        "and initial maize state."
    )

    if P_final > 1.05 * 50:
        remarks.append(
            "The final FAW state exceeds 50 model population units; "
            "check the full trajectory before concluding that the "
            "population increased throughout the season."
        )
    else:
        remarks.append(
            "The final FAW state is at or below 50 model population "
            "units; inspect the trajectory to determine its trend."
        )

    return " ".join(remarks)


# ============================================================
# 7. SIDEBAR: USER CONTROLS
# ============================================================

st.sidebar.header("Simulation controls")

scenario_type = st.sidebar.radio(
    "Choose scenario group",
    [
        "Biological temperature scenarios",
        "Climate-warming scenarios"
    ]
)

if scenario_type == "Biological temperature scenarios":
    T = st.sidebar.selectbox(
        "Temperature (°C)",
        [20.0, 25.0, 26.0, 28.0, 30.0],
        index=1
    )
    scenario_name = f"Biological scenario at {T:.0f}°C"
else:
    warming = st.sidebar.selectbox(
        "Warming relative to 25°C",
        [0, 1, 2, 3],
        index=0,
        format_func=lambda x: f"+{x}°C"
    )
    T = 25.0 + warming
    scenario_name = f"Warming scenario +{warming}°C ({T:.0f}°C)"

initial_fraction = st.sidebar.slider(
    "Initial maize state as a fraction of K",
    min_value=0.01,
    max_value=1.0,
    value=0.10,
    step=0.01
)

B0 = initial_fraction * K

P0 = st.sidebar.number_input(
    "Initial FAW model population",
    min_value=0.0,
    max_value=1000.0,
    value=P0_default,
    step=10.0
)

st.sidebar.caption(
    "The initial FAW population is a model-scale initial condition, "
    "not a measured Trans Nzoia field count."
)

# ============================================================
# 8. RUN THE SELECTED SCENARIO
# ============================================================

times, B, P = simulate(T, B0, P0)
times_free, B_free = simulate_without_faw(T, B0)

B_final = float(B[-1])
P_final = float(P[-1])
B_free_final = float(B_free[-1])

R0 = reproduction_number(T)
loss_percent = 100 * (
    1 - B_final / max(B_free_final, 1e-12)
)

st.header(scenario_name)

# Summary metrics
m1, m2, m3, m4 = st.columns(4)

m1.metric("Temperature", f"{T:.1f} °C")
m2.metric("R₀", f"{R0:.3f}")
m3.metric("Final maize state", f"{B_final:.3f}")
m4.metric("Final FAW state", f"{P_final:.2f}")

st.caption(
    "The maize-state number uses the project's production-state scale "
    "(tonnes/ha-equivalent). It is not independently validated grain yield. "
    "FAW population values are in model units."
)

# ============================================================
# 9. PLOT MAIZE AND FAW TRAJECTORIES
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "Simulation graphs",
    "Temperature comparison",
    "Equilibrium and stability",
    "Model data and assumptions"
])

with tab1:
    st.subheader("Maize state over time")

    fig1, ax1 = plt.subplots()
    ax1.plot(times, B, label="With FAW")
    ax1.plot(
        times_free,
        B_free,
        linestyle="--",
        label="Without FAW"
    )
    ax1.set_xlabel("Time (days)")
    ax1.set_ylabel("Maize state (tonnes/ha-equivalent)")
    ax1.set_title(f"Maize trajectory at {T:.1f}°C")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    st.pyplot(fig1)
    plt.close(fig1)

    st.subheader("Fall Armyworm population state")

    fig2, ax2 = plt.subplots()
    ax2.plot(times, P, label="FAW population state")
    ax2.set_xlabel("Time (days)")
    ax2.set_ylabel("FAW model population units")
    ax2.set_title(f"FAW trajectory at {T:.1f}°C")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    st.pyplot(fig2)
    plt.close(fig2)

    st.subheader("Scientific interpretation")

    remarks = temperature_remark(
        T, R0, B_final, B_free_final, P_final
    )
    st.info(remarks)

    st.warning(
        "R₀ is calculated at B = K. It is not the instantaneous "
        "growth rate of FAW at every point in the simulation. "
        "Use the trajectory and equilibrium analysis together."
    )

with tab2:
    st.subheader("Compare biological and warming scenarios")

    # Includes the project's biological temperatures and all
    # four warming scenarios relative to the 25°C baseline.
    comparison_temperatures = [20.0, 25.0, 26.0, 27.0, 28.0, 30.0]

    rows = []

    for temp in comparison_temperatures:
        t, b, p = simulate(temp, B0, P0)
        _, b_free = simulate_without_faw(temp, B0)

        b_end = float(b[-1])
        p_end = float(p[-1])
        b_free_end = float(b_free[-1])
        r0_value = reproduction_number(temp)

        rows.append({
            "Temperature (°C)": temp,
            "Maize growth rate r(T)": r_maize(temp),
            "FAW intrinsic rate rm(T)": rm_faw(temp),
            "FAW mortality rate μ(T)": mu_faw(temp),
            "R₀": r0_value,
            "Final maize state": b_end,
            "Final FAW state": p_end,
            "No-FAW maize reference": b_free_end,
            "Simulated reduction (%)": 100 * (
                1 - b_end / max(b_free_end, 1e-12)
            ),
            "Scenario remarks": temperature_remark(
                temp, r0_value, b_end, b_free_end, p_end
            )
        })

    comparison = pd.DataFrame(rows)

    st.dataframe(
        comparison.round(4),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Final maize state across temperatures")

    fig3, ax3 = plt.subplots()
    ax3.plot(
        comparison["Temperature (°C)"],
        comparison["Final maize state"],
        marker="o",
        label="With FAW"
    )
    ax3.plot(
        comparison["Temperature (°C)"],
        comparison["No-FAW maize reference"],
        marker="s",
        linestyle="--",
        label="Without FAW"
    )
    ax3.set_xlabel("Temperature (°C)")
    ax3.set_ylabel("Final maize state")
    ax3.set_title("Temperature-scenario comparison")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    st.pyplot(fig3)
    plt.close(fig3)

    st.subheader("Remarks for all scenarios")

    for _, row in comparison.iterrows():
        with st.expander(
            f"Temperature {row['Temperature (°C)']:.0f}°C "
            f"| R₀ = {row['R₀']:.3f}"
        ):
            st.write(row["Scenario remarks"])

    csv_data = comparison.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download temperature comparison (CSV)",
        data=csv_data,
        file_name="maize_faw_temperature_scenarios.csv",
        mime="text/csv"
    )

with tab3:
    st.subheader("Equilibrium and local stability analysis")

    analysis = equilibrium_analysis(T)

    st.write("**Origin equilibrium:**")
    st.latex(r"E_0=(0,0)")
    st.write(
        "Jacobian eigenvalues:",
        np.round(analysis["origin_eigenvalues"], 5)
    )

    st.write("**Pest-free equilibrium:**")
    st.latex(r"E_1=(K,0)")
    st.write(
        "Jacobian eigenvalues:",
        np.round(analysis["pest_free_eigenvalues"], 5)
    )
    st.write("Interpretation:", analysis["pest_free_status"])

    st.write("**Coexistence equilibrium:**")

    if analysis["coexistence"] is not None:
        B_star, P_star = analysis["coexistence"]

        st.latex(
            r"E^*=(B^*,P^*)="
            + f"({B_star:.5f},\\ {P_star:.5f})"
        )
        st.write(
            "Jacobian eigenvalues:",
            np.round(analysis["coexistence_eigenvalues"], 5)
        )
        st.write("Interpretation:", analysis["coexistence_status"])
    else:
        st.write(
            "No biologically feasible positive coexistence equilibrium "
            "exists for this temperature under the current parameters."
        )

    st.markdown("**Basic reproduction number**")
    st.latex(r"R_0(T)=\frac{\beta(T)K}{\mu(T)}")

    st.write(
        "R₀ < 1 indicates that FAW cannot invade the pest-free "
        "equilibrium under the model's assumptions. R₀ > 1 indicates "
        "that the pest-free equilibrium is unstable to FAW invasion. "
        "This threshold statement assumes positive maize growth."
    )

with tab4:
    st.subheader("Model parameters")

    parameter_table = pd.DataFrame({
        "Parameter": [
            "K",
            "Initial maize state B₀",
            "Initial FAW state P₀",
            "rmax",
            "Tmin",
            "ToptB",
            "Tmax",
            "alpha25",
            "c_beta",
            "Simulation duration"
        ],
        "Value": [
            K, B0, P0, rmax, Tmin, ToptB, Tmax,
            alpha25, c_beta, t_end
        ],
        "Description": [
            "Empirical upper production benchmark (tonnes/ha)",
            "Selected fraction of K",
            "Model-scale initial FAW condition",
            "Selected seasonal maize growth scale",
            "Lower maize temperature threshold (°C)",
            "Optimum maize temperature (°C)",
            "Upper maize temperature threshold (°C)",
            "Calibrated maize-loss coefficient at 25°C",
            "Assumed FAW coefficient conversion factor",
            "Days"
        ]
    })

    st.dataframe(parameter_table, use_container_width=True, hide_index=True)

    st.subheader("Important modelling assumptions")

    st.markdown("""
    - Temperature is constant within each simulation and changes
      between scenarios.
    - Rainfall is fixed at a representative rainy-season condition;
      rainfall is not a dynamic model state.
    - FAW rates between 20°C, 25°C and 30°C use linear interpolation.
    - The coefficient alpha25 is a calibrated model coefficient,
      not a direct field measurement.
    - The model does not explicitly include pesticides, natural enemies,
      soil fertility, cultivar differences or stochastic daily weather.
    - Results are deterministic scenario simulations, not validated
      farm-level yield forecasts.
    """)

st.divider()
st.caption(
    "Research application: A Deterministic Mathematical Model of "
    "Maize Yield Response to Climate Warming and Fall Armyworm "
    "Dynamics in Trans Nzoia County, Kenya."
)


# In[ ]:




