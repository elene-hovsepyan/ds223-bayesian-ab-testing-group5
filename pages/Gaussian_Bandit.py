import pandas as pd
import streamlit as st

from gaussian_bandit import simulate_gaussian_bandit


st.title("Gaussian Bandit")
st.caption("Three arms with continuous rewards and independent Normal(0, 25) priors.")

with st.form("gaussian_form"):
    columns = st.columns(3)
    true_means = [
        column.number_input(f"Arm {i + 1} true mean", value=float(i + 1))
        for i, column in enumerate(columns)
    ]
    sigma = st.number_input("Known σ (standard deviation)", value=1.0)
    rounds = st.number_input("Number of rounds", value=1000, step=1)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    submitted = st.form_submit_button("Run")

if submitted:
    try:
        result = simulate_gaussian_bandit(true_means, sigma, rounds, seed)
    except ValueError as error:
        st.error(str(error))
    else:
        st.session_state.gaussian_result = {
            "true_means": true_means,
            "sigma": sigma,
            "rounds": rounds,
            "seed": seed,
            "result": result,
        }

if "gaussian_result" in st.session_state:
    run = st.session_state.gaussian_result
    result = run["result"]
    st.subheader("Last completed run")
    st.caption(f"σ = {run['sigma']}, rounds = {run['rounds']}, seed = {run['seed']}")
    st.dataframe(
        pd.DataFrame({
            "Arm": [1, 2, 3],
            "True mean": run["true_means"],
            "Final posterior mean": result["posterior_means"],
            "Selection count": result["counts"],
        }),
        width="stretch",
        hide_index=True,
    )
    st.metric("Total observed reward", f"{result['total_reward']:.6f}")
    st.write(f"**Recommendation by final posterior mean: Arm {result['recommendation']}**")
