import streamlit as st
import numpy as np
import plotly.express as px
from src.testdesign import design_binomial_experiment
from src.datagen import ABTestGenerator
from src.utils import apply_tests
from src.tests import t_test_clicks, t_test_ctr, mw_test, binom_test, bootstrap_test
from src.plots import (
    plot_ctr_plotly, plot_views_plotly,
    plot_p_hist_all_plotly, plot_p_cdf_all_plotly,
    plot_power_plotly
)

# Global results
result_dict_aa = None
result_dict_ab = None
p_vals_aa = None
p_vals_ab = None

def main():
    global result_dict_aa, result_dict_ab, p_vals_aa, p_vals_ab

    st.set_page_config(
        page_title='📊 A/B Testing Simulator',
        layout='wide',
        initial_sidebar_state='expanded'
    )

    with st.sidebar:
        st.markdown("## 🧪 Data Generation Settings")
        base_ctr_pcnt = st.slider("Base CTR (%)", 0.1, 20.0, 2.0, 0.1)
        uplift_pcnt = st.slider("CTR Uplift (%)", 0.1, 10.0, 0.4, 0.1)
        skew = st.slider("Skew", 0.1, 4.0, 0.6, 0.1)
        ctr_beta = st.slider("Beta", 1, 2000, 1000, 1)
        st.markdown("---")
        st.markdown("## 🧬 Experiment Design Settings")
        alpha = st.slider("Alpha (Type I Error)", 0.01, 0.2, 0.05, 0.01)
        beta = st.slider("Beta (Type II Error)", 0.01, 0.8, 0.2, 0.01)
        mde = st.slider("Minimum Detectable Effect (%)", 0.1, 10.0, 0.4, 0.1)
        n_samples = st.slider("Sample Size per Group", 100, 10000, 1000, 100)
        submit_button = st.button("Apply Settings")

    st.title("📈 A/B Testing Simulator")
    st.markdown("An interactive simulator to design, run, and analyze A/B Tests. Built for Business Analyst roles.")

    if submit_button:
        base_ctr = base_ctr_pcnt / 100
        uplift = uplift_pcnt / 100
        mde = mde / 100

        datagen_aa = ABTestGenerator(base_ctr, 0, ctr_beta, skew)
        datagen_ab = ABTestGenerator(base_ctr, uplift, ctr_beta, skew)

        result_dict_est = datagen_aa.generate_n_experiment(n_samples, 1)
        estimated_ctr_h0 = np.sum(result_dict_est['clicks_0'][0]) / np.sum(result_dict_est['views_0'][0])

        min_samples_required = design_binomial_experiment(
            mde=mde, p_0=estimated_ctr_h0, alpha=alpha, beta=beta
        )

        st.success(f"Estimated Base CTR: {estimated_ctr_h0:.4f} | Required Sample Size: {min_samples_required}")

        result_dict_aa = datagen_aa.generate_n_experiment(n_samples, 500)
        result_dict_ab = datagen_ab.generate_n_experiment(n_samples, 500)

        test_config = {
            'T-test (clicks)': t_test_clicks,
            'T-test (CTR)': t_test_ctr,
            'Mann–Whitney': mw_test,
            'Binomial': binom_test,
            'Bootstrap': bootstrap_test
        }

        p_vals_aa = apply_tests(result_dict_aa, test_config)
        p_vals_ab = apply_tests(result_dict_ab, test_config)

    if result_dict_aa:
        tab1, tab2, tab3 = st.tabs(["📉 Data Distribution", "🧪 A/A Test Results", "🚀 A/B Test Results"])
        with tab1:
            st.markdown("### Control (H0) Distributions")
            plot_ctr_plotly(result_dict_aa, 0)
            plot_views_plotly(result_dict_aa, 0)

            st.markdown("### Treatment (H1) Distributions")
            plot_ctr_plotly(result_dict_ab, 0)
            plot_views_plotly(result_dict_ab, 0)

        with tab2:
            st.markdown("### p-value Distributions (H0)")
            plot_p_hist_all_plotly(p_vals_aa)
            plot_p_cdf_all_plotly(p_vals_aa)

        with tab3:
            st.markdown("### p-value Distributions (H1)")
            plot_p_hist_all_plotly(p_vals_ab)
            plot_p_cdf_all_plotly(p_vals_ab)
            plot_power_plotly(p_vals_ab, alpha)

if __name__ == "__main__":
    main()
