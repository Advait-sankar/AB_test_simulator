import streamlit as st
from src.testdesign import design_binomial_experiment
from src.datagen import ABTestGenerator
from src.plots import plot_ctr, plot_views, plot_p_hist_all
from src.plots import plot_power, plot_p_cdf_all
from src.utils import apply_tests
from src.tests import t_test_clicks, t_test_ctr, mw_test
from src.tests import binom_test, bootstrap_test
import numpy as np

# Global results
result_dict_aa = None
result_dict_ab = None
p_vals_aa = None
p_vals_ab = None

def main():
    global result_dict_aa, result_dict_ab, p_vals_aa, p_vals_ab

    st.set_page_config(
        page_title="📊 A/B Testing Simulator | BA Portfolio",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.sidebar.title("📈 Data Generation Parameters")
    with st.sidebar.form(key='data_form'):
        base_ctr_pcnt = st.slider("Base CTR (%)", 0.1, 20.0, 2.0, 0.1)
        uplift_pcnt = st.slider("CTR Uplift (%)", 0.1, 10.0, 0.4, 0.1)
        skew = st.slider("Skew", 0.1, 4.0, 0.6, 0.1)
        ctr_beta = st.slider("Beta", 1, 2000, 1000, 1)
        sb_submit_button = st.form_submit_button("Apply Settings")

    st.title("📊 A/B Testing Simulator")
    st.caption("An interactive tool to simulate and visualize A/B testing performance using various statistical tests. Built for Business Analyst roles.")

    # Experiment Design Tab
    tab1, tab2, tab3, tab4 = st.tabs(["⚙️ Experiment Design", "📉 Data Distribution", "📊 A/A Test Results", "📈 A/B Test Results"])

    with tab1:
        st.header("⚙️ 1. Experiment Design")
        with st.form(key='exp_form'):
            col1, col2, col3 = st.columns(3)
            alpha = col1.slider("α (Type I Error)", 0.01, 0.2, 0.05, 0.01)
            beta = col2.slider("β (Type II Error)", 0.01, 0.8, 0.2, 0.01)
            mde = col3.slider("Minimum Detectable Effect (%)", 0.1, 10.0, 0.4, 0.1)
            n_samples = st.slider("Sample Size", 100, 10000, 1000, 100)
            ed_submit = st.form_submit_button("Estimate Sample Size & Run Simulations")

        if sb_submit_button or ed_submit:
            uplift = uplift_pcnt / 100
            base_ctr = base_ctr_pcnt / 100
            mde = mde / 100

            datagen_aa = ABTestGenerator(base_ctr, 0, ctr_beta, skew)
            result_dict_estimation = datagen_aa.generate_n_experiment(n_samples, 1)
            clicks_0 = result_dict_estimation['clicks_0'][0]
            views_0 = result_dict_estimation['views_0'][0]
            estimated_ctr_h0 = np.sum(clicks_0) / np.sum(views_0)

            min_samples_required = design_binomial_experiment(mde, estimated_ctr_h0, alpha, beta)

            st.markdown("### 📌 Sample Size Estimation")
            st.metric("Estimated CTR", f"{estimated_ctr_h0:.4f}")
            st.metric("Min Sample Required", f"{min_samples_required:,}")

            datagen_ab = ABTestGenerator(base_ctr, uplift, ctr_beta, skew)
            result_dict_aa = datagen_aa.generate_n_experiment(n_samples, 500)
            result_dict_ab = datagen_ab.generate_n_experiment(n_samples, 500)

    with tab2:
        st.header("📉 2. Data Distributions (H0 vs H1)")
        if result_dict_aa and result_dict_ab:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("H0 (No Uplift)")
                plot_ctr(result_dict_aa, 0)
                plot_views(result_dict_aa, 0)
            with col2:
                st.subheader("H1 (With Uplift)")
                plot_ctr(result_dict_ab, 0)
                plot_views(result_dict_ab, 0)

    with tab3:
        st.header("📊 3. A/A Test Results")
        if result_dict_aa:
            test_config = {
                'T-test, clicks': t_test_clicks,
                'T-test, CTR': t_test_ctr,
                'Mann–Whitney, clicks': mw_test,
                'Binomial, CTR': binom_test,
                'Bootstrap, CTR': bootstrap_test
            }
            p_vals_aa = apply_tests(result_dict_aa, test_config=test_config)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### p-values Histogram (H0)")
                plot_p_hist_all(p_vals_aa)
            with col2:
                st.markdown("#### Empirical CDF (H0)")
                plot_p_cdf_all(p_vals_aa)

    with tab4:
        st.header("📈 4. A/B Test Results")
        if result_dict_ab:
            p_vals_ab = apply_tests(result_dict_ab, test_config=test_config)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### p-values Histogram (H1)")
                plot_p_hist_all(p_vals_ab)
            with col2:
                st.markdown("#### Empirical CDF (H1)")
                plot_p_cdf_all(p_vals_ab)

            st.markdown("#### Statistical Power of Each Test")
            plot_power(p_vals_ab, alpha=alpha)

if __name__ == '__main__':
    main()
