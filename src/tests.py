import numpy as np
import plotly.graph_objs as go
import streamlit as st
from src.utils import empirical_cdf


def plot_ctr(results: dict[str, np.ndarray], i: int) -> None:
    ctrs_0 = results['ctrs_0'][i]
    ctrs_1 = results['ctrs_1'][i]

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=ctrs_0, name='Group A - CTR', opacity=0.6))
    fig.add_trace(go.Histogram(x=ctrs_1, name='Group B - CTR', opacity=0.6))

    fig.update_layout(
        barmode='overlay',
        title='Ground Truth CTR Distribution (H0 vs H1)',
        xaxis_title='CTR',
        yaxis_title='Probability',
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_views(results: dict[str, np.ndarray], i: int) -> None:
    views_0 = results['views_0'][i]
    views_1 = results['views_1'][i]

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=views_0, name='Group A - Views', opacity=0.6))
    fig.add_trace(go.Histogram(x=views_1, name='Group B - Views', opacity=0.6))

    fig.update_layout(
        barmode='overlay',
        title='Ground Truth Views Distribution (H0 vs H1)',
        xaxis_title='Views',
        yaxis_title='Probability',
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_p_hist_all(results_pvals, hist_alpha=0.6) -> None:
    fig = go.Figure()
    for test_name, data in results_pvals.items():
        fig.add_trace(go.Histogram(
            x=data['p_vals'],
            name=test_name,
            opacity=hist_alpha,
            histnorm='probability'
        ))

    fig.update_layout(
        barmode='overlay',
        title='P-Value Distributions Across Tests',
        xaxis_title='p-value',
        yaxis_title='Probability',
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_p_cdf_all(p_vals_dict, alpha=0.05):
    fig = go.Figure()
    for test_name, data in p_vals_dict.items():
        p_vals_sorted, probs = empirical_cdf(data['p_vals'])
        fig.add_trace(go.Scatter(x=p_vals_sorted, y=probs, mode='lines', name=test_name))

    fig.add_shape(
        type='line', x0=alpha, y0=0, x1=alpha, y1=1,
        line=dict(color='gray', dash='dash')
    )

    fig.update_layout(
        title='Empirical CDF of p-values',
        xaxis_title='p-value',
        yaxis_title='Probability',
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_power(tests_results, alpha=0.05):
    powers = {
        test_name: np.mean(data['p_vals'] < alpha)
        for test_name, data in tests_results.items()
    }

    fig = go.Figure(go.Bar(
        x=list(powers.values()),
        y=list(powers.keys()),
        orientation='h'
    ))

    fig.update_layout(
        title='Statistical Power of Tests',
        xaxis_title='Power',
        yaxis_title='Test',
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)
