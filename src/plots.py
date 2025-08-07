import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from src.utils import empirical_cdf

def plot_ctr(results: dict[str, np.ndarray], i: int) -> None:
    df = {
        'Control CTR': results['ctrs_0'][i],
        'Treatment CTR': results['ctrs_1'][i]
    }
    data = []
    for label in df:
        data.append(go.Histogram(x=df[label], name=label, opacity=0.6))

    layout = go.Layout(
        title='CTR Distribution',
        xaxis=dict(title='CTR'),
        yaxis=dict(title='Frequency'),
        barmode='overlay'
    )
    fig = go.Figure(data=data, layout=layout)
    st.plotly_chart(fig, use_container_width=True)


def plot_views(results: dict[str, np.ndarray], i: int) -> None:
    df = {
        'Control Views': results['views_0'][i],
        'Treatment Views': results['views_1'][i]
    }
    data = []
    for label in df:
        data.append(go.Histogram(x=df[label], name=label, opacity=0.6))

    layout = go.Layout(
        title='User Views Distribution',
        xaxis=dict(title='Views'),
        yaxis=dict(title='Frequency'),
        barmode='overlay'
    )
    fig = go.Figure(data=data, layout=layout)
    st.plotly_chart(fig, use_container_width=True)


def plot_p_hist_all(results_pvals: dict[str, dict[str, np.ndarray]]) -> None:
    data = []
    for test_name in results_pvals:
        p_vals = results_pvals[test_name]['p_vals']
        data.append(go.Histogram(x=p_vals, name=test_name, opacity=0.6))

    layout = go.Layout(
        title='p-value Distribution (All Tests)',
        xaxis=dict(title='p-value'),
        yaxis=dict(title='Density'),
        barmode='overlay'
    )
    fig = go.Figure(data=data, layout=layout)
    st.plotly_chart(fig, use_container_width=True)


def plot_p_cdf_all(p_vals_dict, alpha=0.05):
    fig = go.Figure()

    for test_name in p_vals_dict:
        p_vals = p_vals_dict[test_name]['p_vals']
        p_vals_sorted, probs = empirical_cdf(p_vals)
        fig.add_trace(go.Scatter(x=p_vals_sorted, y=probs, mode='lines', name=test_name))

    fig.add_shape(
        type="line",
        x0=alpha, x1=alpha,
        y0=0, y1=1,
        line=dict(color="gray", dash="dash")
    )
    fig.add_shape(
        type="line",
        x0=0, x1=1,
        y0=0, y1=1,
        line=dict(color="gray", dash="dot")
    )
    fig.update_layout(
        title='Empirical CDF of p-values',
        xaxis_title='p-value',
        yaxis_title='Probability',
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_power(tests_results, alpha=0.05):
    powers = {
        test_name: np.mean(tests_results[test_name]['p_vals'] < alpha)
        for test_name in tests_results
    }

    fig = px.bar(
        x=list(powers.values()),
        y=list(powers.keys()),
        orientation='h',
        labels={'x': 'Statistical Power', 'y': 'Test'},
        title='Power of Statistical Tests',
        text=[f"{p*100:.1f}%" for p in powers.values()]
    )
    fig.update_traces(textposition='auto')
    fig.update_layout(xaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

def plot_ctr_plotly(results, i):
    import plotly.express as px
    import streamlit as st

    ctr_values = {
        'Control CTR': results['ctrs_0'][i],
        'Treatment CTR': results['ctrs_1'][i]
    }

    fig = px.histogram(
        x=list(ctr_values.values()),
        labels={'x': 'CTR'},
        nbins=20,
        opacity=0.6
    )
    fig.update_layout(
        title='CTR Distribution',
        xaxis_title='CTR',
        yaxis_title='Frequency',
        barmode='overlay'
    )
    st.plotly_chart(fig, use_container_width=True)
