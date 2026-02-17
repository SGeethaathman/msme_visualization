import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ================= COMMON DARK THEME FUNCTION =================
def apply_dark_theme(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        margin=dict(l=30, r=30, t=50, b=30)
    )
    return fig


# ================= DATA PREPROCESS =================
def load_and_preprocess_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    return df


# ================= KPI SCORECARDS =================
def get_kpi_scorecards(df_filtered):

    fig = make_subplots(
        rows=1,
        cols=4,
        specs=[[{'type': 'indicator'}]*4]
    )

    metrics = [
        ("Total Sales", df_filtered['Gross_Sales'].sum(), "₹", ",.0f"),
        ("Total Profit", df_filtered['Profit_Amount'].sum(), "₹", ",.0f"),
        ("Avg Margin", df_filtered['Profit_Margin_%'].mean(), "", ".1f"),
        ("Total Units", df_filtered['Quantity_Sold'].sum(), "", ",")
    ]

    for i, (label, value, prefix, fmt) in enumerate(metrics, 1):
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=value,
                title={"text": f"<b>{label}</b>"},
                number={'prefix': prefix, 'valueformat': fmt}
            ),
            row=1,
            col=i
        )

    fig.update_layout(height=180)
    return apply_dark_theme(fig)


# ================= SALES TREND =================
def get_sales_trend(df_filtered):

    trend = df_filtered.groupby('Date')['Gross_Sales'].sum().reset_index()

    fig = px.line(
        trend,
        x='Date',
        y='Gross_Sales',
        title='Daily Sales Performance'
    )

    fig.update_traces(line=dict(width=3))
    return apply_dark_theme(fig)


# ================= CATEGORY PERFORMANCE =================
def get_category_performance(df_filtered):

    cat_data = df_filtered.groupby('Product_Category')['Gross_Sales'].sum().reset_index()

    fig = px.bar(
        cat_data,
        x='Gross_Sales',
        y='Product_Category',
        orientation='h',
        title='Sales by Product Category',
        color='Gross_Sales',
        color_continuous_scale='Turbo'
    )

    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    return apply_dark_theme(fig)


# ================= LOCATION CHART =================
def get_location_chart(df_filtered):

    loc_data = df_filtered.groupby('Store_Location')['Gross_Sales'].sum().reset_index()

    fig = px.pie(
        loc_data,
        values='Gross_Sales',
        names='Store_Location',
        title='Revenue Share by Store Location',
        hole=0.4
    )

    fig.update_traces(textinfo='percent+label')
    return apply_dark_theme(fig)


# ================= TOP PRODUCTS =================
def get_top_products(df_filtered):

    prod_data = df_filtered.groupby('Product_Name')['Gross_Sales'].sum().reset_index()

    fig = px.bar(
        prod_data.sort_values("Gross_Sales", ascending=False),
        x='Product_Name',
        y='Gross_Sales',
        title='Top Products by Revenue',
        color='Gross_Sales',
        color_continuous_scale='Plasma'
    )

    return apply_dark_theme(fig)


# ================= PRICE VS QUANTITY =================
def get_price_vs_quantity_scatter(df_filtered):

    df_agg = df_filtered.groupby(
        ['Product_Name', 'Product_Category']
    ).agg({
        'Unit_Price': 'mean',
        'Quantity_Sold': 'sum',
        'Gross_Sales': 'sum'
    }).reset_index()

    fig = px.scatter(
        df_agg,
        x='Unit_Price',
        y='Quantity_Sold',
        color='Product_Category',
        size='Gross_Sales',
        hover_name='Product_Name',
        title='Price vs Quantity Analysis'
    )

    fig.update_traces(
        marker=dict(
            sizemode='area',
            opacity=0.85,
            line=dict(width=1, color='white')
        )
    )

    return apply_dark_theme(fig)


# ================= CUMULATIVE SALES =================
def get_cumulative_sales_chart(df_filtered):

    daily_sales = df_filtered.groupby('Date')['Gross_Sales'].sum().reset_index()
    daily_sales['Cumulative'] = daily_sales['Gross_Sales'].cumsum()

    fig = px.line(
        daily_sales,
        x='Date',
        y='Cumulative',
        title='Cumulative Sales Trend'
    )

    fig.update_traces(line=dict(width=3))
    return apply_dark_theme(fig)


# ================= QUANTITY BY PRODUCT =================
def get_quantity_by_product_chart(df_filtered, top_n=10):

    qty_data = (
        df_filtered.groupby('Product_Name')['Quantity_Sold']
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )

    fig = px.bar(
        qty_data,
        x='Quantity_Sold',
        y='Product_Name',
        orientation='h',
        title='Top Products by Quantity Sold',
        color='Quantity_Sold',
        color_continuous_scale='Turbo'
    )

    fig.update_layout(
    xaxis_title="Total Quantity Sold",
    yaxis_title="Product Name",
    xaxis=dict(showgrid=True),
    yaxis=dict(showgrid=False)
)

    return apply_dark_theme(fig)
