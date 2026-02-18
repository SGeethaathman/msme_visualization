import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
from dash import Dash, dcc, html

# ==============================
# DATABASE CONNECTION
# ==============================

host = "aws-1-ap-southeast-1.pooler.supabase.com"
port = "6543"
database = "postgres"
user = "postgres.lmkavbsqutyrshafjvak"
password = "DataNetra123!"

engine = create_engine(
    f"postgresql://{user}:{password}@{host}:{port}/{database}"
)

df = pd.read_sql_table('demo_rawdata1', con=engine)

df.columns = df.columns.str.strip()
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# ==============================
# COLOR SYSTEM
# ==============================

PAGE_BG = "#F3F4F6"
CARD_BG = "#FFFFFF"
HEADER_BG = "linear-gradient(90deg,#1E3A8A,#2563EB)"

TOP_COLORS = [
    "#2563EB",
    "#7C3AED",
    "#EC4899",
    "#F97316",
    "#10B981",
    "#6366F1",
    "#14B8A6",
    "#8B5CF6",
    "#E11D48",
    "#22C55E"
]

# ==============================
# GLOBAL FIG STYLE
# ==============================

def style_fig(fig):
    fig.update_layout(
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color="#111827", family="Segoe UI"),
        title=dict(x=0.5, font=dict(size=18)),
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)"),
        margin=dict(l=40, r=40, t=60, b=40),
        height=420
    )
    return fig

# ==============================
# KPI DATA
# ==============================

total_sales = df['Gross_Sales'].sum()
total_profit = df['Profit_Amount'].sum()
total_qty = df['Quantity_Sold'].sum()
avg_margin = df['Profit_Margin_%'].mean()

def kpi_card(value, label):
    return html.Div([
        html.H2(value, style={"margin": "0", "fontWeight": "700"}),
        html.P(label, style={"margin": "0", "color": "#6B7280"})
    ], style={
        "backgroundColor": CARD_BG,
        "padding": "25px",
        "borderRadius": "14px",
        "boxShadow": "0px 6px 18px rgba(0,0,0,0.15)",
        "textAlign": "center",
        "flex": "1",
        "margin": "10px"
    })

# ==============================
# DATA PREPARATION
# ==============================

daily_sales = df.groupby('Date', as_index=False)['Gross_Sales'].sum()
daily_sales['Cumulative'] = daily_sales['Gross_Sales'].cumsum()

category_sales = df.groupby('Product_Category', as_index=False)['Gross_Sales'].sum()
store_sales = df.groupby('Store_Location', as_index=False)['Gross_Sales'].sum()

top_products = df.groupby('Product_Name', as_index=False)['Quantity_Sold'].sum().sort_values(
    by="Quantity_Sold", ascending=False).head(10)

product_perf = df.groupby(
    ['Product_Name', 'Product_Category'],
    as_index=False
).agg({
    'Unit_Price': 'mean',
    'Quantity_Sold': 'sum',
    'Gross_Sales': 'sum'
})

# ==============================
# CHARTS
# ==============================

# 2️⃣ Daily Sales
fig1 = px.line(
    daily_sales,
    x="Date",
    y="Gross_Sales",
    markers=True,
    title="Daily Sales Performance"
)
fig1.update_traces(
    line=dict(color="#1F51FF", width=3),
    marker=dict(size=6, color="#1F51FF")
)
fig1 = style_fig(fig1)

# 3️⃣ Sales by Category (VALUES ADDED)
fig2 = px.bar(
    category_sales,
    x="Gross_Sales",
    y="Product_Category",
    orientation="h",
    title="Sales by Product Category",
    color="Product_Category",
    text_auto=True,
    color_discrete_sequence=["#2563EB", "#7C3AED", "#EC4899", "#10B981"]
)
fig2.update_traces(textposition="outside")
fig2 = style_fig(fig2)

# 4️⃣ Donut
fig3 = px.pie(
    store_sales,
    names="Store_Location",
    values="Gross_Sales",
    hole=0.55,
    title="Sales Distribution by Store Location",
    color_discrete_sequence=["#2563EB", "#7C3AED", "#F97316", "#EC4899"]
)
fig3 = style_fig(fig3)

# 5️⃣ Top 10 Products (VALUES ADDED)
fig4 = px.bar(
    top_products,
    x="Quantity_Sold",
    y="Product_Name",
    orientation="h",
    title="Top 10 Products by Volume",
    text_auto=True
)
fig4.update_traces(marker_color=TOP_COLORS, textposition="outside")
fig4.update_layout(yaxis=dict(autorange="reversed"))
fig4 = style_fig(fig4)

# 6️⃣ Cumulative Revenue (No markers)
fig5 = px.line(
    daily_sales,
    x="Date",
    y="Cumulative",
    title="Total Cumulative Revenue Growth"
)
fig5.update_traces(
    line=dict(color="#1F51FF", width=3)
)
fig5 = style_fig(fig5)

# 7️⃣ Scatter
fig6 = px.scatter(
    product_perf,
    x="Unit_Price",
    y="Quantity_Sold",
    size="Gross_Sales",
    color="Product_Category",
    hover_name="Product_Name",
    title="Product Performance: Price vs Quantity",
    size_max=45,
    color_discrete_sequence=["#2563EB", "#7C3AED", "#EC4899", "#10B981"]
)
fig6 = style_fig(fig6)

# ==============================
# DASH LAYOUT
# ==============================

app = Dash(__name__)

app.layout = html.Div([

    html.Div("SALES FORECAST ANALYSIS DASHBOARD ", style={
        "background": HEADER_BG,
        "color": "white",
        "padding": "18px",
        "textAlign": "center",
        "fontSize": "24px",
        "fontWeight": "600",
        "letterSpacing": "1px"
    }),

    html.Div([
        kpi_card(f"{total_sales:,.0f}", "Total Sales"),
        kpi_card(f"{total_profit:,.0f}", "Total Profit"),
        kpi_card(f"{total_qty:,.0f}", "Total Quantity"),
        kpi_card(f"{avg_margin:.1f}%", "Avg Margin")
    ], style={"display": "flex", "justifyContent": "space-between"}),

    dcc.Graph(figure=fig1),
    dcc.Graph(figure=fig2),
    dcc.Graph(figure=fig3),
    dcc.Graph(figure=fig4),
    dcc.Graph(figure=fig5),
    dcc.Graph(figure=fig6),

], style={
    "backgroundColor": PAGE_BG,
    "padding": "20px"
})

def run_dash():
    app.run(host="127.0.0.1", port=8050, debug=False)


