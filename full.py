import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
print(">>> Running from:", os.path.abspath(__file__))

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Retirement Simulator",
    layout="wide",
)

pd.set_option("display.float_format", "{:,.2f}".format)

# ---------------------------------------------------------
# CUSTOM CSS FOR BOLD + LARGE SIDEBAR INPUTS
# ---------------------------------------------------------
st.markdown(
    """
<style>
    /* Make sidebar labels bold and larger */
    .css-1p3j8o1, .css-17eq0hr, .css-1d391kg {
        font-weight: 700 !important;
        font-size: 18px !important;
    }

    /* Make number input text bold */
    input[type=number] {
        font-weight: 700 !important;
        font-size: 18px !important;
    }

    /* Make slider labels bold */
    .stSlider label {
        font-weight: 700 !important;
        font-size: 18px !important;
    }

    /* Make checkbox labels bold */
    .stCheckbox label {
        font-weight: 700 !important;
        font-size: 18px !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# SIDEBAR — INPUTS
# ---------------------------------------------------------
st.sidebar.header("Simulation Inputs")

# Income & Spending
st.sidebar.subheader("Income & Spending")
ss_annual = st.sidebar.number_input("Social Security (Annual)", value=36000)
ss_start_age = st.sidebar.number_input("SS Start Age", value=62)
target_annual = st.sidebar.number_input("Target Spending (Annual)", value=60000)
cola = st.sidebar.slider("COLA (%)", 0.0, 5.0, 2.5)

# Portfolio Accounts
st.sidebar.subheader("Portfolio Accounts")
tax_deferred_start = st.sidebar.number_input("Tax-Deferred (401k / IRA)", value=100000)
taxable_start = st.sidebar.number_input("Taxable Brokerage (Stocks)", value=100000)

# NEW — Emergency Cash
st.sidebar.subheader("Emergency Cash")
emergency_cash = st.sidebar.number_input("Emergency Cash Available", value=25000)

expected_return = st.sidebar.slider("Expected Return (%)", 0.0, 12.0, 5.0)
volatility = st.sidebar.slider("Volatility (%)", 0.0, 25.0, 10.0)
tax_rate = st.sidebar.slider("Tax Rate (%)", 0.0, 30.0, 12.0)
mc_runs = st.sidebar.number_input("Monte Carlo Runs", value=1000)

# RMD & Roth
st.sidebar.subheader("RMD & Roth")
reinvest_rmd = st.sidebar.checkbox("Reinvest RMD?", value=True)
roth_conversion = st.sidebar.number_input("Optional Roth Conversion Amount", value=0)
roth_tax_rate = st.sidebar.slider("Conversion Tax Rate (%)", 0.0, 30.0, 12.0)

# Mortgage
st.sidebar.subheader("Mortgage")
mort_balance = st.sidebar.number_input("Current Mortgage Balance", value=0)
mort_rate = st.sidebar.number_input("Interest Rate (%)", value=0)
mort_payment = st.sidebar.number_input("Monthly Payment (P+I)", value=0)
mort_escrow = st.sidebar.number_input("Escrow Amount", value=0)
extra_principal = st.sidebar.number_input("Extra Principal Payment", value=0)

# Home
st.sidebar.subheader("Home")
home_value = st.sidebar.number_input("Home Value", value=500000)
home_appreciation = st.sidebar.slider("Home Appreciation (%)", 0.0, 10.0, 3.0)

# Simulation Settings
st.sidebar.subheader("Simulation Settings")
start_age = st.sidebar.number_input("Start Age", value=55)
end_age = st.sidebar.number_input("End Age", value=86)
years = end_age - start_age

# ---------------------------------------------------------
# SIMULATION ENGINE
# ---------------------------------------------------------
def run_single_path_with_projection(
    tax_deferred_start,
    taxable_start,
    emergency_cash,   # NEW
    start_year,
    start_age_you,
    start_age_spouse,
    years,
    target_gross_monthly,
    ss_annual,
    ss_start_age,
    cola,
    inflation,
    mean_return,
    volatility,
    effective_tax_rate,
    rmd_rate,
    reinvest_excess_rmd,
    mortgage_balance_start,
    mortgage_rate_annual,
    mortgage_payment_monthly,
    mortgage_escrow_monthly,
    home_value_start,
    home_appreciation_annual,
    random_seed=None,
):
    if random_seed is not None:
        np.random.seed(random_seed)

    months = years * 12

    ss_base_monthly = ss_annual / 12
    monthly_return_mean = (1 + mean_return) ** (1 / 12) - 1
    monthly_vol = volatility / np.sqrt(12)
    monthly_cola = (1 + cola) ** (1 / 12) - 1
    monthly_inflation = (1 + inflation) ** (1 / 12) - 1

    tax_deferred_balance = tax_deferred_start
    taxable_balance = taxable_start
    ss_current = ss_base_monthly
    target_income = target_gross_monthly

    home_value_local = home_value_start
    mortgage_balance_local = mortgage_balance_start
    monthly_home_appreciation = (1 + home_appreciation_annual) ** (1 / 12) - 1

    yearly_rows = []

    for m in range(months):
        year_index = m // 12
        month_in_year = m % 12
        current_year = start_year + year_index
        age_you = start_age_you + year_index
        age_spouse = start_age_spouse + year_index

        # Home appreciation
        home_value_local *= 1 + monthly_home_appreciation

        # Mortgage amortization
        if mortgage_balance_local > 0:
            interest_component = mortgage_balance_local * (mortgage_rate_annual / 12)
            principal_component = (
                mortgage_payment_monthly - interest_component - mortgage_escrow_monthly
            )
            if principal_component < 0:
                principal_component = 0
            if principal_component > mortgage_balance_local:
                principal_component = mortgage_balance_local
            mortgage_balance_local -= principal_component
        else:
            interest_component = 0.0
            principal_component = 0.0

        # Apply market return
        r = np.random.normal(monthly_return_mean, monthly_vol)
        tax_deferred_balance *= 1 + r
        taxable_balance *= 1 + r

        # Social Security
        if age_you >= ss_start_age:
            gross_ss = ss_current
        else:
            gross_ss = 0.0

        # Needed withdrawal
        needed_withdrawal = max(0.0, target_income - gross_ss)

        # RMD
        if age_you >= 73:
            annual_rmd = tax_deferred_balance * rmd_rate
            rmd_monthly = annual_rmd / 12
        else:
            annual_rmd = 0.0
            rmd_monthly = 0.0

        # Withdraw from taxable first
        withdrawal_from_taxable = min(needed_withdrawal, taxable_balance)
        remaining_need = needed_withdrawal - withdrawal_from_taxable

        # Then tax-deferred
        withdrawal_from_tax_deferred = max(remaining_need, rmd_monthly)
        if withdrawal_from_tax_deferred > tax_deferred_balance:
            withdrawal_from_tax_deferred = tax_deferred_balance

        # Taxes
        taxable_income = (
            0.85 * gross_ss
            + withdrawal_from_tax_deferred
            + withdrawal_from_taxable
        )
        taxes = taxable_income * effective_tax_rate

        # Apply withdrawals
        tax_deferred_balance -= withdrawal_from_tax_deferred
        taxable_balance -= withdrawal_from_taxable

        # Reinvest excess RMD
        excess_rmd = max(0.0, withdrawal_from_tax_deferred - remaining_need)
        if reinvest_excess_rmd and excess_rmd > 0:
            taxable_balance += excess_rmd

        # COLA
        if month_in_year == 0 and age_you > ss_start_age:
            ss_current *= (1 + monthly_cola)

        if month_in_year == 11:
            home_equity = home_value_local - mortgage_balance_local
            total_cash_need_annual = (target_income * 12) + (taxes * 12)
            ss_annual_value = gross_ss * 12
            ss_coverage_gap = total_cash_need_annual - ss_annual_value

            yearly_rows.append(
                {
                    "Year": current_year,
                    "Age_You": age_you,
                    "Age_Spouse": age_spouse,
                    "SS_Annual": ss_annual_value,
                    "Target_Annual": target_income * 12,
                    "Taxes_Annual": taxes * 12,
                    "Total_Cash_Need_Annual": total_cash_need_annual,
                    "SS_Coverage_Gap": ss_coverage_gap,
                    "Needed_Withdrawal_Annual": needed_withdrawal * 12,
                    "RMD_Annual": annual_rmd,
                    "Withdrawal_Annual": (withdrawal_from_tax_deferred + withdrawal_from_taxable) * 12,
                    "Tax_Deferred_End": tax_deferred_balance,
                    "Taxable_Account_End": taxable_balance,
                    "Emergency_Cash": emergency_cash,  # NEW
                    "Portfolio_Total": tax_deferred_balance + taxable_balance,
                    "Home_Value": home_value_local,
                    "Mortgage_Balance": mortgage_balance_local,
                    "Home_Equity": home_equity,
                    "Total_Wealth": tax_deferred_balance
                    + taxable_balance
                    + home_equity
                    + emergency_cash,  # NEW
                }
            )

    return pd.DataFrame(yearly_rows)


def monte_carlo_retirement_engine(simulations=1000, emergency_cash=0, **kwargs):
    final_balances = []
    success_count = 0

    for _ in range(simulations):
        projection = run_single_path_with_projection(
            emergency_cash=emergency_cash,
            **kwargs
        )
        final_balance = projection["Total_Wealth"].iloc[-1]
        final_balances.append(final_balance)

        if (projection["Tax_Deferred_End"] > 0).all():
            success_count += 1

    success_rate = success_count / simulations
    df_mc = pd.DataFrame({"Final Balance": final_balances})
    return df_mc, success_rate

# ---------------------------------------------------------
# WRAPPER FOR STREAMLIT
# ---------------------------------------------------------
def run_simulation():
    df_mc, success_rate = monte_carlo_retirement_engine(
        simulations=mc_runs,
        emergency_cash=emergency_cash,  # NEW
        tax_deferred_start=tax_deferred_start,
        taxable_start=taxable_start,
        start_year=2026,
        start_age_you=start_age,
        start_age_spouse=start_age - 1,
        years=years,
        target_gross_monthly=target_annual / 12,
        ss_annual=ss_annual,
        ss_start_age=ss_start_age,
        cola=cola / 100,
        inflation=0.025,
        mean_return=expected_return / 100,
        volatility=volatility / 100,
        effective_tax_rate=tax_rate / 100,
        rmd_rate=0.04,
        reinvest_excess_rmd=reinvest_rmd,
        mortgage_balance_start=mort_balance,
        mortgage_rate_annual=mort_rate / 100,
        mortgage_payment_monthly=mort_payment + mort_escrow + extra_principal,
        mortgage_escrow_monthly=mort_escrow,
        home_value_start=home_value,
        home_appreciation_annual=home_appreciation / 100,
    )

    projection_df = run_single_path_with_projection(
        tax_deferred_start=tax_deferred_start,
        taxable_start=taxable_start,
        emergency_cash=emergency_cash,  # NEW
        start_year=2026,
        start_age_you=start_age,
        start_age_spouse=start_age - 1,
        years=years,
        target_gross_monthly=target_annual / 12,
        ss_annual=ss_annual,
        ss_start_age=ss_start_age,
        cola=cola / 100,
        inflation=0.025,
        mean_return=expected_return / 100,
        volatility=volatility / 100,
        effective_tax_rate=tax_rate / 100,
        rmd_rate=0.04,
        reinvest_excess_rmd=reinvest_rmd,
        mortgage_balance_start=mort_balance,
        mortgage_rate_annual=mort_rate / 100,
        mortgage_payment_monthly=mort_payment + mort_escrow + extra_principal,
        mortgage_escrow_monthly=mort_escrow,
        home_value_start=home_value,
        home_appreciation_annual=home_appreciation / 100,
        random_seed=42,
    )

    payoff_year = None
    if "Mortgage_Balance" in projection_df.columns:
        zero_balance = projection_df[projection_df["Mortgage_Balance"] <= 0]
        if not zero_balance.empty:
            payoff_year = int(zero_balance.iloc[0]["Year"])

    return projection_df, df_mc, success_rate, payoff_year

# ---------------------------------------------------------
# RUN SIMULATION
# ---------------------------------------------------------
df, df_mc, success_rate, mortgage_payoff_year = run_simulation()

# ---------------------------------------------------------
# SUMMARY CARDS
# ---------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Success Rate", f"{success_rate*100:.1f}%")
col2.metric("Final Wealth (Median)", f"${df_mc['Final Balance'].median():,.0f}")
col3.metric(
    "Mortgage Payoff Year",
    f"{mortgage_payoff_year if mortgage_payoff_year is not None else 'N/A'}",
)
col4.metric("Emergency Cash", f"${emergency_cash:,.0f}")  # NEW

final_td = df["Tax_Deferred_End"].iloc[-1]
final_taxable = df["Taxable_Account_End"].iloc[-1]
final_portfolio = final_td + final_taxable
col5.metric("Final Portfolio (401k + Stocks)", f"${final_portfolio:,.0f}")

# ---------------------------------------------------------
# TABLE STYLING
# ---------------------------------------------------------
def style_projection_table(df_in):
    df_styled = df_in.copy()

    dollar_cols = [
        "SS_Annual",
        "Target_Annual",
        "Needed_Withdrawal_Annual",
        "Home_Value",
        "Mortgage_Balance",
        "Home_Equity",
        "RMD_Annual",
        "Withdrawal_Annual",
        "Taxes_Annual",
        "Tax_Deferred_End",
        "Taxable_Account_End",
        "Portfolio_Total",
        "Total_Wealth",
        "Total_Cash_Need_Annual",
        "SS_Coverage_Gap",
        "Emergency_Cash",  # NEW
    ]

    for col in dollar_cols:
        if col in df_styled.columns:
            df_styled[col] = df_styled[col].apply(lambda x: f"${x:,.0f}")

    return df_styled.style.set_properties(
        **{
            "font-size": "18px",
            "font-weight": "600",
        }
    )

# ---------------------------------------------------------
# PROJECTION TABLE
# ---------------------------------------------------------
st.subheader("📘 Projection Table")
st.dataframe(style_projection_table(df), use_container_width=True)

# ---------------------------------------------------------
# CHARTS
# ---------------------------------------------------------
st.subheader("📈 Total Wealth Over Time")

fig, ax = plt.subplots(figsize=(8, 4.8))

ax.plot(
    df["Year"],
    df["Tax_Deferred_End"],
    label="401k / IRA (Tax-Deferred)",
    linewidth=1.5,
)
ax.plot(
    df["Year"],
    df["Taxable_Account_End"],
    label="Stocks (Taxable)",
    linewidth=1.5,
)
ax.plot(
    df["Year"],
    df["Portfolio_Total"],
    label="Portfolio Total",
    linewidth=1.5,
    linestyle="--",
)
ax.plot(
    df["Year"],
    df["Total_Wealth"],
    label="Total Wealth",
    linewidth=1.5,
)

ax.set_title("Total Wealth Over Time", fontsize=6, fontweight="bold")
ax.set_xlabel("Year", fontsize=6, fontweight="bold")
ax.set_ylabel("Dollars ($)", fontsize=6, fontweight="bold")
ax.ticklabel_format(style="plain", axis="y")
ax.tick_params(axis="both", labelsize=6)
ax.legend(fontsize=6)

st.pyplot(fig)

st.subheader("📊 SS vs Total Cash Need")

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(df["Year"], df["SS_Annual"], label="Social Security", linewidth=1.5)
ax.plot(
    df["Year"],
    df["Total_Cash_Need_Annual"],
    label="Total Cash Need",
    linewidth=1.5,
)

ax.set_title("SS vs Total Cash Need", fontsize=6, fontweight="bold")
ax.set_xlabel("Year", fontsize=6, fontweight="bold")
ax.set_ylabel("Dollars ($)", fontsize=6, fontweight="bold")
ax.ticklabel_format(style="plain", axis="y")
ax.tick_params(axis="both", labelsize=6)
ax.legend(fontsize=6)

st.pyplot(fig)