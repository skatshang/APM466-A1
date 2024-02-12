import pandas as pd
import numpy as np
from datetime import date
import scipy.optimize as optimize
import plotly.graph_objects as go


def calculate_ytm(price, coupon_rate, time_to_maturity):
    coupon = coupon_rate / 100 * 1000 / 2
    periods = int(time_to_maturity * 2) + 1
    dt = [(i + 1) / 2 for i in range(periods)]
    ytm_func = lambda y: sum([coupon / (1 + y / 2) ** (2 * t) for t in dt]) + \
                         1000 / (1 + y / 2) ** (periods) - price
    ytm_rate = optimize.newton(ytm_func, 0.05)
    return ytm_rate


def get_ytm(row, date_column):
    Price = float(row[date_column])
    coupon_rate = row['Coupon Rate (%)']
    time_to_maturity = row['Years to Maturity']
    next_coupon_date = pd.Timestamp('2024-03-01')
    last_coupon_date = next_coupon_date - pd.DateOffset(months=6)
    days_last_coupon = (pd.Timestamp(date_column) - last_coupon_date).days
    coupon_period = 365.25 / 2
    accrued_interest = coupon_rate / 100 * 1000 / 2 * (days_last_coupon / coupon_period)
    dirty_price = Price + accrued_interest
    return calculate_ytm(dirty_price, coupon_rate, time_to_maturity)


def get_spot_rate(bond, prev_spots, coupon, periods, date_column):
    pv_coupons = sum([coupon / (1 + prev_spots[t - 1] / 2) ** t for t in range(1, periods)])
    bond_price = bond[date_column]
    spot_rate_func = lambda y: pv_coupons + 1000 / (1 + y / 2) ** periods - bond_price
    spot_rate = optimize.brentq(spot_rate_func, 0, 50)
    return spot_rate


def bootstrap_spot_curve(bonds, date_column):
    spot_curves = []
    prev_spots = []
    for _, bond in bonds.iterrows():
        coupon_rate = bond['Coupon Rate (%)']
        coupon = coupon_rate / 100 * 1000 / 2
        periods = int(bond['Years to Maturity'] * 2) + 1

        if not prev_spots or periods == 1:
            spot_rate = bond['YTM'] / 2
        else:
            spot_rate = get_spot_rate(bond, prev_spots, coupon, periods, date_column)
        spot_curves.append(spot_rate * 2)
        prev_spots.append(spot_rate)
    return spot_curves


def get_forward_rates(spot_rates):
    forward_rates = pd.DataFrame(index=range(2,6), columns=spot_rates.columns)
    for col in spot_rates.columns:
        for start_year in range(1,5):
            end_year = start_year + 1
            S1 = spot_rates.at[start_year, col]
            S2 = spot_rates.at[end_year, col]
            F = ((1+S2) ** end_year / (1+S1) ** start_year) ** (1 / (end_year - start_year)) - 1
            forward_rates.at[end_year, col] = F
    return forward_rates


if __name__=='__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    bonds = pd.read_excel('bond_data.xlsx')
    bonds['Issue Date'] = pd.to_datetime(bonds['Issue Date'])
    bonds['Maturity Date'] = pd.to_datetime(bonds['Maturity Date'])
    bonds['Years to Maturity'] = (bonds['Maturity Date'] - pd.Timestamp('today')).dt.days / 365
    price_columns = [col.strftime('%Y-%m-%d') if isinstance(col, pd.Timestamp) else col for col in bonds.columns[5:-1]]


    ytm_result = pd.DataFrame(index=bonds.index)
    spot_rate_result = pd.DataFrame(index=bonds.index)
    for date in price_columns:
        bonds['YTM'] = bonds.apply(lambda row: get_ytm(row, date), axis=1)
        ytm_result[date] = bonds['YTM']
        spot_rate_result[date] = bootstrap_spot_curve(bonds, date)
    forward_result = get_forward_rates(spot_rate_result)
    fig1, fig2, fig3 = go.Figure(), go.Figure(), go.Figure()

    offset = 0.01
    base_offset = -offset * (len(price_columns) / 2)

    for index, date in enumerate(price_columns):
        date_str = date if isinstance(date, str) else date.strftime('%Y-%m-%d')
        curr_offset = base_offset + (offset * index)
        fig1.add_trace(go.Scatter(
            x=bonds['Years to Maturity'] + curr_offset,
            y=ytm_result[date_str],
            mode='lines+markers',
            name=date_str)
        )
        fig2.add_trace(
            go.Scatter(
                x=bonds['Years to Maturity'] + curr_offset,
                y=spot_rate_result[date_str],
                mode='lines+markers',
                name=date_str)
        )
        fig3.add_trace(
            go.Scatter(
                x=forward_result.index + curr_offset,
                y=forward_result[date],
                mode='lines+markers',
                name=date_str)
        )

    # YTM
    fig1.update_layout(
        title='Yield to Maturity Curve over 5 Years',
        xaxis_title='Years',
        yaxis_title='Yield to Maturity (%)',
        legend_title='Date',
        width=800,
        height=500,
    )
    # Spot Rate
    fig2.update_layout(
        title='Spot Rate over 5 Years',
        xaxis_title='Years',
        yaxis_title='Spot Rate (%)',
        legend_title='Date',
        width=800,
        height=500,
    )
    # Forward Rate
    fig3.update_layout(
        title='1-Year Forward Curves Over Time',
        xaxis_title='Term (Years)',
        yaxis_title='Forward Rate (%)',
        legend_title='Date',
        width=800,
        height=500,
    )

    fig1.show()
    fig2.show()
    fig3.show()

    print("ytm_result", ytm_result)
    print("spot_Rate", spot_rate_result)
    print("forward", forward_result)

    # Covariance Matrix
    log_return_yields = np.log(ytm_result.T/ ytm_result.T.shift(1))
    log_return_yields = log_return_yields.dropna()
    cov_matrix_yields = log_return_yields.cov()
    print("Covariance Matrix for Yields: ")
    print(cov_matrix_yields)
    print('\n\n')

    print("forward_result", forward_result)
    print("forward_shift", forward_result.shift(1))
    forward_result = forward_result.astype(float)
    forward_result[forward_result <= 0] = np.nan
    log_return_forwards = np.log(forward_result / forward_result.shift(1))
    log_return_forwards = log_return_forwards.dropna()
    cov_matrix_forwards = log_return_forwards.cov()
    print("Covariance Matrix for Forwards: ")
    print(cov_matrix_forwards)

    # Eigenvalues
    eigenval_yield, eigenvec_yield = np.linalg.eig(cov_matrix_yields)
    eigenval_forward, eigenvec_forward = np.linalg.eig(cov_matrix_forwards)
    idx_yield = eigenval_yield.argsort()[::-1]
    idx_forward = eigenval_forward.argsort()[::-1]
    eigenval_yield_sorted = eigenval_yield[idx_yield]
    eigenval_forward_sorted = eigenval_forward[idx_forward]
    print("\nEigenvalues for Yield:", eigenval_yield_sorted)
    print("index yield: ", idx_yield)
    print("\nEigenvalues for Forward: ", eigenval_forward_sorted)
    print("index forward: ", idx_forward)

