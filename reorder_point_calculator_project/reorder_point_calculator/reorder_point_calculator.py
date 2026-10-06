"""
Reorder Point Calculator - base project
---------------------------------------
For every product it computes:
    avg daily demand (d), demand std-dev (sigma_d)
    Safety Stock  = z * sqrt(L * sigma_d^2 + d^2 * sigma_L^2)
    Reorder Point = d * L + Safety Stock
    EOQ           = sqrt(2 * annual_demand * order_cost / holding_cost)
and flags products whose current stock has fallen to or below the reorder point.

Usage
-----
  python reorder_point_calculator.py                     # runs on sample data (auto-generated)
  python reorder_point_calculator.py --products products.csv --sales sales_history.csv
  python reorder_point_calculator.py --quick --demand 40 --std 12 --lead 7 --service 0.95

Input files
-----------
  sales_history.csv : date, sku, units_sold
  products.csv      : sku, name, lead_time_days, lead_time_std_days, service_level,
                      order_cost, holding_cost_per_unit_year, on_hand
"""
import argparse
import math
import os
from statistics import NormalDist

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
# CORE FORMULAS
# ----------------------------------------------------------------------------
def z_score(service_level: float) -> float:
    """Z-value for a cycle service level, e.g. 0.95 -> 1.645."""
    return NormalDist().inv_cdf(service_level)


def safety_stock(avg_demand, std_demand, lead_time, lead_time_std, service_level):
    z = z_score(service_level)
    variance = lead_time * std_demand ** 2 + (avg_demand ** 2) * lead_time_std ** 2
    return z * math.sqrt(variance)


def reorder_point(avg_demand, lead_time, ss):
    return avg_demand * lead_time + ss


def eoq(annual_demand, order_cost, holding_cost):
    if holding_cost <= 0 or annual_demand <= 0:
        return 0
    return math.sqrt(2 * annual_demand * order_cost / holding_cost)


def stock_status(on_hand, rop):
    if on_hand <= rop:
        return "REORDER NOW"
    if on_hand <= rop * 1.2:
        return "LOW - REORDER SOON"
    return "OK"


# ----------------------------------------------------------------------------
# SAMPLE DATA (used when no files are given)
# ----------------------------------------------------------------------------
def make_sample_data(folder="."):
    rng = np.random.default_rng(7)
    products = pd.DataFrame(
        {
            "sku": ["P001", "P002", "P003", "P004", "P005", "P006"],
            "name": ["Rice 5kg", "Cooking Oil 1L", "Shampoo 200ml", "LED Bulb 9W", "Notebook A4", "Phone Charger"],
            "lead_time_days": [5, 7, 4, 10, 3, 14],
            "lead_time_std_days": [1, 2, 1, 3, 0.5, 4],
            "service_level": [0.97, 0.95, 0.90, 0.95, 0.90, 0.99],
            "order_cost": [50, 45, 30, 60, 25, 80],
            "holding_cost_per_unit_year": [4.0, 3.0, 2.5, 1.8, 1.2, 6.0],
            "on_hand": [150, 250, 400, 95, 300, 230],
        }
    )
    mean_demand = {"P001": 28, "P002": 22, "P003": 35, "P004": 14, "P005": 40, "P006": 9}
    dates = pd.date_range("2026-01-01", periods=270, freq="D")
    rows = []
    for sku, mu in mean_demand.items():
        units = np.clip(rng.normal(mu, mu * 0.3, len(dates)), 0, None).round().astype(int)
        rows += list(zip(dates, [sku] * len(dates), units))
    sales = pd.DataFrame(rows, columns=["date", "sku", "units_sold"])

    products.to_csv(os.path.join(folder, "products.csv"), index=False)
    sales.to_csv(os.path.join(folder, "sales_history.csv"), index=False)
    return products, sales


# ----------------------------------------------------------------------------
# MAIN CALCULATION
# ----------------------------------------------------------------------------
def calculate(products: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    demand = sales.groupby("sku")["units_sold"].agg(avg_daily_demand="mean", std_daily_demand="std").reset_index()
    df = products.merge(demand, on="sku", how="left").fillna({"std_daily_demand": 0, "avg_daily_demand": 0})

    results = []
    for _, r in df.iterrows():
        ss = safety_stock(r.avg_daily_demand, r.std_daily_demand, r.lead_time_days, r.lead_time_std_days, r.service_level)
        rop = reorder_point(r.avg_daily_demand, r.lead_time_days, ss)
        q = eoq(r.avg_daily_demand * 365, r.order_cost, r.holding_cost_per_unit_year)
        results.append(
            {
                "sku": r.sku,
                "name": r["name"],
                "avg_daily_demand": round(r.avg_daily_demand, 1),
                "std_daily_demand": round(r.std_daily_demand, 1),
                "lead_time_days": r.lead_time_days,
                "service_level": r.service_level,
                "safety_stock": math.ceil(ss),
                "reorder_point": math.ceil(rop),
                "eoq": math.ceil(q),
                "on_hand": int(r.on_hand),
                "status": stock_status(r.on_hand, rop),
                "days_of_stock_left": round(r.on_hand / r.avg_daily_demand, 1) if r.avg_daily_demand else None,
            }
        )
    return pd.DataFrame(results)


def plot(report: pd.DataFrame, path="reorder_chart.png"):
    colors = {"REORDER NOW": "#d62728", "LOW - REORDER SOON": "#ff9f1c", "OK": "#2a9d8f"}
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(report))
    ax.bar(x, report["on_hand"], color=[colors[s] for s in report["status"]], width=0.55, label="On hand")
    ax.scatter(x, report["reorder_point"], color="black", marker="_", s=900, linewidths=3, zorder=3, label="Reorder point")
    ax.set_xticks(x)
    ax.set_xticklabels(report["name"], rotation=20, ha="right")
    ax.set_ylabel("Units")
    ax.set_title("Current Stock vs Reorder Point")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def quick_mode(args):
    ss = safety_stock(args.demand, args.std, args.lead, args.lead_std, args.service)
    rop = reorder_point(args.demand, args.lead, ss)
    print(f"Z-score        : {z_score(args.service):.3f}")
    print(f"Safety stock   : {math.ceil(ss)} units")
    print(f"Reorder point  : {math.ceil(rop)} units")


def main():
    p = argparse.ArgumentParser(description="Reorder Point Calculator")
    p.add_argument("--products", help="products CSV")
    p.add_argument("--sales", help="sales history CSV")
    p.add_argument("--quick", action="store_true", help="single-product calculation")
    p.add_argument("--demand", type=float, help="avg daily demand (quick mode)")
    p.add_argument("--std", type=float, default=0, help="std dev of daily demand (quick mode)")
    p.add_argument("--lead", type=float, help="lead time in days (quick mode)")
    p.add_argument("--lead-std", type=float, default=0, help="std dev of lead time in days (quick mode)")
    p.add_argument("--service", type=float, default=0.95, help="service level, e.g. 0.95")
    args = p.parse_args()

    if args.quick:
        if args.demand is None or args.lead is None:
            p.error("--quick needs --demand and --lead")
        return quick_mode(args)

    if args.products and args.sales:
        products, sales = pd.read_csv(args.products), pd.read_csv(args.sales)
    else:
        print("No input files given - generating sample data (products.csv, sales_history.csv)\n")
        products, sales = make_sample_data()

    report = calculate(products, sales)
    report.to_csv("reorder_report.csv", index=False)
    chart = plot(report)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    print("=== REORDER REPORT ===")
    print(report.drop(columns=["service_level"]).to_string(index=False))

    urgent = report[report["status"] == "REORDER NOW"]
    print("\n=== ACTION LIST ===")
    if urgent.empty:
        print("Nothing needs reordering right now.")
    for _, r in urgent.iterrows():
        print(f"- Order {r.eoq} units of {r['name']} ({r.sku}); stock {r.on_hand} <= reorder point {r.reorder_point}")

    print(f"\nSaved: reorder_report.csv, {chart}")


if __name__ == "__main__":
    main()
