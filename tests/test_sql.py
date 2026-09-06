"""
Tests for SQL Analytics engine and queries.
"""

import pytest
import pandas as pd
from src.sql_analytics import get_sql_analytics


def test_sql_analytics_connection():
    sql = get_sql_analytics()
    conn = sql.get_connection()
    assert conn is not None
    conn.close()


def test_sql_class_summary():
    sql = get_sql_analytics()
    df = sql.get_class_summary()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "Class" in df.columns
    assert "Transactions" in df.columns
    assert "Percentage" in df.columns


def test_sql_amount_summary():
    sql = get_sql_analytics()
    df = sql.get_amount_summary()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "Average_Amount" in df.columns


def test_sql_top_fraud():
    sql = get_sql_analytics()
    df = sql.get_top_fraud_transactions(limit=5)
    assert isinstance(df, pd.DataFrame)
    assert len(df) <= 5
    if len(df) > 0:
        assert "Amount" in df.columns
        assert (df["Class"] == 1).all()


def test_sql_amount_range():
    sql = get_sql_analytics()
    df = sql.get_fraud_by_amount_range()
    assert isinstance(df, pd.DataFrame)
    assert "Amount_Range" in df.columns
    assert "Fraud_Rate_Pct" in df.columns


def test_sql_custom_query_security():
    sql = get_sql_analytics()
    # Read-only SELECT should succeed
    df, err = sql.execute_custom_query("SELECT COUNT(*) AS total FROM transactions")
    assert err is None
    assert isinstance(df, pd.DataFrame)
    assert df["total"].iloc[0] > 0

    # Destructive commands should be rejected
    df2, err2 = sql.execute_custom_query("DROP TABLE transactions")
    assert df2 is None
    assert err2 is not None
    assert "Only read-only SELECT" in err2

    # Invalid SQL syntax should return error string without crashing
    df3, err3 = sql.execute_custom_query("SELECT non_existent_col FROM transactions LIMIT 5;")
    assert df3 is None
    assert err3 is not None
    assert "no such column" in err3
