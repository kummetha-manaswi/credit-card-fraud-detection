"""
SQL Fraud Analytics Module
Executes analytical SQL queries against the SQLite database of transactions.
Supports pre-compiled KPI queries and custom user queries.
"""

import os
import sqlite3
from typing import Dict, Optional, Tuple, Union
import pandas as pd


class FraudSQLAnalytics:
    """
    Interface for SQLite fraud intelligence queries.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            # Priority: database/fraud_detection.db -> fraud_detection.db in root
            candidates = [
                os.path.join(base_dir, "database", "fraud_detection.db"),
                os.path.join(base_dir, "fraud_detection.db"),
                os.path.join(r"C:\Users\manas", "fraud_detection.db")
            ]
            self.db_path = None
            for p in candidates:
                if os.path.exists(p):
                    self.db_path = p
                    break

            # If no sqlite database file exists, initialize from sample_transactions.csv
            if self.db_path is None:
                self.db_path = os.path.join(base_dir, "database", "fraud_detection.db")
                self._init_from_sample(base_dir)
        else:
            self.db_path = db_path

    def _init_from_sample(self, base_dir: str) -> None:
        """Create database from sample_transactions.csv if no DB exists."""
        sample_path = os.path.join(base_dir, "data", "sample_transactions.csv")
        if os.path.exists(sample_path):
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            df = pd.read_csv(sample_path)
            with sqlite3.connect(self.db_path) as conn:
                df.to_sql("transactions", conn, if_exists="replace", index=False)

    def get_connection(self) -> sqlite3.Connection:
        """Return a connection to the SQLite database."""
        return sqlite3.connect(self.db_path)

    def get_class_summary(self) -> pd.DataFrame:
        """Query 1: Transaction count and percentage by class."""
        query = """
        SELECT
            Class,
            CASE WHEN Class = 1 THEN 'Fraud' ELSE 'Legitimate' END AS Status,
            COUNT(*) AS Transactions,
            ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM transactions), 4) AS Percentage
        FROM transactions
        GROUP BY Class
        ORDER BY Class;
        """
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn)

    def get_amount_summary(self) -> pd.DataFrame:
        """Query 2: Transaction amount statistics by class."""
        query = """
        SELECT
            CASE WHEN Class = 1 THEN 'Fraud' ELSE 'Legitimate' END AS Transaction_Type,
            COUNT(*) AS Transactions,
            ROUND(SUM(Amount), 2) AS Total_Amount,
            ROUND(AVG(Amount), 2) AS Average_Amount,
            ROUND(MIN(Amount), 2) AS Minimum_Amount,
            ROUND(MAX(Amount), 2) AS Maximum_Amount
        FROM transactions
        GROUP BY Class;
        """
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn)

    def get_top_fraud_transactions(self, limit: int = 10) -> pd.DataFrame:
        """Query 3: Top highest-value fraudulent transactions."""
        query = f"""
        SELECT
            Time,
            ROUND(Time / 3600.0, 2) AS Elapsed_Hours,
            Amount,
            Class
        FROM transactions
        WHERE Class = 1
        ORDER BY Amount DESC
        LIMIT {int(limit)};
        """
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn)

    def get_fraud_by_amount_range(self) -> pd.DataFrame:
        """Query 4: Fraud rate segmented by transaction amount range."""
        query = """
        SELECT
            CASE
                WHEN Amount <= 1 THEN '$0–1'
                WHEN Amount <= 10 THEN '$1–10'
                WHEN Amount <= 50 THEN '$10–50'
                WHEN Amount <= 100 THEN '$50–100'
                WHEN Amount <= 500 THEN '$100–500'
                WHEN Amount <= 1000 THEN '$500–1K'
                ELSE '$1K+'
            END AS Amount_Range,
            COUNT(*) AS Transactions,
            SUM(Class) AS Fraud_Count,
            ROUND(100.0 * SUM(Class) / COUNT(*), 4) AS Fraud_Rate_Pct
        FROM transactions
        GROUP BY Amount_Range
        ORDER BY
            CASE Amount_Range
                WHEN '$0–1' THEN 1
                WHEN '$1–10' THEN 2
                WHEN '$10–50' THEN 3
                WHEN '$50–100' THEN 4
                WHEN '$100–500' THEN 5
                WHEN '$500–1K' THEN 6
                ELSE 7
            END;
        """
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn)

    def get_fraud_by_hour(self) -> pd.DataFrame:
        """Query 5: Hourly fraud patterns over the 48-hour dataset window."""
        query = """
        SELECT
            CAST((Time / 3600) % 24 AS INT) AS Hour,
            COUNT(*) AS Total_Transactions,
            SUM(Class) AS Fraud_Transactions,
            ROUND(100.0 * SUM(Class) / COUNT(*), 4) AS Fraud_Rate_Pct
        FROM transactions
        GROUP BY Hour
        ORDER BY Hour;
        """
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn)

    def execute_custom_query(self, query: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Execute a custom read-only user query for the interactive SQL explorer.
        Returns (dataframe, error):
            - On success: (pd.DataFrame, None)
            - On error:   (None, error_message_string)
        """
        clean_q = query.strip()
        # Security guard: only allow SELECT, WITH, or EXPLAIN statements
        if not (clean_q.upper().startswith("SELECT") or clean_q.upper().startswith("WITH") or clean_q.upper().startswith("EXPLAIN")):
            return None, "Only read-only SELECT or WITH queries are permitted."

        try:
            with self.get_connection() as conn:
                df = pd.read_sql_query(clean_q, conn)
                return df, None
        except Exception as e:
            return None, str(e)


_sql_analytics_instance: Optional[FraudSQLAnalytics] = None


def get_sql_analytics(db_path: Optional[str] = None) -> FraudSQLAnalytics:
    """Return singleton or initialized instance of FraudSQLAnalytics."""
    global _sql_analytics_instance
    if _sql_analytics_instance is None or db_path is not None:
        _sql_analytics_instance = FraudSQLAnalytics(db_path=db_path)
    return _sql_analytics_instance
