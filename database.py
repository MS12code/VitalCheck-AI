import sqlite3
from datetime import datetime
from models.schemas import TrendPoint

DB_PATH = "vitalcheck.db"

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS family_history (
                patient_name TEXT PRIMARY KEY,
                diabetes INTEGER DEFAULT 0,
                hypertension INTEGER DEFAULT 0,
                thyroid INTEGER DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_name TEXT,
                report_date TEXT,
                health_score INTEGER,
                risk_level TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS report_parameters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_id INTEGER,
                patient_name TEXT,
                report_date TEXT,
                parameter TEXT,
                value REAL,
                unit TEXT,
                normal_range TEXT,
                is_abnormal INTEGER,
                FOREIGN KEY(report_id) REFERENCES reports(id)
            )
        """)
        conn.commit()

def get_family_history(patient_name: str) -> dict:
    if not patient_name:
        return {"diabetes": False, "hypertension": False, "thyroid": False}
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT diabetes, hypertension, thyroid FROM family_history WHERE patient_name = ?",
            (patient_name,)
        ).fetchone()
        if row:
            return {
                "diabetes": bool(row["diabetes"]),
                "hypertension": bool(row["hypertension"]),
                "thyroid": bool(row["thyroid"])
            }
        return {"diabetes": False, "hypertension": False, "thyroid": False}

def save_family_history(patient_name: str, diabetes: bool, hypertension: bool, thyroid: bool):
    if not patient_name:
        return
    with get_db_connection() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO family_history (patient_name, diabetes, hypertension, thyroid)
            VALUES (?, ?, ?, ?)
        """, (patient_name, int(diabetes), int(hypertension), int(thyroid)))
        conn.commit()

def get_parameter_history(patient_name: str) -> list[TrendPoint]:
    if not patient_name:
        return []
    with get_db_connection() as conn:
        rows = conn.execute("""
            SELECT report_date, parameter, value, unit
            FROM report_parameters
            WHERE patient_name = ?
            ORDER BY report_date ASC
        """, (patient_name,)).fetchall()
        
        return [
            TrendPoint(
                report_date=row["report_date"],
                parameter=row["parameter"],
                value=row["value"],
                unit=row["unit"] or ""
            )
            for row in rows
        ]

def save_report(patient_name: str, health_score: int, risk_level: str, parameters: list[dict]):
    if not patient_name:
        return
    report_date = datetime.now().strftime("%Y-%m-%d")
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO reports (patient_name, report_date, health_score, risk_level)
            VALUES (?, ?, ?, ?)
        """, (patient_name, report_date, health_score, risk_level))
        report_id = cursor.lastrowid
        
        for p in parameters:
            cursor.execute("""
                INSERT INTO report_parameters (report_id, patient_name, report_date, parameter, value, unit, normal_range, is_abnormal)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                report_id,
                patient_name,
                report_date,
                p["name"],
                p["value"],
                p.get("unit", ""),
                p.get("normal_range", ""),
                int(p.get("is_abnormal", False))
            ))
        conn.commit()
