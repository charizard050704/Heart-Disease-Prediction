import sqlite3
from datetime import datetime

# Connect once for the whole app
conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()

# ------------------ TABLES ------------------
# Users table: email is primary key; username is display name (not unique)
c.execute("""CREATE TABLE IF NOT EXISTS users (
    email TEXT PRIMARY KEY,
    username TEXT,
    password TEXT,
    role TEXT,         -- "doctor" or "patient"
    age INTEGER,
    gender TEXT
)""")

# Predictions: store doctor_email, patient_name, timestamp, probability, result, inputs
c.execute("""CREATE TABLE IF NOT EXISTS predictions (
    doctor_email TEXT,
    patient_name TEXT,
    timestamp TEXT,
    probability REAL,
    result TEXT,
    inputs TEXT
)""")

# Symptom checks (patients)
c.execute("""CREATE TABLE IF NOT EXISTS symptom_checks (
    email TEXT,
    timestamp TEXT,
    symptoms TEXT,
    recommendation TEXT
)""")

conn.commit()

# ------------------ User DB Helpers ------------------
def add_user(email, username, password_hash, role, age=None, gender=None):
    """
    Add a new user. Returns (True, None) on success; (False, error_message) on failure.
    """
    try:
        c.execute(
            "INSERT INTO users (email, username, password, role, age, gender) VALUES (?, ?, ?, ?, ?, ?)",
            (email, username, password_hash, role, age, gender)
        )
        conn.commit()
        return True, None
    except sqlite3.IntegrityError as ie:
        # Primary key conflict (email already exists)
        return False, "Email already registered."
    except Exception as e:
        return False, str(e)

def get_user_by_email(email):
    """
    Return user row by email in the order:
    (email, username, password, role, age, gender)
    or None if not found.
    """
    c.execute("SELECT email, username, password, role, age, gender FROM users WHERE email=?", (email,))
    return c.fetchone()

# ------------------ Prediction DB Helpers ------------------
def save_prediction(doctor_email, patient_name, probability, result, inputs):
    """
    Save a prediction made by a doctor for a patient.
    """
    c.execute("INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?)",
              (doctor_email, patient_name,
               datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               probability, result, str(inputs)))
    conn.commit()

def load_history(doctor_email):
    """
    Load prediction history for a given doctor (by email).
    Returns list of tuples: (doctor_email, patient_name, timestamp, probability, result, inputs)
    """
    c.execute("SELECT * FROM predictions WHERE doctor_email=? ORDER BY timestamp DESC", (doctor_email,))
    return c.fetchall()

# ------------------ Symptom DB Helpers (Patients) ------------------
def save_symptom_check(email, symptoms, recommendation):
    c.execute("INSERT INTO symptom_checks VALUES (?, ?, ?, ?)",
              (email, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               str(symptoms), recommendation))
    conn.commit()

def load_symptom_history(email):
    c.execute("SELECT * FROM symptom_checks WHERE email=? ORDER BY timestamp DESC", (email,))
    return c.fetchall()
