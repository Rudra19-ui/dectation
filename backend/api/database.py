"""
SQLite database module for patient and report storage
"""
import sqlite3
import os
import datetime
from contextlib import contextmanager

# Database path
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'breast_cancer.db')

def get_db_connection():
    """Create a database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Enable row factory for dict-like access
    return conn

@contextmanager
def get_db():
    """Context manager for database connections"""
    conn = get_db_connection()
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    """Initialize the database with required tables"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Create patients table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                date_of_birth TEXT,
                gender TEXT DEFAULT 'Female',
                phone TEXT,
                email TEXT,
                address TEXT,
                reason_for_exam TEXT,
                previous_mammogram TEXT DEFAULT 'No',
                family_history TEXT DEFAULT 'No',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(first_name, last_name, date_of_birth)
            )
        ''')
        
        # Create reports table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_id TEXT NOT NULL UNIQUE,
                patient_id INTEGER NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                probabilities_normal REAL,
                probabilities_benign REAL,
                probabilities_malignant REAL,
                analysis_date TEXT NOT NULL,
                pdf_path TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (patient_id) REFERENCES patients (id)
            )
        ''')
        
        conn.commit()
        print(f"Database initialized at {DB_PATH}")

def get_or_create_patient(patient_info):
    """Get existing patient or create new one"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        now = datetime.datetime.now().isoformat()
        
        # Try to find existing patient
        cursor.execute('''
            SELECT id FROM patients 
            WHERE first_name = ? AND last_name = ? AND date_of_birth = ?
        ''', (
            patient_info.get('firstName', ''),
            patient_info.get('lastName', ''),
            patient_info.get('dateOfBirth', '')
        ))
        
        result = cursor.fetchone()
        
        if result:
            # Update existing patient
            patient_id = result[0]
            cursor.execute('''
                UPDATE patients SET
                    phone = ?,
                    email = ?,
                    address = ?,
                    reason_for_exam = ?,
                    previous_mammogram = ?,
                    family_history = ?,
                    updated_at = ?
                WHERE id = ?
            ''', (
                patient_info.get('phone', ''),
                patient_info.get('email', ''),
                patient_info.get('address', ''),
                patient_info.get('reasonForExam', ''),
                patient_info.get('previousMammogram', 'No'),
                patient_info.get('familyHistory', 'No'),
                now,
                patient_id
            ))
            conn.commit()
            return patient_id
        else:
            # Create new patient
            cursor.execute('''
                INSERT INTO patients (
                    first_name, last_name, date_of_birth, gender,
                    phone, email, address, reason_for_exam,
                    previous_mammogram, family_history,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                patient_info.get('firstName', ''),
                patient_info.get('lastName', ''),
                patient_info.get('dateOfBirth', ''),
                patient_info.get('gender', 'Female'),
                patient_info.get('phone', ''),
                patient_info.get('email', ''),
                patient_info.get('address', ''),
                patient_info.get('reasonForExam', ''),
                patient_info.get('previousMammogram', 'No'),
                patient_info.get('familyHistory', 'No'),
                now,
                now
            ))
            conn.commit()
            return cursor.lastrowid

def create_report(patient_id, report_id, prediction, confidence, probabilities, pdf_path=None):
    """Create a new report"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        now = datetime.datetime.now().isoformat()
        
        cursor.execute('''
            INSERT INTO reports (
                report_id, patient_id, prediction, confidence,
                probabilities_normal, probabilities_benign, probabilities_malignant,
                analysis_date, pdf_path, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            report_id,
            patient_id,
            prediction,
            confidence,
            float(probabilities.get('Normal', 0)) if probabilities else 0,
            float(probabilities.get('Benign', 0)) if probabilities else 0,
            float(probabilities.get('Malignant', 0)) if probabilities else 0,
            now,
            pdf_path,
            now
        ))
        conn.commit()
        return cursor.lastrowid

def get_all_patients():
    """Get all patients with their reports"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                p.id, p.first_name, p.last_name, p.date_of_birth, p.gender,
                p.phone, p.email, p.address, p.reason_for_exam,
                p.previous_mammogram, p.family_history, p.created_at
            FROM patients p
            ORDER BY p.created_at DESC
        ''')
        
        patients = []
        for row in cursor.fetchall():
            patient = dict(row)
            
            # Get reports for this patient
            cursor.execute('''
                SELECT report_id, prediction, confidence, 
                       probabilities_normal, probabilities_benign, probabilities_malignant,
                       analysis_date, pdf_path, created_at
                FROM reports
                WHERE patient_id = ?
                ORDER BY analysis_date DESC
            ''', (patient['id'],))
            
            patient['reports'] = [dict(r) for r in cursor.fetchall()]
            patients.append(patient)
        
        return patients

def get_patient_by_id(patient_id):
    """Get a single patient by ID"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                id, first_name, last_name, date_of_birth, gender,
                phone, email, address, reason_for_exam,
                previous_mammogram, family_history, created_at
            FROM patients
            WHERE id = ?
        ''', (patient_id,))
        
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None

def get_reports_by_patient_id(patient_id):
    """Get all reports for a patient"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                id, report_id, prediction, confidence,
                probabilities_normal, probabilities_benign, probabilities_malignant,
                analysis_date, pdf_path, created_at
            FROM reports
            WHERE patient_id = ?
            ORDER BY analysis_date DESC
        ''', (patient_id,))
        
        return [dict(row) for row in cursor.fetchall()]

def delete_patient(patient_id):
    """Delete a patient and their reports"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Delete reports first
        cursor.execute('DELETE FROM reports WHERE patient_id = ?', (patient_id,))
        # Delete patient
        cursor.execute('DELETE FROM patients WHERE id = ?', (patient_id,))
        conn.commit()

# Initialize database on module load
if __name__ == '__main__':
    init_db()
    print("Database initialized successfully!")
