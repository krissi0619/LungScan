from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session
import os
import uuid
from datetime import datetime, timedelta
import json
from typing import List, Optional

from database import get_db, User, PatientReport, Patient, AnalysisSession
from auth import (
    get_password_hash, verify_password, create_access_token, 
    get_current_user, get_current_active_user, create_user, login_user,
    SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
)
# Import your model and report generator (adjust paths as needed)
# from models.model import LungDiseaseModel
# from utils.report_generator import ReportGenerator, generate_report_id

app = FastAPI(
    title="LungScan AI API",
    description="AI-powered lung disease detection and analysis platform",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components (commented out for now - add your actual implementations)
# MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/lung_disease_model.h5")
# model = LungDiseaseModel(MODEL_PATH)
# report_generator = ReportGenerator()
security = HTTPBearer()

# Ensure directories exist
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)

# Helper functions
def generate_report_id():
    """Generate unique report ID"""
    return f"RPT-{uuid.uuid4().hex[:8].upper()}"

def save_uploaded_file(file: UploadFile) -> str:
    """Save uploaded file and return file path"""
    file_extension = os.path.splitext(file.filename)[1]
    filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    with open(file_path, "wb") as buffer:
        content = file.file.read()
        buffer.write(content)
    
    return filename, file_path

# Mock prediction function - replace with your actual model
def mock_predict(image_data) -> dict:
    """Mock prediction function - replace with actual model inference"""
    # This is a placeholder - implement your actual model prediction here
    import random
    conditions = ["Normal", "Pneumonia", "COVID-19", "Tuberculosis", "Lung Cancer"]
    probabilities = [random.random() for _ in conditions]
    total = sum(probabilities)
    probabilities = [p/total for p in probabilities]
    
    main_idx = probabilities.index(max(probabilities))
    
    return {
        'class': conditions[main_idx],
        'confidence': round(probabilities[main_idx] * 100, 2),
        'all_predictions': {
            condition: round(prob * 100, 2) 
            for condition, prob in zip(conditions, probabilities)
        },
        'risk_level': 'High' if conditions[main_idx] != 'Normal' else 'Low',
        'recommendations': [
            "Consult with a pulmonologist",
            "Follow up with additional tests if symptoms persist",
            "Maintain regular health checkups"
        ]
    }

# Updated Authentication endpoints
@app.post("/api/register", status_code=status.HTTP_201_CREATED)
async def register(
    full_name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    """Register a new user"""
    try:
        # Create user using your auth function
        user = create_user(db, full_name, email, password)
        
        # Return success response
        return {
            "message": "User created successfully", 
            "user_id": user.id,
            "email": user.email
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )

@app.post("/api/login")
async def login(
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    """Login user and return access token"""
    try:
        # Authenticate user
        user = authenticate_user(db, email, password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.email}, 
            expires_delta=access_token_expires
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "id": user.id,
                "full_name": user.full_name,
                "email": user.email
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )

# Helper function for authentication
def authenticate_user(db: Session, email: str, password: str):
    """Authenticate user with email and password"""
    try:
        user = db.query(User).filter(User.email == email).first()
        if not user:
            return False
        
        # Make sure you're using the correct password field name
        if not verify_password(password, user.hashed_password):
            return False
            
        return user
    except Exception as e:
        print(f"Authentication error: {e}")
        return False

# Protected endpoints
@app.get("/api/user/profile")
async def get_profile(current_user: User = Depends(get_current_active_user)):
    """Get current user profile"""
    return {
        "id": current_user.id,
        "full_name": current_user.full_name,
        "email": current_user.email,
        "role": current_user.role,
        "is_active": current_user.is_active,
        "created_at": current_user.created_at.isoformat() if current_user.created_at else None
    }

@app.post("/api/change-password")
async def change_password(
    current_password: str = Form(...),
    new_password: str = Form(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Change user password"""
    from auth import change_password as change_user_password
    
    try:
        change_user_password(db, current_user, current_password, new_password)
        return {"message": "Password updated successfully"}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password change failed"
        )

# Patient management
@app.post("/api/patients")
async def create_patient(
    name: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    contact_info: Optional[str] = Form(None),
    medical_history: Optional[str] = Form(None),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Create a new patient record"""
    try:
        patient = Patient(
            user_id=current_user.id,
            name=name,
            age=age,
            gender=gender,
            contact_info=contact_info,
            medical_history=medical_history,
            patient_id=f"PT-{uuid.uuid4().hex[:6].upper()}",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        db.add(patient)
        db.commit()
        db.refresh(patient)
        
        return {
            "message": "Patient created successfully",
            "patient_id": patient.patient_id,
            "patient": {
                "id": patient.id,
                "name": patient.name,
                "age": patient.age,
                "gender": patient.gender,
                "patient_id": patient.patient_id
            }
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create patient"
        )

@app.get("/api/patients")
async def get_patients(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get all patients for current user"""
    patients = db.query(Patient).filter(Patient.user_id == current_user.id).all()
    
    return [
        {
            "id": patient.id,
            "name": patient.name,
            "age": patient.age,
            "gender": patient.gender,
            "patient_id": patient.patient_id,
            "contact_info": patient.contact_info,
            "created_at": patient.created_at.isoformat() if patient.created_at else None
        }
        for patient in patients
    ]

# Analysis endpoints
@app.post("/api/analyze")
async def analyze_xray(
    patient_name: str = Form(...),
    patient_age: int = Form(...),
    patient_gender: str = Form(...),
    scan_type: str = Form("CT"),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Analyze lung CT scan or X-ray image"""
    # Validate file
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image"
        )
    
    try:
        # Save uploaded file
        filename, file_path = save_uploaded_file(file)
        
        # Read file for prediction
        with open(file_path, "rb") as f:
            image_data = f.read()
        
        # Make prediction (using mock for now - replace with actual model)
        result = mock_predict(image_data)
        
        if 'error' in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result['error']
            )
        
        # Generate report ID
        report_id = generate_report_id()
        
        # Save to database
        report = PatientReport(
            user_id=current_user.id,
            patient_name=patient_name,
            patient_age=patient_age,
            patient_gender=patient_gender,
            image_filename=filename,
            image_filepath=file_path,
            original_filename=file.filename,
            file_size=len(image_data),
            prediction_result=result['class'],
            confidence=result['confidence'],
            all_predictions=json.dumps(result['all_predictions']),
            risk_level=result['risk_level'],
            recommendations='\n'.join(result['recommendations']),
            scan_type=scan_type,
            scan_date=datetime.utcnow(),
            report_id=report_id,
            report_date=datetime.utcnow()
        )
        
        db.add(report)
        db.commit()
        db.refresh(report)
        
        # Add report info to result
        result['report_id'] = report_id
        result['patient_info'] = {
            'name': patient_name,
            'age': patient_age,
            'gender': patient_gender
        }
        result['file_info'] = {
            'original_filename': file.filename,
            'saved_filename': filename
        }
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )

@app.get("/api/reports")
async def get_user_reports(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get all reports for current user"""
    reports = db.query(PatientReport).filter(
        PatientReport.user_id == current_user.id
    ).order_by(PatientReport.report_date.desc()).all()
    
    return [
        {
            "id": report.id,
            "report_id": report.report_id,
            "patient_name": report.patient_name,
            "patient_age": report.patient_age,
            "patient_gender": report.patient_gender,
            "prediction_result": report.prediction_result,
            "confidence": round(report.confidence, 2),
            "risk_level": report.risk_level,
            "scan_type": report.scan_type,
            "report_date": report.report_date.isoformat() if report.report_date else None,
            "image_filename": report.image_filename
        }
        for report in reports
    ]

@app.get("/api/reports/{report_id}")
async def get_report_details(
    report_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get detailed information for a specific report"""
    report = db.query(PatientReport).filter(
        PatientReport.report_id == report_id,
        PatientReport.user_id == current_user.id
    ).first()
    
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found"
        )
    
    # Parse all predictions JSON
    try:
        all_predictions = json.loads(report.all_predictions) if report.all_predictions else {}
    except:
        all_predictions = {}
    
    return {
        "id": report.id,
        "report_id": report.report_id,
        "patient_name": report.patient_name,
        "patient_age": report.patient_age,
        "patient_gender": report.patient_gender,
        "prediction_result": report.prediction_result,
        "confidence": round(report.confidence, 2),
        "risk_level": report.risk_level,
        "all_predictions": all_predictions,
        "recommendations": report.recommendations.split('\n') if report.recommendations else [],
        "scan_type": report.scan_type,
        "report_date": report.report_date.isoformat() if report.report_date else None,
        "scan_date": report.scan_date.isoformat() if report.scan_date else None
    }

@app.get("/api/reports/{report_id}/pdf")
async def download_pdf_report(
    report_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Download PDF report"""
    report = db.query(PatientReport).filter(
        PatientReport.report_id == report_id,
        PatientReport.user_id == current_user.id
    ).first()
    
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found"
        )
    
    # Mock PDF generation - implement your actual PDF generation
    pdf_filename = f"{report_id}.pdf"
    pdf_path = os.path.join(REPORT_DIR, pdf_filename)
    
    # Create a simple text file as placeholder for PDF
    with open(pdf_path, 'w') as f:
        f.write(f"LungScan AI Report - {report_id}\n")
        f.write(f"Patient: {report.patient_name}\n")
        f.write(f"Result: {report.prediction_result}\n")
        f.write(f"Confidence: {report.confidence}%\n")
    
    return FileResponse(
        pdf_path,
        media_type='application/pdf',
        filename=f"LungScan_Report_{report_id}.pdf"
    )

# Debug endpoints to check database
@app.get("/api/debug/users")
async def debug_users(db: Session = Depends(get_db)):
    """Debug endpoint to check users in database"""
    users = db.query(User).all()
    return {
        "total_users": len(users),
        "users": [
            {
                "id": user.id,
                "email": user.email,
                "full_name": user.full_name,
                "has_password": bool(user.hashed_password),
                "created_at": user.created_at.isoformat() if user.created_at else None
            }
            for user in users
        ]
    }

@app.get("/api/debug/user/{email}")
async def debug_user(email: str, db: Session = Depends(get_db)):
    """Debug endpoint to check specific user"""
    user = db.query(User).filter(User.email == email).first()
    if user:
        return {
            "exists": True,
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "hashed_password_length": len(user.hashed_password) if user.hashed_password else 0,
            "created_at": user.created_at.isoformat() if user.created_at else None
        }
    return {"exists": False}

# Health and status endpoints
@app.get("/api/health")
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint"""
    try:
        # Test database connection
        db.execute("SELECT 1")
        return {
            "status": "healthy",
            "service": "lungscan-ai",
            "database": "connected",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy - database connection failed"
        )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "LungScan AI API",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)