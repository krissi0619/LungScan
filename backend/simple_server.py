from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from datetime import datetime, timedelta
import jwt
import os
import uuid
import json
import time
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

app = Flask(__name__)
CORS(app)

app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['REPORTS_FOLDER'] = 'reports'

# Create folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['REPORTS_FOLDER'], exist_ok=True)

# Default users
users_db = [
    {
        'id': 1,
        'full_name': 'Ash Shangak',
        'email': 'ashshangak570@gmail.com',
        'password': 'password123',
        'created_at': datetime.now().isoformat()
    },
    {
        'id': 2, 
        'full_name': 'Demo User',
        'email': 'demo@lungscan.com',
        'password': 'demo123',
        'created_at': datetime.now().isoformat()
    }
]

# Sample scans data
scans_db = [
    {
        'id': 'PT-0012',
        'patient_id': 'PT-0012',
        'patient_name': 'John Smith',
        'scan_date': '2023-11-15',
        'condition': 'Pneumonia',
        'confidence': 92.5,
        'status': 'Completed',
        'image_url': '/api/scans/PT-0012/image',
        'report_url': '/api/reports/PT-0012',
        'created_at': datetime.now().isoformat()
    },
    {
        'id': 'PT-0011',
        'patient_id': 'PT-0011',
        'patient_name': 'Sarah Johnson',
        'scan_date': '2023-11-14',
        'condition': 'Normal',
        'confidence': 98.2,
        'status': 'Completed',
        'image_url': '/api/scans/PT-0011/image',
        'report_url': '/api/reports/PT-0011',
        'created_at': datetime.now().isoformat()
    },
    {
        'id': 'PT-0010',
        'patient_id': 'PT-0010',
        'patient_name': 'Mike Davis',
        'scan_date': '2023-11-14',
        'condition': 'COVID-19',
        'confidence': 87.3,
        'status': 'Completed',
        'image_url': '/api/scans/PT-0010/image',
        'report_url': '/api/reports/PT-0010',
        'created_at': datetime.now().isoformat()
    },
    {
        'id': 'PT-0009',
        'patient_id': 'PT-0009',
        'patient_name': 'Emma Wilson',
        'scan_date': '2023-11-13',
        'condition': 'Tuberculosis',
        'confidence': 78.9,
        'status': 'Processing',
        'image_url': '/api/scans/PT-0009/image',
        'report_url': '/api/reports/PT-0009',
        'created_at': datetime.now().isoformat()
    }
]

# Authentication middleware
def get_current_user():
    token = request.headers.get('Authorization')
    if not token or not token.startswith('Bearer '):
        return None
    
    try:
        token = token.split(' ')[1]
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        user = next((u for u in users_db if u['id'] == payload['user_id']), None)
        return user
    except:
        return None

# Health check
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Server is running"})

# Authentication endpoints
@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        
        if any(user['email'] == data['email'] for user in users_db):
            return jsonify({"error": "Email already registered"}), 400
        
        new_user = {
            'id': len(users_db) + 1,
            'full_name': data['full_name'],
            'email': data['email'],
            'password': data['password'],
            'created_at': datetime.now().isoformat()
        }
        
        users_db.append(new_user)
        
        token = jwt.encode({
            'user_id': new_user['id'],
            'email': new_user['email'],
            'exp': datetime.utcnow() + timedelta(hours=24)
        }, app.config['SECRET_KEY'], algorithm='HS256')
        
        return jsonify({
            "message": "User created successfully",
            "access_token": token,
            "user": {
                "id": new_user['id'],
                "email": new_user['email'],
                "full_name": new_user['full_name']
            }
        }), 201
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        user = next((u for u in users_db if u['email'] == email and u['password'] == password), None)
        
        if not user:
            return jsonify({"error": "Invalid email or password"}), 401
        
        token = jwt.encode({
            'user_id': user['id'],
            'email': user['email'],
            'exp': datetime.utcnow() + timedelta(hours=24)
        }, app.config['SECRET_KEY'], algorithm='HS256')
        
        return jsonify({
            "access_token": token,
            "user": {
                "id": user['id'],
                "email": user['email'],
                "full_name": user['full_name']
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Scan management endpoints
@app.route('/api/scans', methods=['GET'])
def get_scans():
    user = get_current_user()
    if not user:
        return jsonify({"error": "Unauthorized"}), 401
    
    return jsonify({
        "scans": scans_db,
        "total": len(scans_db)
    })

@app.route('/api/scans/upload', methods=['POST'])
def upload_scan():
    user = get_current_user()
    if not user:
        return jsonify({"error": "Unauthorized"}), 401
    
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Generate unique ID for the scan
        scan_id = f"PT-{str(len(scans_db) + 1).zfill(4)}"
        
        # Save the file
        filename = f"{scan_id}_{file.filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Create new scan entry
        new_scan = {
            'id': scan_id,
            'patient_id': scan_id,
            'patient_name': request.form.get('patient_name', 'Unknown Patient'),
            'scan_date': datetime.now().strftime('%Y-%m-%d'),
            'condition': 'Analyzing...',
            'confidence': 0,
            'status': 'Processing',
            'image_url': f'/api/scans/{scan_id}/image',
            'report_url': f'/api/reports/{scan_id}',
            'created_at': datetime.now().isoformat()
        }
        
        scans_db.insert(0, new_scan)  # Add to beginning of list
        
        # Simulate AI analysis (in real app, this would call your ML model)
        time.sleep(2)
        
        # Update with "analysis results"
        conditions = ['Normal', 'Pneumonia', 'COVID-19', 'Tuberculosis']
        new_scan['condition'] = conditions[len(scans_db) % len(conditions)]
        new_scan['confidence'] = round(80 + (len(scans_db) * 3.7), 1)
        new_scan['status'] = 'Completed'
        
        return jsonify({
            "message": "Scan uploaded successfully",
            "scan": new_scan
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/scans/<scan_id>/image', methods=['GET'])
def get_scan_image(scan_id):
    user = get_current_user()
    if not user:
        return jsonify({"error": "Unauthorized"}), 401
    
    # Return a placeholder image or actual image
    # For now, return a success message
    return jsonify({
        "message": f"Image for scan {scan_id}",
        "image_url": f"/api/scans/{scan_id}/image"
    })

@app.route('/api/reports/<scan_id>', methods=['GET'])
def get_report(scan_id):
    user = get_current_user()
    if not user:
        return jsonify({"error": "Unauthorized"}), 401
    
    scan = next((s for s in scans_db if s['id'] == scan_id), None)
    if not scan:
        return jsonify({"error": "Scan not found"}), 404
    
    # Generate report data
    report = {
        'scan_id': scan_id,
        'patient_name': scan['patient_name'],
        'scan_date': scan['scan_date'],
        'condition': scan['condition'],
        'confidence': scan['confidence'],
        'status': scan['status'],
        'findings': f"The AI analysis detected {scan['condition']} with {scan['confidence']}% confidence.",
        'recommendations': [
            "Consult with a pulmonologist for further evaluation",
            "Follow up in 2 weeks for repeat imaging if symptoms persist",
            "Consider additional diagnostic tests if clinically indicated"
        ],
        'generated_at': datetime.now().isoformat()
    }
    
    return jsonify(report)

# Add this new route for PDF generation
@app.route('/api/reports/<scan_id>/pdf', methods=['GET'])
def generate_pdf_report(scan_id):
    user = get_current_user()
    if not user:
        return jsonify({"error": "Unauthorized"}), 401
    
    scan = next((s for s in scans_db if s['id'] == scan_id), None)
    if not scan:
        return jsonify({"error": "Scan not found"}), 404
    
    try:
        # Create PDF in memory
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        
        # Create the styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
            textColor=colors.HexColor('#2c3e50')
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=12,
            spaceAfter=12,
            textColor=colors.HexColor('#4facfe')
        )
        
        normal_style = styles["BodyText"]
        
        # Build the story (content)
        story = []
        
        # Title
        story.append(Paragraph("LUNGSCAN AI - MEDICAL REPORT", title_style))
        story.append(Spacer(1, 20))
        
        # Patient Information
        story.append(Paragraph("PATIENT INFORMATION", heading_style))
        patient_data = [
            ["Patient Name:", scan['patient_name']],
            ["Patient ID:", scan['patient_id']],
            ["Scan Date:", scan['scan_date']],
            ["Report Date:", datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        ]
        patient_table = Table(patient_data, colWidths=[1.5*inch, 4*inch])
        patient_table.setStyle(TableStyle([
            ('FONT', (0, 0), (-1, -1), 'Helvetica', 10),
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f8f9fa')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2c3e50')),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#e8ecef')),
            ('INNERGRID', (0, 0), (-1, -1), 1, colors.HexColor('#e8ecef')),
        ]))
        story.append(patient_table)
        story.append(Spacer(1, 20))
        
        # Diagnosis
        story.append(Paragraph("DIAGNOSIS", heading_style))
        diagnosis_data = [
            ["Condition Detected:", scan['condition']],
            ["Confidence Level:", f"{scan['confidence']}%"],
            ["Analysis Status:", scan['status']]
        ]
        diagnosis_table = Table(diagnosis_data, colWidths=[1.5*inch, 4*inch])
        diagnosis_table.setStyle(TableStyle([
            ('FONT', (0, 0), (-1, -1), 'Helvetica', 10),
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f8f9fa')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2c3e50')),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#e8ecef')),
        ]))
        story.append(diagnosis_table)
        story.append(Spacer(1, 20))
        
        # Findings
        story.append(Paragraph("FINDINGS", heading_style))
        findings_text = f"The AI analysis detected {scan['condition']} with {scan['confidence']}% confidence level."
        story.append(Paragraph(findings_text, normal_style))
        story.append(Spacer(1, 15))
        
        # Recommendations
        story.append(Paragraph("RECOMMENDATIONS", heading_style))
        recommendations = [
            "1. Consult with a pulmonologist for further evaluation",
            "2. Follow up in 2 weeks for repeat imaging if symptoms persist",
            "3. Consider additional diagnostic tests if clinically indicated",
            "4. Monitor symptoms and seek immediate care if condition worsens"
        ]
        for rec in recommendations:
            story.append(Paragraph(rec, normal_style))
            story.append(Spacer(1, 5))
        
        story.append(Spacer(1, 20))
        
        # Disclaimer
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=styles["BodyText"],
            fontSize=8,
            textColor=colors.gray,
            alignment=1  # Center aligned
        )
        disclaimer_text = "This report is generated by AI and should be reviewed by a qualified healthcare professional. Always consult with medical experts for diagnosis and treatment decisions."
        story.append(Paragraph(disclaimer_text, disclaimer_style))
        
        # Build PDF
        doc.build(story)
        
        # Get PDF value from buffer
        pdf = buffer.getvalue()
        buffer.close()
        
        # Return PDF as response
        response = Response(pdf, mimetype='application/pdf')
        response.headers['Content-Disposition'] = f'attachment; filename=lungscan-report-{scan_id}.pdf'
        return response
        
    except Exception as e:
        return jsonify({"error": f"PDF generation failed: {str(e)}"}), 500

# Also add this simple download endpoint for testing
@app.route('/api/reports/<scan_id>/download', methods=['GET'])
def download_report(scan_id):
    user = get_current_user()
    if not user:
        return jsonify({"error": "Unauthorized"}), 401
    
    scan = next((s for s in scans_db if s['id'] == scan_id), None)
    if not scan:
        return jsonify({"error": "Scan not found"}), 404
    
    return jsonify({
        "message": f"Report ready for download: {scan_id}",
        "download_url": f"/api/reports/{scan_id}/pdf"
    })

@app.route('/api/patients', methods=['POST'])
def add_patient():
    user = get_current_user()
    if not user:
        return jsonify({"error": "Unauthorized"}), 401
    
    try:
        data = request.get_json()
        
        # Create new patient (simplified - just return success)
        return jsonify({
            "message": "Patient added successfully",
            "patient_id": f"PT-{str(len(scans_db) + 1).zfill(4)}"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    print("🚀 Starting Enhanced LungScan AI Server on http://localhost:8000")
    print("📡 Available endpoints:")
    print("  GET  /api/health")
    print("  POST /api/register")
    print("  POST /api/login")
    print("  GET  /api/scans")
    print("  POST /api/scans/upload")
    print("  GET  /api/reports/<scan_id>")
    print("  GET  /api/reports/<scan_id>/pdf")
    print("  POST /api/patients")
    print("👤 Default users:")
    for user in users_db:
        print(f"  - {user['email']} : {user['password']}")
    app.run(debug=True, port=8000, host='0.0.0.0')