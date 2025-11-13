# Example of creating a patient report
from datetime import datetime
import json

def create_patient_report(
    db: Session, 
    user_id: int,
    patient_data: dict,
    analysis_results: dict
):
    report = PatientReport(
        user_id=user_id,
        patient_name=patient_data['name'],
        patient_age=patient_data['age'],
        patient_gender=patient_data['gender'],
        image_filename=analysis_results['processed_filename'],
        original_filename=analysis_results['original_filename'],
        prediction_result=analysis_results['primary_diagnosis'],
        confidence=analysis_results['confidence'],
        all_predictions=json.dumps(analysis_results['all_predictions']),
        risk_level=analysis_results['risk_level'],
        recommendations=analysis_results['recommendations'],
        findings_summary=analysis_results['findings'],
        scan_type=patient_data.get('scan_type', 'CT'),
        scan_date=patient_data.get('scan_date', datetime.utcnow()),
        report_date=datetime.utcnow()
    )
    
    db.add(report)
    db.commit()
    db.refresh(report)
    return report