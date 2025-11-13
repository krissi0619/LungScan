from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from datetime import datetime
import json
import os
import logging
from typing import Dict, List, Optional
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
        self.setup_colors()
    
    def setup_colors(self):
        """Define color scheme for the report"""
        self.colors = {
            'primary': colors.HexColor('#2c3e50'),      # Dark blue
            'secondary': colors.HexColor('#34495e'),    # Medium blue
            'accent': colors.HexColor('#3498db'),       # Light blue
            'success': colors.HexColor('#27ae60'),      # Green
            'warning': colors.HexColor('#f39c12'),      # Orange
            'danger': colors.HexColor('#e74c3c'),       # Red
            'light': colors.HexColor('#ecf0f1'),        # Light gray
            'dark': colors.HexColor('#2c3e50'),         # Dark gray
        }
    
    def setup_custom_styles(self):
        """Setup custom styles for the report"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=20,
            spaceAfter=20,
            textColor=colors.HexColor('#2c3e50'),
            alignment=1,  # Center aligned
            fontName='Helvetica-Bold'
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.HexColor('#2c3e50'),
            fontName='Helvetica-Bold',
            leftIndent=0
        ))
        
        # Subsection style
        self.styles.add(ParagraphStyle(
            name='SubSection',
            parent=self.styles['Heading3'],
            fontSize=12,
            spaceAfter=8,
            textColor=colors.HexColor('#34495e'),
            fontName='Helvetica-Bold'
        ))
        
        # Body text style
        self.styles.add(ParagraphStyle(
            name='BodyText',
            parent=self.styles['BodyText'],
            fontSize=10,
            spaceAfter=6,
            textColor=colors.black
        ))
        
        # Emphasis style
        self.styles.add(ParagraphStyle(
            name='Emphasis',
            parent=self.styles['BodyText'],
            fontSize=10,
            spaceAfter=6,
            textColor=colors.HexColor('#e74c3c'),
            fontName='Helvetica-Bold'
        ))
        
        # Footer style
        self.styles.add(ParagraphStyle(
            name='Footer',
            fontSize=8,
            textColor=colors.gray,
            alignment=1  # Center aligned
        ))
    
    def get_risk_color(self, risk_level: str) -> colors.Color:
        """Get color based on risk level"""
        risk_colors = {
            'LOW': self.colors['success'],
            'MEDIUM': self.colors['warning'],
            'HIGH': self.colors['danger'],
            'CRITICAL': colors.HexColor('#8e44ad'),  # Purple for critical
            'NORMAL': self.colors['success']
        }
        return risk_colors.get(risk_level.upper(), colors.black)
    
    def create_header(self, story: List):
        """Create report header"""
        # Main title
        story.append(Paragraph("LUNGSCAN AI - DIAGNOSTIC REPORT", self.styles['ReportTitle']))
        story.append(Spacer(1, 0.1*inch))
        
        # Subtitle
        subtitle_style = ParagraphStyle(
            name='Subtitle',
            fontSize=12,
            textColor=colors.HexColor('#7f8c8d'),
            alignment=1
        )
        story.append(Paragraph("Advanced AI-Powered Lung Disease Detection System", subtitle_style))
        story.append(Spacer(1, 0.3*inch))
    
    def create_patient_section(self, story: List, report_data: Dict):
        """Create patient information section"""
        story.append(Paragraph("PATIENT INFORMATION", self.styles['SectionHeader']))
        
        patient_info = [
            ["Patient Name:", f"<b>{report_data['patient_name']}</b>"],
            ["Age / Gender:", f"<b>{report_data['patient_age']} years / {report_data['patient_gender']}</b>"],
            ["Report ID:", f"<b>{report_data['report_id']}</b>"],
            ["Scan Date:", f"<b>{report_data.get('scan_date', report_data['report_date'])}</b>"],
            ["Scan Type:", f"<b>{report_data.get('scan_type', 'CT Scan')}</b>"],
            ["Report Date:", f"<b>{report_data['report_date']}</b>"]
        ]
        
        patient_table = Table(patient_info, colWidths=[1.5*inch, 4.5*inch])
        patient_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), self.colors['light']),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(patient_table)
        story.append(Spacer(1, 0.3*inch))
    
    def create_diagnosis_section(self, story: List, report_data: Dict):
        """Create diagnosis results section"""
        story.append(Paragraph("DIAGNOSIS SUMMARY", self.styles['SectionHeader']))
        
        risk_color = self.get_risk_color(report_data['risk_level'])
        confidence = report_data.get('confidence', 0)
        
        diagnosis_info = [
            ["Primary Diagnosis:", f"<b>{report_data['prediction_result']}</b>"],
            ["Confidence Level:", f"<b>{confidence:.1f}%</b>"],
            ["Risk Assessment:", f"<font color={risk_color.hexval()}><b>{report_data['risk_level']} RISK</b></font>"],
            ["Clinical Findings:", f"<i>{report_data.get('findings_summary', 'No specific findings noted.')}</i>"]
        ]
        
        diagnosis_table = Table(diagnosis_info, colWidths=[1.8*inch, 4.2*inch])
        diagnosis_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), self.colors['light']),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.white, colors.white, colors.white]),
        ]))
        story.append(diagnosis_table)
        story.append(Spacer(1, 0.3*inch))
    
    def create_detailed_analysis_section(self, story: List, report_data: Dict):
        """Create detailed analysis section with predictions"""
        story.append(Paragraph("DETAILED ANALYSIS", self.styles['SectionHeader']))
        
        # Parse predictions
        try:
            if isinstance(report_data['all_predictions'], str):
                predictions = json.loads(report_data['all_predictions'])
            else:
                predictions = report_data['all_predictions']
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Could not parse predictions: {e}")
            predictions = {}
        
        # Create prediction table
        prediction_data = [["Condition", "Confidence", "Risk Level"]]
        
        for condition, data in predictions.items():
            if isinstance(data, dict):
                confidence = data.get('percentage', data.get('confidence', 0)) * (100 if data.get('confidence') else 1)
                risk_level = "HIGH" if condition != "NORMAL" and confidence > 50 else "LOW"
            else:
                confidence = float(data) * 100
                risk_level = "HIGH" if condition != "NORMAL" and confidence > 50 else "LOW"
            
            risk_color = self.get_risk_color(risk_level)
            prediction_data.append([
                condition.replace('_', ' ').title(),
                f"{confidence:.1f}%",
                f"<font color={risk_color.hexval()}>{risk_level}</font>"
            ])
        
        if len(prediction_data) > 1:
            prediction_table = Table(prediction_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
            prediction_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), self.colors['secondary']),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, self.colors['light']]),
            ]))
            story.append(prediction_table)
        else:
            story.append(Paragraph("No detailed prediction data available.", self.styles['BodyText']))
        
        story.append(Spacer(1, 0.3*inch))
    
    def create_recommendations_section(self, story: List, report_data: Dict):
        """Create medical recommendations section"""
        story.append(Paragraph("MEDICAL RECOMMENDATIONS", self.styles['SectionHeader']))
        
        # Parse recommendations
        recommendations = []
        if 'recommendations' in report_data:
            if isinstance(report_data['recommendations'], str):
                recommendations = [rec.strip() for rec in report_data['recommendations'].split('\n') if rec.strip()]
            elif isinstance(report_data['recommendations'], list):
                recommendations = report_data['recommendations']
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                story.append(Paragraph(f"{i}. {rec}", self.styles['BodyText']))
        else:
            story.append(Paragraph("No specific recommendations available. Please consult with a healthcare professional.", self.styles['BodyText']))
        
        story.append(Spacer(1, 0.3*inch))
    
    def create_disclaimer_section(self, story: List):
        """Create disclaimer section"""
        story.append(Paragraph("IMPORTANT DISCLAIMER", self.styles['SectionHeader']))
        
        disclaimer_text = """
        <b>This AI-generated report is for educational and preliminary screening purposes only.</b><br/><br/>
        
        • This analysis is based on AI algorithms and should not be considered a definitive medical diagnosis<br/>
        • Always consult qualified healthcare professionals for medical concerns and treatment decisions<br/>
        • The accuracy of this analysis depends on image quality and clinical context<br/>
        • False positives and false negatives are possible with any diagnostic tool<br/>
        • This report does not replace comprehensive medical evaluation by licensed physicians<br/>
        • Treatment decisions should not be based solely on this automated analysis<br/>
        """
        
        story.append(Paragraph(disclaimer_text, self.styles['BodyText']))
        story.append(Spacer(1, 0.3*inch))
    
    def create_footer(self, story: List):
        """Create report footer"""
        story.append(Spacer(1, 0.5*inch))
        
        footer_text = f"""
        Generated by LungScan AI Medical Imaging System | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
        Advanced Deep Learning for Pulmonary Disease Detection | Confidential Medical Document
        """
        
        story.append(Paragraph(footer_text, self.styles['Footer']))
    
    def generate_pdf_report(self, report_data: Dict, output_path: str) -> str:
        """Generate comprehensive PDF report for lung disease analysis"""
        try:
            logger.info(f"Generating PDF report: {output_path}")
            
            # Create document with margins
            doc = SimpleDocTemplate(
                output_path, 
                pagesize=A4,
                topMargin=0.5*inch,
                bottomMargin=0.5*inch,
                leftMargin=0.5*inch,
                rightMargin=0.5*inch
            )
            
            story = []
            
            # Build report sections
            self.create_header(story)
            self.create_patient_section(story, report_data)
            self.create_diagnosis_section(story, report_data)
            self.create_detailed_analysis_section(story, report_data)
            self.create_recommendations_section(story, report_data)
            self.create_disclaimer_section(story)
            self.create_footer(story)
            
            # Build PDF
            doc.build(story)
            logger.info(f"PDF report generated successfully: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating PDF report: {str(e)}")
            raise Exception(f"Failed to generate PDF report: {str(e)}")
    
    def generate_html_report(self, report_data: Dict, output_path: str) -> str:
        """Generate HTML report (alternative format)"""
        try:
            # Basic HTML report implementation
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>LungScan AI Report - {report_data['report_id']}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .header {{ text-align: center; color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 20px; }}
                    .section {{ margin: 20px 0; }}
                    .section-title {{ color: #2c3e50; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; }}
                    table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                    th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                    th {{ background-color: #34495e; color: white; }}
                    .disclaimer {{ background-color: #f8f9fa; padding: 15px; border-left: 4px solid #e74c3c; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>LUNGSCAN AI - DIAGNOSTIC REPORT</h1>
                    <p>Advanced AI-Powered Lung Disease Detection System</p>
                </div>
                
                <div class="section">
                    <h2 class="section-title">Patient Information</h2>
                    <p><strong>Name:</strong> {report_data['patient_name']}</p>
                    <p><strong>Age/Gender:</strong> {report_data['patient_age']} years / {report_data['patient_gender']}</p>
                    <p><strong>Report ID:</strong> {report_data['report_id']}</p>
                    <p><strong>Report Date:</strong> {report_data['report_date']}</p>
                </div>
                
                <div class="section">
                    <h2 class="section-title">Diagnosis Summary</h2>
                    <p><strong>Primary Diagnosis:</strong> {report_data['prediction_result']}</p>
                    <p><strong>Confidence Level:</strong> {report_data.get('confidence', 0):.1f}%</p>
                    <p><strong>Risk Assessment:</strong> {report_data['risk_level']} RISK</p>
                </div>
                
                <div class="disclaimer">
                    <h3>Important Disclaimer</h3>
                    <p>This AI-generated report is for educational and preliminary screening purposes only. Always consult qualified healthcare professionals for medical diagnosis and treatment.</p>
                </div>
            </body>
            </html>
            """
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"HTML report generated: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating HTML report: {str(e)}")
            raise

def generate_report_id() -> str:
    """Generate unique report ID"""
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    unique_id = uuid.uuid4().hex[:6].upper()
    return f"LUNG_{timestamp}_{unique_id}"

# Example usage and testing
if __name__ == "__main__":
    # Test the report generator
    generator = ReportGenerator()
    
    # Sample report data
    sample_data = {
        'patient_name': 'John Doe',
        'patient_age': 45,
        'patient_gender': 'Male',
        'report_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'scan_date': '2024-01-15',
        'scan_type': 'CT Scan',
        'report_id': generate_report_id(),
        'prediction_result': 'PNEUMONIA',
        'confidence': 87.5,
        'risk_level': 'HIGH',
        'findings_summary': 'Consolidation in right lower lobe suggestive of bacterial pneumonia.',
        'all_predictions': json.dumps({
            'NORMAL': {'percentage': 5.2},
            'PNEUMONIA': {'percentage': 87.5},
            'COVID-19': {'percentage': 4.1},
            'TUBERCULOSIS': {'percentage': 2.1},
            'LUNG_CANCER': {'percentage': 1.1}
        }),
        'recommendations': [
            'Consult pulmonologist immediately',
            'Complete prescribed antibiotic course',
            'Monitor symptoms and follow up in 48 hours',
            'Chest X-ray follow-up recommended in 2 weeks'
        ]
    }
    
    # Generate PDF report
    output_pdf = "test_report.pdf"
    generator.generate_pdf_report(sample_data, output_pdf)
    print(f"Test PDF report generated: {output_pdf}")
    
    # Generate HTML report
    output_html = "test_report.html"
    generator.generate_html_report(sample_data, output_html)
    print(f"Test HTML report generated: {output_html}")