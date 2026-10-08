import pandas as pd
file_path ="c:\\Users\\SASIDHAR\\Downloads\\medical_ai_appointment_chatbot.xlsx"


def load_data(file_path):

    excel_file = pd.ExcelFile(file_path)

    doctors = pd.read_excel(excel_file, sheet_name="Doctors")
    hospitals = pd.read_excel(excel_file, sheet_name="Hospitals")
    specialties = pd.read_excel(excel_file, sheet_name="Specialties")
    availability = pd.read_excel(excel_file, sheet_name="Availability")
    appointment_rules = pd.read_excel(
        excel_file,
        sheet_name="Appointment_Rules"
    )
    example_queries = pd.read_excel(
        excel_file,
        sheet_name="Example_Queries"
    )

    return (
        doctors,
        hospitals,
        specialties,
        availability,
        appointment_rules,
        example_queries
    )