from data_ingestion import load_data,file_path


# Excel file path
#file_path ="c:\\Users\\SASIDHAR\\Downloads\\medical_ai_appointment_chatbot.xlsx"


# Get the data returned from data_ingestion.py
(
    doctors,
    hospitals,
    specialties,
    availability,
    appointment_rules,
    example_queries
) = load_data(file_path)


# --------------------------------
# Check missing / NULL values
# --------------------------------

print("\n========== MISSING VALUES ==========")

print("\nDoctors:")
print(doctors.isnull().sum())

print("\nHospitals:")
print(hospitals.isnull().sum())

print("\nSpecialties:")
print(specialties.isnull().sum())

print("\nAvailability:")
print(availability.isnull().sum())

print("\nAppointment Rules:")
print(appointment_rules.isnull().sum())

print("\nExample Queries:")
print(example_queries.isnull().sum())


# --------------------------------
# Check duplicate rows
# --------------------------------

print("\n========== DUPLICATE VALUES ==========")

print("Doctors:", doctors.duplicated().sum())
print("Hospitals:", hospitals.duplicated().sum())
print("Specialties:", specialties.duplicated().sum())
print("Availability:", availability.duplicated().sum())
print("Appointment Rules:", appointment_rules.duplicated().sum())
print("Example Queries:", example_queries.duplicated().sum())