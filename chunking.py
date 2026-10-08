from data_ingestion import load_data, file_path


# --------------------------------
# 1. Get data from data_ingestion
# --------------------------------

(
    doctors,
    hospitals,
    specialties,
    availability,
    appointment_rules,
    example_queries
) = load_data(file_path)


# --------------------------------
# 2. Convert DataFrame to text
# --------------------------------

def convert_to_text(data):

    text = data.to_string(index=False)

    return text


# --------------------------------
# 3. Fixed-size chunking
# --------------------------------

def create_chunks(text, chunk_size=250):

    chunks = []

    for start in range(0, len(text), chunk_size):

        chunk = text[start:start + chunk_size]

        chunks.append(chunk)

    return chunks


# --------------------------------
# 4. Convert data into text
# --------------------------------

doctors_text = convert_to_text(doctors)
hospitals_text = convert_to_text(hospitals)
specialties_text = convert_to_text(specialties)
availability_text = convert_to_text(availability)
appointment_rules_text = convert_to_text(appointment_rules)
example_queries_text = convert_to_text(example_queries)


# --------------------------------
# 5. Create 250-character chunks
# --------------------------------

doctor_chunks = create_chunks(doctors_text)
hospital_chunks = create_chunks(hospitals_text)
specialty_chunks = create_chunks(specialties_text)
availability_chunks = create_chunks(availability_text)
appointment_rule_chunks = create_chunks(appointment_rules_text)
example_query_chunks = create_chunks(example_queries_text)


# --------------------------------
# 6. Check the chunks
# --------------------------------

# print("Number of doctor chunks:", len(doctor_chunks))
# print("Number of hospital chunks:", len(hospital_chunks))
# print("Number of specialty chunks:", len(specialty_chunks))
# print("Number of availability chunks:", len(availability_chunks))
# print("Number of appointment rule chunks:", len(appointment_rule_chunks))
# print("Number of example query chunks:", len(example_query_chunks))


# print("\nFirst Doctor Chunk:")
# print(doctor_chunks[0])


# --------------------------------
# 7. Return chunks for embeddings
# --------------------------------

return_chunks = (
    doctor_chunks,
    hospital_chunks,
    specialty_chunks,
    availability_chunks,
    appointment_rule_chunks,
    example_query_chunks
)