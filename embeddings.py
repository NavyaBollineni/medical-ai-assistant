from chunking import return_chunks
from sentence_transformers import SentenceTransformer


# --------------------------------
# 1. Get chunks from chunking.py
# --------------------------------

(
    doctor_chunks,
    hospital_chunks,
    specialty_chunks,
    availability_chunks,
    appointment_rule_chunks,
    example_query_chunks
) = return_chunks


# --------------------------------
# 2. Load embedding model
# --------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------
# 3. Create embeddings
# --------------------------------

doctor_embeddings = model.encode(doctor_chunks)

hospital_embeddings = model.encode(hospital_chunks)

specialty_embeddings = model.encode(specialty_chunks)

availability_embeddings = model.encode(availability_chunks)

appointment_rule_embeddings = model.encode(
    appointment_rule_chunks
)

example_query_embeddings = model.encode(
    example_query_chunks
)


# --------------------------------
# 4. Check embedding shapes
# --------------------------------

# print("\nEmbedding shapes:")

# print("Doctors:", doctor_embeddings.shape)
# print("Hospitals:", hospital_embeddings.shape)
# print("Specialties:", specialty_embeddings.shape)
# print("Availability:", availability_embeddings.shape)
# print("Appointment Rules:", appointment_rule_embeddings.shape)
# print("Example Queries:", example_query_embeddings.shape)


# --------------------------------
# 5. Return embeddings
# --------------------------------

return_embeddings = (
    doctor_embeddings,
    hospital_embeddings,
    specialty_embeddings,
    availability_embeddings,
    appointment_rule_embeddings,
    example_query_embeddings
)