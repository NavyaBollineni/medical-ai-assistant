from embeddings import return_embeddings
import faiss
import numpy as np


# --------------------------------
# 1. Get embeddings
# --------------------------------

(
    doctor_embeddings,
    hospital_embeddings,
    specialty_embeddings,
    availability_embeddings,
    appointment_rule_embeddings,
    example_query_embeddings
) = return_embeddings


# --------------------------------
# 2. Convert embeddings to NumPy
# --------------------------------

doctor_embeddings = np.array(
    doctor_embeddings
).astype("float32")

hospital_embeddings = np.array(
    hospital_embeddings
).astype("float32")

specialty_embeddings = np.array(
    specialty_embeddings
).astype("float32")

availability_embeddings = np.array(
    availability_embeddings
).astype("float32")

appointment_rule_embeddings = np.array(
    appointment_rule_embeddings
).astype("float32")

example_query_embeddings = np.array(
    example_query_embeddings
).astype("float32")


# --------------------------------
# 3. Create FAISS indexes
# --------------------------------

doctor_index = faiss.IndexFlatL2(384)

hospital_index = faiss.IndexFlatL2(384)

specialty_index = faiss.IndexFlatL2(384)

availability_index = faiss.IndexFlatL2(384)

appointment_rule_index = faiss.IndexFlatL2(384)

example_query_index = faiss.IndexFlatL2(384)


# --------------------------------
# 4. Add embeddings to FAISS
# --------------------------------

doctor_index.add(doctor_embeddings)

hospital_index.add(hospital_embeddings)

specialty_index.add(specialty_embeddings)

availability_index.add(availability_embeddings)

appointment_rule_index.add(
    appointment_rule_embeddings
)

example_query_index.add(
    example_query_embeddings
)


# --------------------------------
# 5. Check number of vectors
# --------------------------------

# print("\nNumber of vectors in FAISS:")

# print("Doctors:", doctor_index.ntotal)
# print("Hospitals:", hospital_index.ntotal)
# print("Specialties:", specialty_index.ntotal)
# print("Availability:", availability_index.ntotal)
# print("Appointment Rules:", appointment_rule_index.ntotal)
# print("Example Queries:", example_query_index.ntotal)


# --------------------------------
# 6. Return vector stores
# --------------------------------

return_vector_stores = (
    doctor_index,
    hospital_index,
    specialty_index,
    availability_index,
    appointment_rule_index,
    example_query_index
)