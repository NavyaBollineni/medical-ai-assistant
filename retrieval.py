from vector_db import return_vector_stores
from chunking import return_chunks
from sentence_transformers import SentenceTransformer


# --------------------------------
# 1. Get vector stores
# --------------------------------

(
    doctor_index,
    hospital_index,
    specialty_index,
    availability_index,
    appointment_rule_index,
    example_query_index
) = return_vector_stores


# --------------------------------
# 2. Get original chunks
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
# 3. Load embedding model
# --------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------
# 4. Retrieval function
# --------------------------------

def retrieve_information(query, top_k=3):

    # Convert user query into embedding

    query_embedding = model.encode(
        [query]
    ).astype("float32")


    # --------------------------------
    # Search each FAISS index
    # --------------------------------

    doctor_distances, doctor_indices = doctor_index.search(
        query_embedding,
        top_k
    )

    hospital_distances, hospital_indices = hospital_index.search(
        query_embedding,
        top_k
    )

    specialty_distances, specialty_indices = specialty_index.search(
        query_embedding,
        top_k
    )

    availability_distances, availability_indices = availability_index.search(
        query_embedding,
        top_k
    )

    appointment_rule_distances, appointment_rule_indices = appointment_rule_index.search(
        query_embedding,
        top_k
    )

    example_query_distances, example_query_indices = example_query_index.search(
        query_embedding,
        top_k
    )


    # --------------------------------
    # Get matching chunks
    # --------------------------------

    retrieved_doctors = [
        doctor_chunks[i]
        for i in doctor_indices[0]
        if i != -1
    ]

    retrieved_hospitals = [
        hospital_chunks[i]
        for i in hospital_indices[0]
        if i != -1
    ]

    retrieved_specialties = [
        specialty_chunks[i]
        for i in specialty_indices[0]
        if i != -1
    ]

    retrieved_availability = [
        availability_chunks[i]
        for i in availability_indices[0]
        if i != -1
    ]

    retrieved_appointment_rules = [
        appointment_rule_chunks[i]
        for i in appointment_rule_indices[0]
        if i != -1
    ]

    retrieved_example_queries = [
        example_query_chunks[i]
        for i in example_query_indices[0]
        if i != -1
    ]


    # --------------------------------
    # Combine retrieved information
    # --------------------------------

    retrieved_information = {

        "doctors": retrieved_doctors,

        "hospitals": retrieved_hospitals,

        "specialties": retrieved_specialties,

        "availability": retrieved_availability,

        "appointment_rules": retrieved_appointment_rules,

        "example_queries": retrieved_example_queries
    }


    # --------------------------------
    # Return information for LLM
    # --------------------------------

    return retrieved_information