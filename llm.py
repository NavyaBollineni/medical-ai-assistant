from retrieval import retrieve_information
from google import genai


# --------------------------------
# 1. Create Gemini client
# --------------------------------

client = genai.Client(
    api_key="AQ.Ab8RN6KSNAeqIPzMlT7qYM2msVqD0a-z9XfN0ohBQ7V3VxMUyQY"
)



# --------------------------------
# 2. LLM function
# --------------------------------

def generate_answer(query):

    # Get relevant information from retrieval.py

    retrieved_information = retrieve_information(query)


    # --------------------------------
    # 3. Create context
    # --------------------------------

    context = f"""
Doctors:
{retrieved_information["doctors"]}

Hospitals:
{retrieved_information["hospitals"]}

Specialties:
{retrieved_information["specialties"]}

Availability:
{retrieved_information["availability"]}

Appointment Rules:
{retrieved_information["appointment_rules"]}

Example Queries:
{retrieved_information["example_queries"]}
"""


    # --------------------------------
    # 4. Create prompt
    # --------------------------------

    prompt = f"""
You are a medical appointment assistant.

Your job is to help users with:
- finding doctors
- finding hospitals
- checking specialties
- checking appointment availability
- explaining appointment rules

Use the provided context to answer the user's question.

Do not invent information.

If the requested information is not available in the context,
clearly say that the information is not available.

Do not diagnose diseases or provide medical treatment advice.

Give a clear, concise and friendly answer.

User Question:
{query}

Retrieved Context:
{context}
"""


    # --------------------------------
    # 5. Send prompt to LLM
    # --------------------------------

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )


    # --------------------------------
    # 6. Get final answer
    # --------------------------------

    final_answer = response.text


    # --------------------------------
    # 7. Return answer
    # --------------------------------

    return final_answer