from llm import generate_answer


query = input("You: ")


answer = generate_answer(query)


print("\nAssistant:", answer)