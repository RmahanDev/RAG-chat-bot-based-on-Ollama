from rag import ask

while True:
    q = input("You: ")

    if q.lower() == "exit":
        break

    print("\nBot:", ask(q))
    print()