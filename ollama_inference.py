import ollama

messages = [
    {"role": "user", "content": "When was Utahraptor first discovered and who discovered it?"},
]

response = ollama.chat(
    model="gpt-oss:20b",
    messages=messages,
)

print(response['message']['content'])
