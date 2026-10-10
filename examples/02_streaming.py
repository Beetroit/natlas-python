"""Real-time token streaming with N-ATLAS."""

from natlas import Natlas

client = Natlas()

print("Streaming Hausa response:")
for chunk in client.chat.stream(
    language="hausa",
    messages=[{"role": "user", "content": "Faɗa mini tarihin Najeriya a taƙaice."}],
):
    delta = chunk.choices[0].delta.content or ""
    print(delta, end="", flush=True)
print("\n")

client.close()
