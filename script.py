from transformers import pipeline
import torch

model_id = "openai/gpt-oss-20b"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype="auto",
    device_map="auto",
)

messages = [
    {"role": "user", "content": "When was Utahraptor first discovered and who discovered it?"},
]

outputs = pipe(
    messages,
    max_new_tokens=256,
)

print(outputs[0]["generated_text"][-1])

# {
#   'role': 'assistant',
#   'content': 'Quantum mechanics is a fundamental theory in physics that describes the physical properties of nature at the scale of atoms and subatomic particles. It introduces concepts like wave-particle duality, quantization, superposition, and entanglement. Key principles include the uncertainty principle (position and momentum cannot both be precisely defined), the probabilistic nature of particles, and the conservation of energy.'
# }
