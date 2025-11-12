from transformers import pipeline
import torch

model_id = "google/gemma-3-1b-it"

pipe = pipeline(
  "text-generation",
  model=model_id,
  device="cuda",
  torch_dtype=torch.bfloat16
)

messages = [
  {
    "role": "system",
    "content": "You are a paleontologist. You are a specialist in dromaeosaurid. If you dont' have information on a specific dromaeosaurid, please say so."
  },
  {
    "role": "user",
    "content": "Who discovered Utahraptor ostrommaysi?"
  },
]

outputs = pipe(
  messages,
  max_new_tokens=512,  # Increase token limit to avoid cutoff
)

# Print the full response with proper formatting
response = outputs[0]["generated_text"][-1]
print(response['content'])

# {
#   'role': 'assistant',
#   'content': 'Quantum mechanics is a fundamental theory in physics that describes the physical properties of nature at the scale of atoms and subatomic particles. It introduces concepts like wave-particle duality, quantization, superposition, and entanglement. Key principles include the uncertainty principle (position and momentum cannot both be precisely defined), the probabilistic nature of particles, and the conservation of energy.'
# }
