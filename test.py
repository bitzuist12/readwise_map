from nomic import embed
import numpy as np

NOMIC_API_KEY = 'nk-IVcWXyMuLa6GqhjNgAwJ9JZxPjDcBDn06ykHy_l8ZTQ'

output = embed.text(
    texts=[
        "Who is Laurens van der Maaten?",
        "What is dimensionality reduction?",
    ],
    model='nomic-embed-text-v1',
)

print(output['usage'])
embeddings = np.array(output['embeddings'])
print(embeddings.shape)
