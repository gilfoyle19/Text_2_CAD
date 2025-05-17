# Text_2_CAD

Generate CAD models using Llama 3.1 and the CadQuery library via Retrieval Augmented Generation (RAG). This project uses a Colab notebook to combine LLM-powered code generation with vector search over CadQuery documentation and few-shot examples.

## Features

- **Retrieval Augmented Generation (RAG):** Uses ChromaDB to retrieve relevant CadQuery documentation snippets.
- **LLM Integration:** Employs Ollama with Llama 3.1 8B for code generation.
- **Few-shot Prompting:** Guides the LLM with curated CadQuery code examples.
- **Automatic CAD Export:** Generated code always exports models as STEP files.
- **Colab/Google Colab Ready:** Designed for interactive use with GPU acceleration.

## How to run
Run every cell in the sequence by using T4 GPU runtime.

## Usage Example

### Generate a Cube

```python
prompt = "Create a cube with side length 20 mm and export it as a STEP file."
code = generate_cad_code(prompt)
print(code)
exec(code)
```

### Generate a Cylinder

```python
prompt = "Create a cylinder with radius 10 mm and height 30 mm, then export it as a STEP file."
code = generate_cad_code(prompt)
print(code)
exec(code)
```

### Generate a Sphere

```python
prompt = "Create a sphere with radius 15 mm and export it as a STEP file."
code = generate_cad_code(prompt)
print(code)
exec(code)
```

## Downloading the STEP File

After running the generated code, download the file (in Colab):

```python
from google.colab import files
import os
files.download(os.path.join(os.getcwd(), 'sphere_5mm.step'))
```

## Notes

- Make sure Ollama is running and the required models are pulled.
- The notebook will automatically scrape CadQuery documentation and store it in a vector database for retrieval.
- All generated code will use `cadquery` and export models as STEP files.

---

**Author:** [Your Name]  
**License:** MIT
