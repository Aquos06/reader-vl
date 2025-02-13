# Reader VL

![PyPI](https://img.shields.io/pypi/v/reader_vl) ![License](https://img.shields.io/github/license/yourusername/reader_vl) ![Python](https://img.shields.io/badge/python-3.8%2B-blue)

## Overview

**Reader VL** goes beyond simple parsing. Our SDK and CLI empower you to build intelligent applications by converting diverse document formats (PDF, DOCX, HTML, and others) into a unified structure. Critically, we leverage multimodal LLMs to enrich the parsed content, adding layers of meaning and context essential for maximizing the performance of your generative AI pipelines.

## Features

- 📄 **Multi-format Support**: Process PDFs, DOCX, HTML, and more.
- 🤖 **Multimodal LLM Integration**: Enhance extracted data with AI-powered insights.
- 🏗 **Structured Output**: Convert documents into a well-defined schema.
- 🏎 **Fast & Efficient**: Leverages YOLO for object detection and Tesseract OCR.
- 🔧 **Extensible**: Easily integrate with your AI pipelines.

## Installation

You can install **Reader VL** via pip:

```bash
pip install reader_vl
```

## Usage

### Basic Example

```python
from reader_vl.docs.reader import DocReader
from reader_vl.llm.client import OpenAIClient, VllmClient

# Initialize OpenAI client
llm = OpenAIClient(api_key="your-api-key", model="gpt-4")

# Load a PDF file
reader = DocReader(llm=llm, file_path="sample.pdf")

# Parse the document
parsed_document = reader.parse()

# Display structured output
print(parsed_document)
```

Alternatively, if you're using a self-hosted LLM:

```python
# Initialize VLLM client
llm = VllmClient(url="http://localhost:8000", model="custom-llm", temperature=0.7, max_tokens=512)

# Load and parse the document
reader = DocReader(llm=llm, file_path="sample.pdf")
parsed_document = reader.parse()
print(parsed_document)
```

### Asynchronous Parsing

```python
import asyncio

async def main():
    parsed_doc = await reader.aparse()
    print(parsed_doc)

asyncio.run(main())
```

## LLM Clients

Reader VL provides integration with multiple LLM clients for processing document content.

### OpenAI Client (DocReader LLM)

`OpenAIClient` integrates with OpenAI models for text completion and chat-based interactions within **DocReader LLM**.

#### Usage:

```python
from reader_vl.llm.client import OpenAIClient

client = OpenAIClient(api_key="your-api-key", model="gpt-4")

response = client.completion("Summarize this text:")
print(response)
```

#### Features:
- Supports text completion and chat interactions.
- Works with OpenAI's API.
- Provides both synchronous and asynchronous methods.

### VLLM Client (DocReader LLM)

`VllmClient` is designed to interact with self-hosted or remote LLM APIs within **DocReader LLM**.

#### Usage:

```python
from reader_vl.llm.client import VllmClient

client = VllmClient(url="http://localhost:8000", model="custom-llm", temperature=0.7, max_tokens=512)

response = client.completion("Extract key points from this document.")
print(response)
```

#### Features:
- Works with self-hosted LLMs via REST APIs.
- Supports streaming responses for efficiency.
- Provides synchronous and asynchronous support.

## Components

**Reader VL** extracts structured components from documents, such as:

- **Header** (Page numbers, section titles)
- **Title** (Main document title)
- **Section** (Text sections with hierarchy)
- **Tables** (Markdown-formatted tables)
- **Lists** (Ordered and unordered lists)
- **Equations** (Extracted and explained formulas)
- **Charts** (Visual data representations)
- **References & Captions** (Bibliographic elements and figure/table captions)

## Architecture

**Reader VL** follows a modular approach:

- **YOLO-based Object Detection**: Detects document elements (headers, sections, tables, etc.).
- **OCR (Tesseract)**: Extracts text from detected regions.
- **Multimodal LLM Processing**: Enhances extracted content for AI pipelines.
- **Schema-based Output**: Provides structured data (JSON-like `Document` schema).

## Extending Reader VL

Reader VL supports custom document structures by registering new components:

```python
from reader_vl.docs.structure.core import StructureBase
from reader_vl.docs.structure.registry import register_class

@register_class(12)
class CustomComponent(StructureBase):
    @property
    def label(self):
        return "CUSTOM_COMPONENT"

    def get_content(self, image):
        return "Custom content extraction logic here."
```

## CLI Usage

Reader VL also includes a command-line interface:

```bash
reader_vl parse --file sample.pdf --output result.json
```

## Contributing

We welcome contributions! Feel free to submit pull requests, report issues, or suggest new features.

### Development Setup

```bash
git clone https://github.com/Aquos06/reader_vl.git

cd reader_vl
pip install -r requirements.txt
```

## License

**Reader VL** is released under the MIT License.

## Contact
For questions or support, open an issue on GitHub or reach out at [ruliciohansen@gmail.com].

