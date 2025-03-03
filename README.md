# Reader VL

![PyPI](https://img.shields.io/pypi/v/reader_vl) ![License](https://img.shields.io/github/license/yourusername/reader_vl) ![Python](https://img.shields.io/badge/python-3.8%2B-blue)

## Overview

**Reader VL** goes beyond simple parsing. Our SDK and CLI empower you to build intelligent applications by converting diverse document formats (PDF, DOCX) into a unified structure. Critically, we leverage multimodal LLMs to enrich the parsed content, adding layers of meaning and context essential for maximizing the performance of your generative AI pipelines.

## Features

- 📄 **Multi-format Support**: Process PDFs and DOCX (for now).
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
from pathlib import Path
from reader_vl.document_reader import PDFReader 
from reader_vl.llm.open_ai import OpenAIClient
from reader_vl.document_reader import DocxReader

# Initialize OpenAI client
llm = OpenAIClient(api_key="your-api-key", model="gpt-4")

# Load a PDF file
parsed_doc = PDFreader(llm=llm, file_path="sample.pdf")
# or Docx file
parsed_doc = DocxReader(llm=llm, file_path="sample.docx")

# Display structured output
print(parsed_document.export_to_json())
# or
parsed_document.get_content() 
```

Alternatively, if you're using a self-hosted LLM:

```python
from reader_vl.llm.vllm import VllmClient

# Initialize VLLM client
llm = VllmClient(url="http://localhost:8000", model="custom-llm", temperature=0.7, max_tokens=512)

# Load and parse the document
parsed_doc = DocReader(llm=llm, file_path="sample.pdf")
print(parsed_document.export_to_json())
# or
parsed_doc.get_content()
```

### Asynchronous Parsing

```python
import asyncio
from reader_vl.document_reader import PDFReader

async def main():
    reader = PDFReader(llm=llm, file_path="sample.pdf", auto_parse = False)
    parsed_doc = await reader.aparse()
    print(parsed_doc)

asyncio.run(main())
```
#### Features:
- Works with self-hosted LLMs via REST APIs.
- Supports streaming responses for efficiency.
- Provides synchronous and asynchronous support.

## Components

**Reader VL** extracts structured components from documents, such as:

- **Header** 
- **Title** 
- **Image** 
- **Section** 
- **Table** 
- **List** 
- **Equation**
- **Chart** 
- **Footer** 
- **Reference** (Reference of a paper)
- **Figure Caption**
- **Table Caption**

## Architecture

**Reader VL** follows a modular approach:

- **YOLO-based Object Detection**: Detects document elements (headers, sections, tables, etc.).
- **OCR (Tesseract)**: Extracts text from detected regions.
- **Multimodal LLM Processing**: Enhances extracted content for AI pipelines.
- **Schema-based Output**: Provides structured data (JSON-like `Document` schema).

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
