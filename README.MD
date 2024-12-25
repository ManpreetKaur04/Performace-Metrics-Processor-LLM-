# Company Metrics Query Processor

A Python application that uses Large Language Models (LLM) to process natural language queries about company performance metrics and convert them into structured JSON format.

## Features

- Natural language processing of company metric queries using Groq's LLM
- Support for multiple company comparisons
- Flexible date parsing including relative dates
- Conversation history management
- ISO 8601 date format standardization
- Comprehensive error handling

## Project Structure

```
company-metrics-query-processor/
├── README.md
├── requirements.txt
├── .env
├── main.py
├── static/
│   ├── styles.css
├── venv/
│   └── ..
└── templates/
    ├── index.html
    
```

## Dependencies

### Dependencies
```
flask=3.1.0
flask-Cors=5.0.0
groq==0.4.0
python-dateutil==2.8.2
python-dotenv==1.0.0
pytest==7.4.3
black==23.12.1
isort==5.13.2
jupyter==1.0.0
pre-commit==3.5.0
pytest-cov==4.1.0
mypy==1.7.1
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/company-metrics-processor.git
cd company-metrics-processor
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
 touch .env
# Edit .env and add your Groq API key
```

## Environment Variables
```
GROQ_API_KEY=your-groq-api-key
MODEL_NAME=llama-3.1-8b-instant
MAX_HISTORY=6
```

