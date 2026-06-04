# Spoken English Evaluation Pipeline

Automated evaluation of spoken English proficiency using AI. Provides detailed feedback on pronunciation, fluency, grammar, and vocabulary based on IELTS/CEFR criteria.

## Features

- **Speech-to-Text**: Whisper medium model for accurate transcription
- **Linguistic Analysis**: Detailed feedback powered by LLMs
- **Multiple LLM Providers**: Support for MetaCentrum AIaaS, OpenAI, Ollama, vLLM
- **WER Calculation**: Optional word error rate against expected text
- **Structured Reports**: Comprehensive evaluation with actionable recommendations

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Using MetaCentrum AIaaS (Default)

```bash
# Set your API token
export E_INFRA_API_TOKEN="your-token-here"

# Run evaluation
python english-review.py data/Good.wav
```

### Using OpenAI

```bash
export OPENAI_API_KEY="your-key-here"
python english-review.py data/Good.wav --provider openai --model gpt-4o
```

### Using Local Ollama

```bash
# First, start Ollama with a model
ollama pull llama3.2
ollama serve

# Run evaluation
python english-review.py data/Good.wav --provider ollama --model llama3.2
```

### Using vLLM

```bash
python english-review.py data/Good.wav --provider vllm --model mistral --base-url http://localhost:8000/v1
```

## Command Line Reference

```
usage: python english-review.py [OPTIONS] [AUDIO_FILE]

positional arguments:
  audio_file            Path to .wav audio file (default: data/Good.wav)

options:
  -p, --provider        LLM provider: metacentrum, openai, ollama, vllm
  -m, --model           Model name or alias (default: deepseek-v4-pro-thinking)
  -k, --api-key         API key for the provider
  -b, --base-url        Custom base URL for local providers
  -e, --expected-text   Expected transcript for WER calculation
  -o, --output          Output file path
  -d, --results-dir     Results directory (default: results)
  -h, --help            Show help message
```

## Provider Configuration

| Provider | Environment Variable | Default Model | Notes |
|----------|---------------------|---------------|-------|
| metacentrum | `E_INFRA_API_TOKEN` | deepseek-v4-pro-thinking | Czech research infrastructure |
| openai | `OPENAI_API_KEY` | gpt-4o | Requires paid account |
| ollama | `OLLAMA_BASE_URL` | llama3.2 | Local, free |
| vllm | `VLLM_BASE_URL` | mistral | Local inference server |


## Output

Results are saved to `results/evaluation_YYYYMMDD_HHMMSS.txt` containing:

- **Transcription**: The recognized speech
- **Acoustic Analysis**: Pronunciation score
- **Pronunciation Issues**: Specific words with guidance
- **Fluency Analysis**: Repetitions, self-corrections, run-ons
- **Grammar Specifics**: Exact errors with corrections
- **Vocabulary Feedback**: Strengths and improvements
- **Overall Assessment**: CEFR level, strengths, priorities, exercises

## Example Usage

```bash
# Basic usage with default settings
python english-review.py data/Good.wav

# With expected text for accuracy measurement
python english-review.py data/test.wav \
    --expected-text "Hello, I am a student at the university."

# Using a specific model
python english-review.py data/test.wav \
    --provider openai \
    --model gpt-4o-mini

# Custom output location
python english-review.py data/test.wav \
    --output reports/my_evaluation.txt
```

## Project Structure

```
.
├── english-review.py      # Main evaluation script
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── secrets_manifest.md   # Secrets management guide
├── .env                  # API tokens (gitignored)
├── .gitignore            # Git ignore rules
├── venv/                 # Virtual environment
├── data/                 # Audio files
│   └── Good.wav
└── results/              # Evaluation outputs
    └── evaluation_*.txt
```
