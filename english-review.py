#!/usr/bin/env python3
"""
Spoken English Evaluation Pipeline

Evaluates spoken English proficiency using:
- Whisper (ASR) for speech-to-text transcription
- LLM (DeepSeek, OpenAI, or local) for linguistic evaluation

Provides detailed feedback on:
- Pronunciation issues (based on ASR confidence and common learner challenges)
- Fluency breakdowns (repetitions, self-corrections, run-on sentences)
- Grammar specifics (exact errors with corrections)
- Vocabulary feedback
- CEFR level estimation and practice recommendations

Usage:
    # Using MetaCentrum AIaaS (default)
    export E_INFRA_API_TOKEN="your-token"
    python english-review.py

    # Using OpenAI
    export OPENAI_API_KEY="your-key"
    python english-review.py --provider openai --model gpt-4o

    # Using local Ollama
    python english-review.py --provider ollama --model llama3.2

Requirements:
    pip install -r requirements.txt
"""

import os
import argparse
import datetime
from pathlib import Path

import librosa
import torch
from dotenv import load_dotenv
from jiwer import wer
from openai import OpenAI
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Load environment variables from .env file
load_dotenv()


# Supported providers
PROVIDERS = ["metacentrum", "openai", "ollama", "vllm"]


def create_llm_client(provider: str, model: str, api_key: str | None = None, base_url: str | None = None) -> OpenAI:
    """
    Create an LLM client based on the selected provider.

    Args:
        provider: One of 'metacentrum', 'openai', 'ollama', 'vllm'
        model: Model name
        api_key: API key (required for metacentrum and openai)
        base_url: Custom base URL (optional, used for ollama/vllm)

    Returns:
        Configured OpenAI-compatible client
    """
    resolved_model = model

    if provider == "metacentrum":
        token = api_key or os.getenv("E_INFRA_API_TOKEN")
        if not token:
            raise ValueError("E_INFRA_API_TOKEN not found. Set it via env var or --api-key flag.")
        return OpenAI(
            api_key=token,
            base_url="https://llm.ai.e-infra.cz/v1",
        )

    elif provider == "openai":
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not found. Set it via env var or --api-key flag.")
        return OpenAI(api_key=key)

    elif provider == "ollama":
        url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        return OpenAI(
            api_key="ollama",  # Placeholder, not used by Ollama
            base_url=url,
        )

    elif provider == "vllm":
        url = base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        key = api_key or os.getenv("VLLM_API_KEY", "placeholder")
        return OpenAI(
            api_key=key,
            base_url=url,
        )

    else:
        raise ValueError(f"Unknown provider: {provider}. Choose from {PROVIDERS}")


class SpokenEnglishEvaluationPipeline:
    """
    Pipeline for evaluating spoken English proficiency.

    Attributes:
        llm_client: Configured LLM client for linguistic evaluation
        llm_model: Model name used for evaluation
        whisper_processor: Whisper tokenizer/feature extractor
        whisper_model: Whisper model for ASR
    """

    def __init__(
        self,
        provider: str = "metacentrum",
        model: str = "deepseek-v4-pro-thinking",
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        """
        Initialize the evaluation pipeline.

        Args:
            provider: LLM provider ('metacentrum', 'openai', 'ollama', 'vllm')
            model: Model name or alias for linguistic evaluation
            api_key: API key for the provider
            base_url: Custom base URL for local providers
        """
        print("Loading models from Hugging Face...")

        self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")

        self.llm_client = create_llm_client(provider, model, api_key, base_url)
        self.llm_model = model
        print(f"Using LLM: {self.llm_model} via {provider}")

    def process_audio(self, audio_path: str, expected_text: str | None = None) -> dict:
        """
        Process an audio file and return comprehensive English evaluation.

        Args:
            audio_path: Path to .wav audio file
            expected_text: Optional expected transcript for WER calculation

        Returns:
            Dictionary with transcription, acoustic feedback, and linguistic evaluation
        """
        audio_input, sample_rate = librosa.load(audio_path, sr=16000)

        print("Transcribing speech to text...")
        input_features = self.whisper_processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_features

        generated_tokens = self.whisper_model.generate(
            input_features,
            language="en",
            task="transcribe",
            max_new_tokens=256,
        )

        transcription = self.whisper_processor.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()

        if not transcription:
            raise ValueError("No speech detected or transcription failed.")

        print(f"Transcription: '{transcription}'")

        word_segments: list[dict] = []
        low_confidence_words: list[dict] = []

        if expected_text:
            print("Calculating Word Error Rate against expected text...")
            word_error_rate = wer(expected_text, transcription)
            pronunciation_score = max(0, 100 - (word_error_rate * 100))
        else:
            word_error_rate = None
            pronunciation_score = 100.0 if not low_confidence_words else 100 - (len(low_confidence_words) * 5)

        pronunciation_class = f"Pronunciation Score: {pronunciation_score:.1f}/100"
        if low_confidence_words:
            pronunciation_class += f" ({len(low_confidence_words)} words with low ASR confidence)"

        print(f"Performing detailed linguistic evaluation with {self.llm_model}...")
        linguistic_evaluation = self._evaluate_spoken_english(transcription, word_segments, low_confidence_words, expected_text)

        return {
            "audio_file": audio_path,
            "transcription": transcription,
            "word_segments": word_segments,
            "low_confidence_words": low_confidence_words,
            "expected_text": expected_text,
            "word_error_rate": word_error_rate,
            "acoustic_feedback": pronunciation_class,
            "linguistic_evaluation": linguistic_evaluation,
        }

    def _evaluate_spoken_english(
        self,
        transcript: str,
        word_segments: list[dict],
        low_confidence_words: list[dict],
        expected_text: str | None = None,
    ) -> str:
        """Generate detailed linguistic evaluation using the LLM."""
        system_prompt = """You are an expert certified English language examiner (IELTS/CEFR).
Your task is to provide detailed, actionable feedback on spoken English. Focus on specific examples rather than general scores."""

        word_analysis = ""
        if low_confidence_words:
            words_list = ", ".join([f"'{w['word']}'" for w in low_confidence_words[:8]])
            word_analysis = f"\n\n**Words with potential pronunciation issues (low ASR confidence):** {words_list}"

        expected_analysis = ""
        if expected_text:
            expected_analysis = f"""
**Expected/Target Text:**
"{expected_text}"

Compare the student's actual speech against this target text."""

        user_prompt = f"""Analyze the following student's spoken English and provide detailed, specific feedback.

**Transcript:**
"{transcript}"

{word_analysis}
{expected_analysis}

**Provide specific feedback in these areas:**

## 1. Pronunciation Issues
Based on the transcript and any flagged words, identify words that are commonly mispronounced by learners or may have been pronounced poorly:
- List each potentially problematic word from the transcript
- Explain the likely pronunciation issue (stress pattern, vowel sound, consonant cluster, silent letters, etc.)
- Provide correct pronunciation guidance using phonetic spelling or syllable breakdown

If low-confidence words were detected during transcription, those likely indicate pronunciation difficulties - address them specifically. For other complex words in the transcript, note common pronunciation challenges learners face.

## 2. Fluency Analysis
Identify specific moments where fluency was disrupted:
- Repetitions (e.g., "I I think...")
- Self-corrections (e.g., "She go... goes to school")
- Fillers or hesitations evident in the speech pattern
- Run-on sentences reducing clarity

Quote the exact phrases where these occur.

## 3. Grammar Specifics
Instead of a score, point out:
- Exact sentences with grammatical errors
- What the error is
- How to correct it

If grammar is strong, highlight what makes it good.

## 4. Vocabulary Feedback
- Highlight excellent word choices
- Point out any awkward or incorrect word usage with better alternatives
- Note variety and appropriateness of vocabulary

## 5. Overall Assessment
- Estimated CEFR level (A1-C2) with justification
- Top 3 strengths
- Top 3 priority areas for improvement
- 2-3 specific practice exercises targeting identified weaknesses

Be constructive, specific, and actionable. Quote exact words/phrases from the transcript when giving feedback.
"""

        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=2500,
        )
        return response.choices[0].message.content

    def generate_report(self, results: dict) -> str:
        """Format evaluation results into a Markdown report."""
        provider_url = str(self.llm_client.base_url)

        report = f"""# Spoken English Evaluation Report

| Field | Value |
|-------|-------|
| **File** | `{results['audio_file']}` |
| **Model** | `{self.llm_model}` |
| **Provider** | `{provider_url}` |

---

## 🎤 Transcription

> {results['transcription']}

"""
        if results.get("expected_text"):
            report += f"""
## 📝 Expected Text

> {results['expected_text']}

**Word Error Rate**: {results['word_error_rate']:.1%}

"""
        report += f"""
## 🔊 Acoustic Analysis

{results['acoustic_feedback']}

"""
        if results.get("low_confidence_words"):
            words = ", ".join([f"`{w['word']}`" for w in results['low_confidence_words']])
            report += f"""
**Low confidence words**: {words}

"""
        report += f"""
---

## 📋 Detailed Feedback

{results['linguistic_evaluation']}

---

*Generated by {self.llm_model} via {provider_url}*
"""
        return report


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Evaluate spoken English proficiency using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using MetaCentrum AIaaS (default)
  export E_INFRA_API_TOKEN="your-token"
  python english-review.py data/Good.wav

  # Using OpenAI
  export OPENAI_API_KEY="your-key"
  python english-review.py data/Good.wav --provider openai --model gpt-4o

  # Using local Ollama
  python english-review.py data/Good.wav --provider ollama --model llama3.2

  # With expected text for WER calculation
  python english-review.py data/Good.wav --expected-text "Hello, I'm a student..."

  # Save to specific output file
  python english-review.py data/Good.wav --output my_report.txt
        """,
    )

    parser.add_argument("audio_file", nargs="?", default="data/Good.wav", help="Path to .wav audio file (default: data/Good.wav)")
    parser.add_argument("--provider", "-p", choices=PROVIDERS, default="metacentrum", help="LLM provider (default: metacentrum)")
    parser.add_argument("--model", "-m", default="deepseek-v4-pro-thinking", help="Model name or alias (default: deepseek-v4-pro-thinking)")
    parser.add_argument("--api-key", "-k", help="API key for the provider")
    parser.add_argument("--base-url", "-b", help="Custom base URL for local providers")
    parser.add_argument("--expected-text", "-e", help="Expected transcript for WER calculation")
    parser.add_argument("--output", "-o", help="Output file path (default: results/evaluation_TIMESTAMP.txt)")
    parser.add_argument("--results-dir", "-d", default="results", help="Results directory (default: results)")

    args = parser.parse_args()

    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = results_dir / output_path
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"evaluation_{timestamp}.md"

    # Initialize pipeline
    try:
        pipeline = SpokenEnglishEvaluationPipeline(
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
        )
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        return 1

    # Process audio
    if not os.path.exists(args.audio_file):
        print(f"Audio file '{args.audio_file}' not found.")
        return 1

    try:
        results = pipeline.process_audio(args.audio_file, args.expected_text)
        report = pipeline.generate_report(results)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"Results saved to: {output_path}")
        return 0

    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
