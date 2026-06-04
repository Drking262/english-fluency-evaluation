import os
import librosa
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from openai import OpenAI
from dotenv import load_dotenv
from jiwer import wer

load_dotenv()

llm_client = OpenAI(
    api_key=os.environ.get("E_INFRA_API_TOKEN", "your-api-key"),
    base_url="https://llm.ai.e-infra.cz/v1"
)

MODEL_ALIASES = {
    "mini": "gpt-oss-120b",
    "coder": "qwen3.5-122b",
    "agentic": "qwen3.5-122b",
    "thinker": "kimi-k2.6",
}

def resolve_model(model_name: str) -> str:
    return MODEL_ALIASES.get(model_name, model_name)


class SpokenEnglishEvaluationPipeline:

    def __init__(self, llm_model: str = "deepseek-v4-pro-thinking"):
        print("Loading models from Hugging Face...")

        self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")

        self.llm_model = resolve_model(llm_model)
        print(f"Using LLM: {self.llm_model}")

    def process_audio(self, audio_path: str, expected_text: str = None) -> dict:
        audio_input, sample_rate = librosa.load(audio_path, sr=16000)

        print("Transcribing speech to text...")
        input_features = self.whisper_processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_features

        generated_tokens = self.whisper_model.generate(
            input_features,
            language="en",
            task="transcribe",
            max_new_tokens=256
        )

        transcription = self.whisper_processor.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()

        if not transcription:
            raise ValueError("No speech detected or transcription failed.")

        print(f"Transcription: '{transcription}'")

        word_segments = []
        low_confidence_words = []

        if expected_text:
            print("Calculating Word Error Rate against expected text...")
            word_error_rate = wer(expected_text, transcription)
            pronunciation_score = max(0, 100 - (word_error_rate * 100))
        else:
            word_error_rate = None
            pronunciation_score = 100 - (len(low_confidence_words) * 5) if word_segments else 50

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
            "linguistic_evaluation": linguistic_evaluation
        }

    def _evaluate_spoken_english(self, transcript: str, word_segments: list, low_confidence_words: list, expected_text: str = None) -> str:
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

        response = llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=2500
        )
        return response.choices[0].message.content

    def generate_report(self, results: dict) -> str:
        report = f"""
{'='*70}
SPOKEN ENGLISH DETAILED EVALUATION REPORT
{'='*70}

File: {results['audio_file']}

{'-'*50}
TRANSCRIPTION
{'-'*50}
{results['transcription']}

"""
        if results.get('expected_text'):
            report += f"""
{'-'*50}
EXPECTED TEXT
{'-'*50}
{results['expected_text']}

"""
        report += f"""
{'-'*50}
ACOUSTIC ANALYSIS
{'-'*50}
{results['acoustic_feedback']}

"""
        if results.get('low_confidence_words'):
            report += f"""
Low confidence words: {[w['word'] for w in results['low_confidence_words']]}

"""
        report += f"""
{'-'*50}
DETAILED FEEDBACK
{'-'*50}
{results['linguistic_evaluation']}

{'='*70}
Generated by: {self.llm_model}
{'='*70}
"""
        return report


if __name__ == "__main__":
    import datetime

    RESULTS_DIR = "results"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"evaluation_{timestamp}.md"
    output_path = os.path.join(RESULTS_DIR, output_filename)

    pipeline = SpokenEnglishEvaluationPipeline(llm_model="deepseek-v4-pro-thinking")

    audio_file = "data/Wrong.wav"
    expected_text = None

    if os.path.exists(audio_file):
        try:
            results = pipeline.process_audio(audio_file, expected_text)
            report = pipeline.generate_report(results)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report)

            print(f"Results saved to: {output_path}")

        except Exception as e:
            print(f"Error during evaluation: {e}")
    else:
        print(f"Audio file '{audio_file}' not found.")
        print("Please provide a valid .wav file path for evaluation.")
