#!/usr/bin/env python3
"""
llm_engine.py — Local HF LLM engine (free)

Replaces OpenAI gpt-4o-mini with a local, free model:
    microsoft/Phi-3-mini-4k-instruct

Dependencies (in retrieval_env):
    pip install "transformers>=4.40.0" sentencepiece accelerate

Usage (quick test):
    python Model/llm_engine.py
"""

import textwrap
from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# --------- CONFIG ---------
HF_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"  # already downloaded on your machine
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.2
TOP_P = 0.9


@dataclass
class LLMAnswer:
    answer: str
    used_tokens: int
    model_name: str
    prompt: str


class LLMEngine:
    def __init__(self, model_name: str = HF_MODEL_NAME):
        """
        Initialize a local HF causal LLM for NCERT Q&A (free, no API keys).
        """
        self.model_name = model_name

        print(f"[LLMEngine] Loading local HF model: {model_name}")

        # Pick device
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            # CPU fallback
            self.device = "cpu"

        # Load tokenizer & model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"[LLMEngine] Model loaded on device: {self.device}")

    # ------------------ PROMPT TEMPLATE ------------------ #
    def build_prompt(self, question: str, passages: List[str]) -> str:
        """
        Build a NCERT-grounded prompt.
        """
        # Join passages as context with numbered bullets
        context_blocks = []
        for i, p in enumerate(passages, start=1):
            # truncate very long passages just to be safe
            p_short = p.strip()
            if len(p_short) > 1200:
                p_short = p_short[:1200] + " ..."
            context_blocks.append(f"[PASSAGE {i}]\n{p_short}")
        context_text = "\n\n".join(context_blocks) if context_blocks else "No context available."

        system_instructions = textwrap.dedent(
            """
            You are an AI tutor for Indian school students, explaining topics strictly from NCERT textbooks.
            Rules:
            - Answer clearly, step-by-step, in simple language.
            - Base your answer ONLY on the provided passages.
            - If the passages do not fully answer the question, say so and mention what is missing.
            - Keep the answer short and focused; do not invent content outside NCERT.
            - Where relevant, refer to the passages as [PASSAGE 1], [PASSAGE 2], etc.
            """
        ).strip()

        user_prompt = textwrap.dedent(
            f"""
            {system_instructions}

            ==== NCERT PASSAGES ====
            {context_text}

            ==== QUESTION ====
            {question}

            ==== TASK ====
            Using ONLY the information from the NCERT passages above, write a clear, student-friendly answer.
            If the answer is not fully covered, clearly mention which part is missing.
            """
        ).strip()

        return user_prompt

    # ------------------ GENERATION ------------------ #
    def generate(self, prompt: str) -> str:
        """
        Run the local HF model with the given prompt and return the generated text.
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # The model will echo the prompt; we just return the part after the prompt
        if full_text.startswith(prompt):
            answer = full_text[len(prompt) :].strip()
        else:
            # Fallback: try to split on TASK
            split_tok = "==== TASK ===="
            if split_tok in full_text:
                answer = full_text.split(split_tok)[-1].strip()
            else:
                answer = full_text.strip()

        return answer

    # ------------------ PUBLIC API ------------------ #
    def answer_question(
        self,
        question: str,
        passages: Optional[List[str]] = None,
    ) -> LLMAnswer:
        """
        Main entrypoint used by the rest of the system.
        `passages` should be a list of strings from BM25 retrieval.
        """
        passages = passages or []
        prompt = self.build_prompt(question, passages)
        answer_text = self.generate(prompt)

        # crude token count
        used_tokens = len(self.tokenizer.encode(prompt + answer_text))

        return LLMAnswer(
            answer=answer_text,
            used_tokens=used_tokens,
            model_name=self.model_name,
            prompt=prompt,
        )


# ------------- QUICK MANUAL TEST ------------- #
if __name__ == "__main__":
    engine = LLMEngine()

    demo_question = "What is photosynthesis?"
    demo_passages = [
        "This process by which plants prepare food in the presence of sunlight and chlorophyll is called photosynthesis. "
        "A leaf is the primary site for photosynthesis. Water, sunlight, carbon dioxide from the air, and chlorophyll are necessary to carry out this process."
    ]

    result = engine.answer_question(demo_question, demo_passages)
    print("\n=== MODEL ANSWER ===")
    print(result.answer)
    print("\n[Meta]")
    print(f"Model: {result.model_name}")
    print(f"Approx. tokens used: {result.used_tokens}")
