import torch


DEFAULT_EXPERT_ROLE_PROMPT = (
    "You are an expert olympiad mathematician. Solve the problem with rigorous, "
    "independent reasoning. Identify the key structure before calculating, verify "
    "each non-trivial step, reconsider doubtful branches, and check the final answer."
)

NEUTRAL_CONTROL_PROMPT = (
    "You are an assistant. Respond to the user's request clearly and carefully. "
    "Follow the requested output format, keep the response relevant, and review the "
    "answer before finishing."
)


class SelfDistillationDataCollator:
    """
    Data collator for self-distillation that creates both student and teacher inputs.

    Student: sees the problem, optionally with a system context.
    Teacher: sees the problem plus either the reference solution, no system
    context, or a teacher-only system context (with chat template).

    To enable batch-level operations (like original GKD), we pad prompts to the same length
    within each batch, and track the actual (unpadded) prompt lengths for loss masking.
    """

    def __init__(
        self,
        tokenizer,
        max_length=2048,
        reason_first=True,
        student_thinking=False,
        teacher_thinking=True,
        student_context_mode="none",
        student_role_prompt=None,
        teacher_context_mode="solution",
        teacher_role_prompt=None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.reason_first = reason_first
        self.student_thinking = student_thinking
        self.teacher_thinking = teacher_thinking
        self.student_context_mode = student_context_mode
        self.student_role_prompt = student_role_prompt or DEFAULT_EXPERT_ROLE_PROMPT
        self.teacher_context_mode = teacher_context_mode
        self.teacher_role_prompt = teacher_role_prompt or DEFAULT_EXPERT_ROLE_PROMPT

        student_context_modes = {"none", "role", "neutral"}
        if self.student_context_mode not in student_context_modes:
            raise ValueError(
                f"student_context_mode must be one of {sorted(student_context_modes)}, "
                f"got {self.student_context_mode!r}."
            )
        teacher_context_modes = {"solution", "none", "role", "neutral"}
        if self.teacher_context_mode not in teacher_context_modes:
            raise ValueError(
                f"teacher_context_mode must be one of {sorted(teacher_context_modes)}, "
                f"got {self.teacher_context_mode!r}."
            )
        if self.reason_first and self.teacher_context_mode != "solution":
            raise ValueError("reason_first is only supported with teacher_context_mode='solution'.")

        # Prompt for reasoning about the solution before teaching
        self.reason_first_prompt = (
            "\n\nThe reference reasoning above arrives at the correct answer. "
            "Please analyze this solution and explain the key reasoning steps and problem-solving strategies employed. "
            "Do NOT use <think> tags. Do NOT derive your own solution. "
            "Simply analyze and explain the reference solution provided above.\n"
        )
        # Prompt for transitioning to teaching mode after reasoning
        self.transition_prompt = (
            "\n\nAfter reading the reference solution above, make sure you truly understand "
            "the reasoning behind each step — do not copy or paraphrase it. Now, using your "
            "own words and independent reasoning, derive the same final answer to the problem above. "
            "Think step by step, explore different approaches, and don't be afraid to backtrack "
            "or reconsider if something doesn't work out:\n"
        )

        # Set padding side explicitly for consistency
        print(f"[DataCollator] Original padding_side: {self.tokenizer.padding_side}")
        self.tokenizer.padding_side = "right"
        print(f"[DataCollator] Set padding_side to: {self.tokenizer.padding_side}")
        print(f"[DataCollator] Reason first mode: {self.reason_first}")
        print(f"[DataCollator] Student context mode: {self.student_context_mode}")
        print(f"[DataCollator] Teacher context mode: {self.teacher_context_mode}")

    def __call__(self, features):

        batch_size = len(features)

        # Prepare student and teacher prompts using chat template (matching evaluation)
        student_prompts = []
        teacher_prompts = []
        teacher_reasoning_prompts = []  # NEW: for reason_first mode

        for feature in features:
            # Extract problem and solution from dataset
            # Handle different possible column names
            problem = feature["problem"]
            solution = feature["solution"]

            # Student prompt: just the problem with instruction (matching evaluation format)
            student_user_message = f"Problem: {problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
            if self.student_context_mode == "none":
                student_messages = [{"role": "user", "content": student_user_message}]
            else:
                student_system_prompt = (
                    self.student_role_prompt
                    if self.student_context_mode == "role"
                    else NEUTRAL_CONTROL_PROMPT
                )
                student_messages = [
                    {"role": "system", "content": student_system_prompt},
                    {"role": "user", "content": student_user_message},
                ]

            # Apply chat template for student (matching evaluation)
            student_prompt = self.tokenizer.apply_chat_template(
                student_messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.student_thinking
            )
            student_prompts.append(student_prompt)

            if self.reason_first:
                # Reasoning prompt: ask teacher to analyze the solution
                reasoning_user_message = (
                    f"Problem: {problem}\n\n"
                    f"Here is a correct reasoning to this problem:"
                    f"=== Reference Reasoning Start ===\n"
                    f"{solution}\n"
                    f"=== Reference Reasoning End ===\n\n"
                    f"{self.reason_first_prompt}"
                )
                reasoning_messages = [{"role": "user", "content": reasoning_user_message}]
                reasoning_prompt = self.tokenizer.apply_chat_template(
                    reasoning_messages, tokenize=False, add_generation_prompt=True
                )
                teacher_reasoning_prompts.append(reasoning_prompt)

                # Teacher prompt will be constructed during training after reasoning
                # For now, create placeholder (will be replaced in training_step)
                teacher_prompts.append("")  # Placeholder
            else:
                if self.teacher_context_mode == "solution":
                    # Original OPSD teacher: sample-specific verified reasoning is privileged.
                    teacher_user_message = (
                        f"Problem: {problem}\n\n"
                        f"Here is a reference solution to this problem:\n"
                        f"=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n"
                        f"{self.transition_prompt}\n"
                        f"Please reason step by step, and put your final answer within \\boxed{{}}."
                    )
                    teacher_messages = [{"role": "user", "content": teacher_user_message}]
                elif self.teacher_context_mode == "none":
                    # Reverse-locus ablation: the role is visible to the student
                    # rollout, while the fixed teacher receives the deployment
                    # prompt with no role or reference answer.
                    teacher_messages = [{"role": "user", "content": student_user_message}]
                else:
                    # Role-privileged OPSD: the teacher sees no answer or reference trace.
                    # Only its system context differs from the inference-time student.
                    system_prompt = (
                        self.teacher_role_prompt
                        if self.teacher_context_mode == "role"
                        else NEUTRAL_CONTROL_PROMPT
                    )
                    teacher_messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": student_user_message},
                    ]

                # Apply chat template for teacher
                teacher_prompt = self.tokenizer.apply_chat_template(
                    teacher_messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.teacher_thinking
                )
                teacher_prompts.append(teacher_prompt)

        # Tokenize WITHOUT padding first to get true lengths
        student_encoded_no_pad = self.tokenizer(
            student_prompts,
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        student_prompt_lengths = [len(ids) for ids in student_encoded_no_pad["input_ids"]]

        # Find max lengths in this batch
        max_student_prompt_len = max(student_prompt_lengths)

        # Tokenize WITH padding to max length in batch
        student_encoded = self.tokenizer(
            student_prompts,
            padding="max_length",
            truncation=True,
            max_length=max_student_prompt_len,
            return_tensors="pt",
        )

        result = {
            "student_prompts": student_encoded["input_ids"],
            "student_prompt_attention_mask": student_encoded["attention_mask"],
            "student_prompt_length": max_student_prompt_len,  # Single value for batch!
            # Keep individual lengths for proper masking
            "student_prompt_lengths_per_example": torch.tensor(student_prompt_lengths),
        }

        if self.reason_first:
            # Tokenize reasoning prompts
            reasoning_encoded_no_pad = self.tokenizer(
                teacher_reasoning_prompts,
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )
            reasoning_prompt_lengths = [len(ids) for ids in reasoning_encoded_no_pad["input_ids"]]
            max_reasoning_prompt_len = max(reasoning_prompt_lengths)

            reasoning_encoded = self.tokenizer(
                teacher_reasoning_prompts,
                padding="max_length",
                truncation=True,
                max_length=max_reasoning_prompt_len,
                return_tensors="pt",
            )

            # Tokenize transition prompt (this will be appended after reasoning)
            # Don't use chat template here - just the raw text
            transition_text = f"\n{self.transition_prompt}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
            transition_encoded = self.tokenizer(
                [transition_text] * batch_size,
                padding=False,
                truncation=False,
                return_tensors="pt",
            )

            result.update(
                {
                    "teacher_reasoning_prompts": reasoning_encoded["input_ids"],
                    "teacher_reasoning_attention_mask": reasoning_encoded["attention_mask"],
                    "teacher_reasoning_prompt_length": max_reasoning_prompt_len,
                    "teacher_transition_tokens": transition_encoded["input_ids"],
                }
            )
        else:
            # Normal mode: tokenize teacher prompts
            teacher_encoded_no_pad = self.tokenizer(
                teacher_prompts,
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )
            teacher_prompt_lengths = [len(ids) for ids in teacher_encoded_no_pad["input_ids"]]
            max_teacher_prompt_len = max(teacher_prompt_lengths)

            teacher_encoded = self.tokenizer(
                teacher_prompts,
                padding="max_length",
                truncation=True,
                max_length=max_teacher_prompt_len,
                return_tensors="pt",
            )

            result.update(
                {
                    "teacher_prompts": teacher_encoded["input_ids"],
                    "teacher_prompt_attention_mask": teacher_encoded["attention_mask"],
                    "teacher_prompt_length": max_teacher_prompt_len,
                    "teacher_prompt_lengths_per_example": torch.tensor(teacher_prompt_lengths),
                }
            )

        return result
