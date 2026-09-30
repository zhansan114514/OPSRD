import unittest

import torch

from data_collator import DEFAULT_EXPERT_ROLE_PROMPT, SelfDistillationDataCollator
from opsd_trainer import OPSDTrainer


class RecordingTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    padding_side = "left"

    def __init__(self):
        self.chat_calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        self.chat_calls.append((messages, kwargs))
        rendered = "\n".join(f"<{message['role']}>{message['content']}" for message in messages)
        return rendered + ("\n<assistant>" if add_generation_prompt else "")

    def __call__(self, texts, padding=False, truncation=False, max_length=None, return_tensors=None):
        encoded = []
        for text in texts:
            ids = [2 + (ord(char) % 89) for char in text]
            if truncation and max_length is not None:
                ids = ids[:max_length]
            encoded.append(ids)

        if padding == "max_length":
            target_length = max_length
            encoded = [ids + [self.pad_token_id] * (target_length - len(ids)) for ids in encoded]

        attention_mask = [[int(token != self.pad_token_id) for token in ids] for ids in encoded]
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(encoded, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }
        return {"input_ids": encoded, "attention_mask": attention_mask}


class PromptIsolationTests(unittest.TestCase):
    feature = {"problem": "What is 1+1?", "solution": "SECRET_SOLUTION: 2"}

    def test_role_teacher_receives_role_but_not_solution(self):
        tokenizer = RecordingTokenizer()
        collator = SelfDistillationDataCollator(
            tokenizer,
            reason_first=False,
            teacher_context_mode="role",
        )
        collator([self.feature])

        teacher_messages = tokenizer.chat_calls[1][0]
        self.assertEqual(teacher_messages[0], {"role": "system", "content": DEFAULT_EXPERT_ROLE_PROMPT})
        self.assertNotIn("SECRET_SOLUTION", str(teacher_messages))
        self.assertEqual(teacher_messages[1]["role"], "user")
        self.assertIn(self.feature["problem"], teacher_messages[1]["content"])

    def test_role_guided_student_receives_role_before_rollout(self):
        tokenizer = RecordingTokenizer()
        collator = SelfDistillationDataCollator(
            tokenizer,
            reason_first=False,
            student_context_mode="role",
            teacher_context_mode="role",
        )
        collator([self.feature])

        student_messages = tokenizer.chat_calls[0][0]
        self.assertEqual(student_messages[0], {"role": "system", "content": DEFAULT_EXPERT_ROLE_PROMPT})
        self.assertNotIn("SECRET_SOLUTION", str(student_messages))
        self.assertEqual(student_messages[1]["role"], "user")

    def test_student_only_ablation_keeps_teacher_role_free(self):
        tokenizer = RecordingTokenizer()
        collator = SelfDistillationDataCollator(
            tokenizer,
            reason_first=False,
            student_context_mode="role",
            teacher_context_mode="none",
        )
        collator([self.feature])

        student_messages = tokenizer.chat_calls[0][0]
        teacher_messages = tokenizer.chat_calls[1][0]
        self.assertEqual(student_messages[0], {"role": "system", "content": DEFAULT_EXPERT_ROLE_PROMPT})
        self.assertEqual(len(teacher_messages), 1)
        self.assertEqual(teacher_messages[0]["role"], "user")
        self.assertNotIn(DEFAULT_EXPERT_ROLE_PROMPT, str(teacher_messages))
        self.assertNotIn("SECRET_SOLUTION", str(teacher_messages))

    def test_default_student_does_not_receive_system_role(self):
        tokenizer = RecordingTokenizer()
        collator = SelfDistillationDataCollator(
            tokenizer,
            reason_first=False,
            teacher_context_mode="role",
        )
        collator([self.feature])

        student_messages = tokenizer.chat_calls[0][0]
        self.assertEqual(student_messages[0]["role"], "user")
        self.assertNotIn(DEFAULT_EXPERT_ROLE_PROMPT, str(student_messages))

    def test_solution_teacher_receives_reference_solution(self):
        tokenizer = RecordingTokenizer()
        collator = SelfDistillationDataCollator(
            tokenizer,
            reason_first=False,
            teacher_context_mode="solution",
        )
        collator([self.feature])

        teacher_messages = tokenizer.chat_calls[1][0]
        self.assertIn("SECRET_SOLUTION", teacher_messages[0]["content"])

    def test_reason_first_rejects_role_context(self):
        with self.assertRaisesRegex(ValueError, "only supported"):
            SelfDistillationDataCollator(
                RecordingTokenizer(),
                reason_first=True,
                teacher_context_mode="role",
            )


class EntropyMaskTests(unittest.TestCase):
    def test_generalized_jsd_endpoints_and_midpoint_are_nonnegative(self):
        student_logits = torch.tensor([[[0.5, -0.3, 0.1]]], dtype=torch.float64)
        teacher_logits = torch.tensor([[[-0.2, 0.7, 0.0]]], dtype=torch.float64)
        labels = torch.tensor([[1]])

        forward = OPSDTrainer.generalized_jsd_loss(
            student_logits, teacher_logits, labels=labels, beta=0, token_clip=None
        )
        jsd = OPSDTrainer.generalized_jsd_loss(
            student_logits, teacher_logits, labels=labels, beta=0.5, token_clip=None
        )
        reverse = OPSDTrainer.generalized_jsd_loss(
            student_logits, teacher_logits, labels=labels, beta=1, token_clip=None
        )

        self.assertGreater(forward.item(), 0)
        self.assertGreater(jsd.item(), 0)
        self.assertGreater(reverse.item(), 0)
        self.assertFalse(torch.allclose(forward, reverse))

    def test_top_entropy_fraction_respects_labels_and_sequence_boundaries(self):
        # Sequence 0 has three valid positions with increasing entropy; sequence 1 has two.
        logits = torch.tensor(
            [
                [[9.0, 0.0], [3.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [8.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ]
        )
        labels = torch.tensor([[1, 1, 1, -100], [1, 1, -100, -100]])

        mask, entropy = OPSDTrainer.top_entropy_position_mask(logits, labels, fraction=0.5)

        self.assertEqual(mask.sum(dim=1).tolist(), [2, 1])
        self.assertTrue(mask[0, 2])
        self.assertTrue(mask[1, 0])
        self.assertFalse(mask[0, 3])
        self.assertGreater(entropy[0, 2], entropy[0, 0])

    def test_position_mask_blocks_unselected_gradients(self):
        student_logits = torch.tensor(
            [[[0.2, -0.2], [0.4, -0.4]]], dtype=torch.float32, requires_grad=True
        )
        teacher_logits = torch.tensor([[[1.0, -1.0], [-1.0, 1.0]]], dtype=torch.float32)
        labels = torch.tensor([[1, 1]])
        position_mask = torch.tensor([[True, False]])

        loss = OPSDTrainer.generalized_jsd_loss(
            student_logits,
            teacher_logits,
            labels=labels,
            beta=0,
            position_mask=position_mask,
        )
        loss.backward()

        self.assertGreater(student_logits.grad[0, 0].abs().sum().item(), 0)
        self.assertEqual(student_logits.grad[0, 1].abs().sum().item(), 0)

    def test_raw_divergence_metric_precedes_pointwise_clipping(self):
        student_logits = torch.tensor([[[0.2, -0.2], [0.4, -0.4]]], dtype=torch.float32)
        teacher_logits = torch.tensor([[[1.0, -1.0], [-1.0, 1.0]]], dtype=torch.float32)
        labels = torch.tensor([[1, 1]])

        plain_loss = OPSDTrainer.generalized_jsd_loss(
            student_logits,
            teacher_logits,
            labels=labels,
            beta=0,
            token_clip=0.01,
        )
        measured_loss, metrics = OPSDTrainer.generalized_jsd_loss(
            student_logits,
            teacher_logits,
            labels=labels,
            beta=0,
            token_clip=0.01,
            return_metrics=True,
        )

        self.assertTrue(torch.allclose(plain_loss, measured_loss))
        self.assertGreaterEqual(metrics["raw_position_divergence"].item(), 0)


if __name__ == "__main__":
    unittest.main()
