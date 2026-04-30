import torch
import torch.nn as nn


class HardCE(nn.Module):
    """
    CE-only baseline for student training.

    Returns teacher logits for metric computation, but contributes no
    method-specific loss term.
    """

    def __init__(self, teacher, student):
        super(HardCE, self).__init__()
        self.teacher = teacher
        self.student = student

        # Freeze teacher parameters.
        for param in self.teacher.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            teacher_logits = self.teacher(x)
        student_logits = self.student(x)
        return teacher_logits, student_logits, None
