import torch
import torch.nn as nn
import torch.nn.functional as F


class LogitMSE(nn.Module):
    """
    Logit matching baseline.

    Uses CE in the trainer plus an MSE loss between student and teacher logits.
    """

    def __init__(self, teacher, student):
        super(LogitMSE, self).__init__()
        self.teacher = teacher
        self.student = student

        # Freeze teacher parameters.
        for param in self.teacher.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            teacher_logits = self.teacher(x)
        student_logits = self.student(x)
        logit_mse = F.mse_loss(student_logits, teacher_logits)
        return teacher_logits, student_logits, logit_mse
