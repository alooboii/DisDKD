import torch
import torch.nn as nn
import torchvision.models as models


def _extract_state_dict(checkpoint_obj):
    """Extract a model state_dict from common checkpoint formats."""
    if not isinstance(checkpoint_obj, dict):
        return checkpoint_obj, None

    candidate_keys = [
        "state_dict",
        "model_state_dict",
        "model",
        "teacher_state_dict",
        "student_state_dict",
        "net",
    ]
    for key in candidate_keys:
        value = checkpoint_obj.get(key)
        if isinstance(value, dict):
            return value, key

    return checkpoint_obj, None


def _strip_common_prefixes(state_dict):
    """Strip common wrapping prefixes used by DataParallel/checkpoint wrappers."""
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        return state_dict

    keys = list(state_dict.keys())
    if all(k.startswith("module.") for k in keys):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    if all(k.startswith("model.") for k in keys):
        return {k[len("model.") :]: v for k, v in state_dict.items()}
    return state_dict


class TeacherModel(nn.Module):
    """Teacher model wrapper."""
    def __init__(self, model_name: str, num_classes: int = 100, weights_path: str = None, 
                 pretrained: bool = True):
        super().__init__()
        self.model = self._build_model(model_name, pretrained=pretrained, num_classes=num_classes)
        if weights_path:
            self.load_teacher_weights(weights_path)

    def _build_model(self, model_name, pretrained, num_classes):
        """Build the model architecture."""
        if model_name.startswith('resnet'):
            # Extract number from model name (e.g., "50" from "resnet50")
            num = model_name[len('resnet'):]
            if pretrained:
                weights_enum = getattr(models, f"ResNet{num}_Weights")
                weights = weights_enum.DEFAULT
                print(f"[Teacher] Loading pretrained ImageNet weights for {model_name}")
            else:
                weights = None
                print(f"[Teacher] Initializing {model_name} with random weights")
            
            model = getattr(models, model_name)(weights=weights)
            
            # Replace final head if needed
            if num_classes != model.fc.out_features:
                model.fc = nn.Linear(model.fc.in_features, num_classes)
                if pretrained:
                    print(
                        f"[Teacher WARNING] Replaced {model_name} classifier for "
                        f"{num_classes} classes. If no matching fine-tuned "
                        "teacher_weights are loaded, teacher logits may be weak."
                    )

        elif model_name.startswith('vgg'):
            if pretrained:
                weights_enum = getattr(models, f"{model_name.upper()}_Weights")
                weights = weights_enum.DEFAULT
                print(f"[Teacher] Loading pretrained ImageNet weights for {model_name}")
            else:
                weights = None
                print(f"[Teacher] Initializing {model_name} with random weights")
            
            model = getattr(models, model_name)(weights=weights)
            
            # Replace final head if needed
            if num_classes != model.classifier[-1].out_features:
                layers = list(model.classifier)
                for i in reversed(range(len(layers))):
                    if isinstance(layers[i], nn.Linear):
                        in_feats = layers[i].in_features
                        layers[i] = nn.Linear(in_feats, num_classes)
                        break
                model.classifier = nn.Sequential(*layers)
                if pretrained:
                    print(
                        f"[Teacher WARNING] Replaced {model_name} classifier for "
                        f"{num_classes} classes. If no matching fine-tuned "
                        "teacher_weights are loaded, teacher logits may be weak."
                    )

        else:
            raise ValueError(f"Unsupported model: {model_name}")

        return model

    def load_teacher_weights(self, weights_path):
        """Load pretrained weights for teacher."""
        checkpoint = torch.load(weights_path, weights_only=False)
        sd, source_key = _extract_state_dict(checkpoint)
        sd = _strip_common_prefixes(sd)

        if not isinstance(sd, dict):
            raise ValueError(
                f"[Teacher] Expected a state_dict-like mapping in {weights_path}, got {type(sd)}"
            )

        model_keys = set(self.model.state_dict().keys())
        ckpt_keys = set(sd.keys())
        matched = len(model_keys & ckpt_keys)

        incompatible = self.model.load_state_dict(sd, strict=False)
        missing = len(incompatible.missing_keys)
        unexpected = len(incompatible.unexpected_keys)

        source_desc = f" key='{source_key}'" if source_key else ""
        print(
            f"[Teacher] Loaded custom weights from {weights_path}{source_desc} "
            f"(matched={matched}, missing={missing}, unexpected={unexpected})"
        )
        if matched == 0:
            print(
                "[Teacher WARNING] 0 parameter keys matched the teacher model. "
                "This usually means a wrong architecture or checkpoint format."
            )

    def forward(self, x):
        return self.model(x)


class StudentModel(nn.Module):
    """Student model wrapper."""
    def __init__(self, model_name: str, num_classes: int = 100, weights_path: str = None):
        super().__init__()
        self.model = self._build_model(model_name, pretrained=False, num_classes=num_classes)
        if weights_path:
            self.load_student_weights(weights_path)

    def _build_model(self, model_name, pretrained, num_classes):
        """Build the model architecture."""
        if model_name.startswith('resnet'):
            model = getattr(models, model_name)(weights=None)
            if num_classes != model.fc.out_features:
                model.fc = nn.Linear(model.fc.in_features, num_classes)

        elif model_name.startswith('vgg'):
            model = getattr(models, model_name)(weights=None)
            if num_classes != model.classifier[-1].out_features:
                layers = list(model.classifier)
                for i in reversed(range(len(layers))):
                    if isinstance(layers[i], nn.Linear):
                        in_feats = layers[i].in_features
                        layers[i] = nn.Linear(in_feats, num_classes)
                        break
                model.classifier = nn.Sequential(*layers)
                
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        return model

    def load_student_weights(self, weights_path):
        """Load pretrained weights for student."""
        checkpoint = torch.load(weights_path, weights_only=False)
        sd, source_key = _extract_state_dict(checkpoint)
        sd = _strip_common_prefixes(sd)

        if not isinstance(sd, dict):
            raise ValueError(
                f"[Student] Expected a state_dict-like mapping in {weights_path}, got {type(sd)}"
            )

        model_keys = set(self.model.state_dict().keys())
        ckpt_keys = set(sd.keys())
        matched = len(model_keys & ckpt_keys)

        incompatible = self.model.load_state_dict(sd, strict=False)
        missing = len(incompatible.missing_keys)
        unexpected = len(incompatible.unexpected_keys)

        source_desc = f" key='{source_key}'" if source_key else ""
        print(
            f"[Student] Loaded weights from {weights_path}{source_desc} "
            f"(matched={matched}, missing={missing}, unexpected={unexpected})"
        )
        if matched == 0:
            print(
                "[Student WARNING] 0 parameter keys matched the student model. "
                "This usually means a wrong architecture or checkpoint format."
            )

    def forward(self, x):
        return self.model(x)
