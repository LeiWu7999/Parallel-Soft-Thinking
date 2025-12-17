# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import json
import os

from huggingface_hub import snapshot_download


class LoRAConfig:
    def __init__(
        self,
        path: str,
    ) -> None:
        self.path = path
        self.hf_config = self.get_lora_config()
        self.target_modules = self.hf_config["target_modules"]

        # TODO: Support more modules
        if any(module in self.target_modules for module in ["embed_tokens", "lm_head"]):
            raise ValueError("Not supported yet")

        self.r = self.hf_config["r"]
        self.lora_alpha = self.hf_config["lora_alpha"]

    @staticmethod
    def _normalize_target_modules(target_modules):
        if target_modules is None:
            return None
        if isinstance(target_modules, str):
            return [target_modules]
        return list(target_modules)

    @classmethod
    def _convert_legacy_lora_config(cls, legacy_cfg: dict) -> dict:
        """Convert training-time `lora_config.json` into a PEFT-like config dict."""
        target_modules = legacy_cfg.get("target_modules") or legacy_cfg.get("targets")
        target_modules = cls._normalize_target_modules(target_modules)
        if not target_modules:
            raise ValueError(
                "Invalid lora_config.json: missing `targets`/`target_modules`."
            )

        r = legacy_cfg.get("r", None)
        if r is None:
            r = legacy_cfg.get("lora_r", None)
        if r is None:
            raise ValueError("Invalid lora_config.json: missing `lora_r`/`r`.")

        lora_alpha = legacy_cfg.get("lora_alpha", None)
        if lora_alpha is None:
            lora_alpha = legacy_cfg.get("alpha", None)
        if lora_alpha is None:
            raise ValueError("Invalid lora_config.json: missing `lora_alpha`/`alpha`.")

        lora_dropout = legacy_cfg.get("lora_dropout", 0.0)

        # Populate the minimal PEFT fields used by SGLang.
        return {
            "peft_type": legacy_cfg.get("peft_type", "LORA"),
            "target_modules": target_modules,
            "r": int(r),
            "lora_alpha": int(lora_alpha),
            "lora_dropout": float(lora_dropout),
            # Optional fields (kept for debugging / parity).
            "base_model_name_or_path": legacy_cfg.get("base_model", None),
            "gated_tokenwise": legacy_cfg.get("gated_tokenwise", None),
        }

    def get_lora_config(self, dummy=False):
        if dummy:
            raise NotImplementedError()
        else:
            if not os.path.isdir(self.path):
                weights_dir = snapshot_download(self.path, allow_patterns=["*.json"])
            else:
                weights_dir = self.path
            peft_cfg_path = os.path.join(weights_dir, "adapter_config.json")
            if os.path.isfile(peft_cfg_path):
                with open(peft_cfg_path, "r") as f:
                    cfg = json.load(f)
                cfg["target_modules"] = self._normalize_target_modules(
                    cfg.get("target_modules")
                )
                return cfg

            legacy_cfg_path = os.path.join(weights_dir, "lora_config.json")
            if os.path.isfile(legacy_cfg_path):
                with open(legacy_cfg_path, "r", encoding="utf-8") as f:
                    legacy_cfg = json.load(f)
                return self._convert_legacy_lora_config(legacy_cfg)

            raise FileNotFoundError(
                f"Cannot find LoRA config in {weights_dir}. Expected `adapter_config.json` or `lora_config.json`."
            )
