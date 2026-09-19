"""
Versioned prompt registry & loader.

All system/agent prompts live as versioned YAML files under config/prompts/
(see registry.yaml). Selection is:

  1. entry in registry.yaml -> defaults.active_versions[name]
  2. overridden by the PROMPT_VERSION env var:
       - "responder_technical@2.0"  -> that prompt only
       - "2.0"                      -> applies to every prompt
"""
import functools
import os
import re
from string import Template
from typing import Any

import logfire
import yaml
from pydantic import BaseModel

from app.config import settings


class PromptSpec(BaseModel):
    name: str
    version: str
    description: str | None = None
    template: str


class PromptRegistry(BaseModel):
    version: str
    defaults: dict[str, Any]
    prompts: dict[str, dict[str, str]]


@functools.lru_cache(maxsize=1)
def _load_registry() -> PromptRegistry:
    path = os.path.join(settings.PROMPTS_DIR, "registry.yaml")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return PromptRegistry(**data)


def _prompt_file(name: str) -> str:
    registry = _load_registry()
    entry = registry.prompts.get(name)
    if not entry:
        raise KeyError(f"Prompt '{name}' is not registered in the registry.")
    return os.path.join(settings.PROMPTS_DIR, entry["file"])


def _active_version(name: str) -> str:
    registry = _load_registry()
    active = registry.defaults.get("active_versions", {}).get(name, "1.0")

    env = settings.PROMPT_VERSION
    if env:
        # Accept "name@version" (single prompt) or a bare version (applies to all).
        if "@" in env:
            env_name, env_version = env.rsplit("@", 1)
            if env_name == name:
                return env_version
        else:
            return env

    return str(active)


@functools.lru_cache(maxsize=64)
def get_prompt(name: str) -> PromptSpec:
    """Load the active version of a named prompt (cached by name + version)."""
    version = _active_version(name)
    with open(_prompt_file(name), "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    spec = PromptSpec(**data)

    if data.get("version") != version:
        raise ValueError(
            f"Prompt '{name}' version '{version}' requested but the file defines "
            f"'{data.get('version')}'. Update the registry or PROMPT_VERSION."
        )
    logfire.info(f"Loaded prompt '{name}' v{spec.version}.")
    return spec


def render_prompt(name: str, **params: str) -> str:
    """Render the active prompt template with the given $placeholders."""
    spec = get_prompt(name)
    missing = _missing_placeholders(spec.template, params)
    if missing:
        raise KeyError(f"Prompt '{name}' is missing required parameters: {sorted(missing)}")
    return Template(spec.template).substitute(**params)


def _missing_placeholders(template: str, params: dict[str, Any]) -> set[str]:
    """Find $placeholders in the template that are not supplied via params."""
    found: set[str] = set()
    for match in re.finditer(r"\$\{?([a-zA-Z_][a-zA-Z0-9_]*)\}?", template):
        name = match.group(1)
        if name not in params:
            found.add(name)
    try:
        Template(template)
    except ValueError as e:
        # Invalid templates (e.g. lone "$") never render; surface a clear error.
        raise ValueError(f"Invalid prompt template: {e}") from e
    return found