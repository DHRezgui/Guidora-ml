"""Tour semantic role inference (Phase 2).

This script powers the Phase 2 path of the SDK hybrid semantic layer.
Unlike Phase 1, which uses a deterministic rule classifier, this module
uses the existing sentence-transformers wrapper (`SemanticEmbedder`) to
embed each candidate label and compare it against a small bank of role
prototypes via cosine similarity.

I/O contract
============

Input (stdin, JSON):
    {
      "candidates": [
        {"selector": "...", "label": "Créer un projet", "tag": "button",
         "intent": "primary-action", "zone": "main"},
        ...
      ],
      "snapshot": {"hasForm": bool, "hasNavigation": bool, "formFieldCount": int,
                    "navigationLinkCount": int, "ctaCount": int},
      "hints": ["..."]?, "objectives": ["..."]?
    }

Output (stdout, JSON):
    {
      "success": true,
      "roles": [
        {"selector": "...", "role": "cta-primary", "confidence": 0.72,
         "rationale": ["embedding similarity 0.61 vs cta-primary"]},
        ...
      ],
      "preferredOrder": ["..."],
      "implementation": {
        "kind": "sentence-transformers",
        "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "phase": "phase-2-embeddings"
      }
    }

On unrecoverable error: success=false + non-zero exit code, so the
backend can detect the failure and fall back to the rule-based engine.

Confidence calibration
======================

Cosine similarities for short UI labels on a same-domain corpus typically
sit in the [0.20, 0.65] range. A raw cosine is therefore NOT directly
comparable to the SDK's 0.55 confidence guard, which was calibrated for
the rule-vote engine on a [0, 1] vote scale.

To keep the SDK's existing threshold meaningful without re-plumbing, we
report a *margin-calibrated* confidence:

    margin = top_sim - second_best_sim
    confidence = clamp(0.5 + 2.5 * margin, 0, 1)

This mapping has well-defined endpoints:
- margin = 0.00 (perfect tie between top-2 roles) -> confidence 0.50
  (just below the SDK guard -> step will be suppressed: honest "unsure").
- margin = 0.02 -> confidence 0.55 (right at the guard).
- margin = 0.20 -> confidence 1.00 (very clear winner).

An additional absolute floor (`MIN_TOP_SIMILARITY`) suppresses candidates
whose best role similarity is below ~0.30 — these are typically truly
out-of-domain labels (icon-only buttons, raw IDs).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

# Make ml/ importable regardless of cwd (matches semantic_search.py).
ML_ROOT = Path(__file__).resolve().parents[1]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

try:
    from .semantic_embedder import SemanticEmbedder  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - script execution path
    from semantic_embedder import SemanticEmbedder  # type: ignore[import-not-found]

from config import SEMANTIC_CONFIG  # noqa: E402  (path setup must run first)


# ---------------------------------------------------------------------------
# Role prototypes (multilingual, intentionally short and label-like).
# Keep these synchronised with the SDK's SemanticRole union — adding a
# new role here also requires updating semantic-step-intelligence.ts and
# the SDK type union in types/sdk.ts.
# ---------------------------------------------------------------------------
ROLE_PROTOTYPES: Dict[str, List[str]] = {
    "cta-primary": [
        "Créer un nouveau projet",
        "Démarrer un nouvel onboarding",
        "Lancer la simulation",
        "Confirmer l'action principale",
        "Ajouter un élément",
        "Inviter un collaborateur",
        "Get started now",
        "Create a new account",
        "Launch the process",
        "Add new item",
    ],
    "form-submit": [
        "Enregistrer le formulaire",
        "Valider les informations saisies",
        "Soumettre la demande",
        "Publier le document",
        "Sauvegarder les modifications",
        "Save changes",
        "Submit the form",
        "Confirm submission",
    ],
    "form-field": [
        "Saisir une adresse email",
        "Renseigner le nom de famille",
        "Champ de mot de passe",
        "Date de naissance",
        "Numéro de téléphone",
        "Type your message here",
        "Enter your email",
    ],
    "navigation": [
        "Aller à l'accueil",
        "Lien vers la section budget",
        "Menu de navigation principale",
        "Onglet Tableau de bord",
        "Ouvrir le panneau analytique",
        "Go to dashboard",
        "Navigate to settings",
    ],
    "secondary": [
        "Consulter le guide utilisateur",
        "Voir l'aide en ligne",
        "Découvrir les nouveautés",
        "Tutoriel d'introduction",
        "Read the documentation",
        "Open the help center",
    ],
    "utility": [
        "Ouvrir les réglages",
        "Paramètres du compte",
        "Préférences utilisateur",
        "Configuration avancée",
        "Options d'affichage",
        "Open settings",
    ],
    "entry": [
        "Bienvenue sur le tableau de bord",
        "Aperçu général du parcours",
        "Page d'accueil de l'application",
        "Introduction au module",
        "Prise en main",
        "Welcome screen",
    ],
    "result": [
        "Résultat de l'analyse",
        "Statut: opération terminée avec succès",
        "Aperçu des données générées",
        "Operation completed",
        "Rapport final",
    ],
    "generic-click": [
        "Cliquer pour interagir",
        "Bouton générique de la page",
        "Action contextuelle",
        "Interactive element",
    ],
}

# Roles ranked from highest to lowest for ordering proposals when
# similarities are tied. This mirrors the SDK ROLE_ORDER constant.
ROLE_ORDER_FOR_SEQUENCE: List[str] = [
    "entry",
    "navigation",
    "cta-primary",
    "form-field",
    "form-submit",
    "utility",
    "secondary",
    "generic-click",
    "result",
]

MIN_TOP_SIMILARITY = 0.30
MARGIN_SCALE = 2.5
DEFAULT_CONFIDENCE_FLOOR = 0.50

# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def _flatten_prototypes(prototypes: Dict[str, List[str]]) -> Tuple[List[str], List[Tuple[str, int]]]:
    """Flatten role -> list[text] into (texts, role_index_pairs)."""
    flat_texts: List[str] = []
    index: List[Tuple[str, int]] = []
    for role, examples in prototypes.items():
        for example in examples:
            index.append((role, len(flat_texts)))
            flat_texts.append(example)
    return flat_texts, index


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _build_candidate_text(candidate: Dict[str, Any]) -> str:
    """Build the text used as semantic input for a single candidate.

    Concatenates label + heuristic intent + tag. Adding the intent and
    tag as soft hints lets the embedder differentiate between a
    "Confirmer" link (likely secondary nav) and a "Confirmer" submit
    button without baking those signals into the prototypes themselves.
    """
    parts: List[str] = []
    label = (candidate.get("label") or "").strip()
    if label:
        parts.append(label)
    intent = (candidate.get("intent") or "").strip()
    if intent:
        parts.append(f"intention métier: {intent}")
    tag = (candidate.get("tag") or "").strip()
    if tag:
        parts.append(f"élément <{tag}>")
    if not parts:
        return "élément générique de l'interface"
    return " — ".join(parts)


def _aggregate_similarities_per_role(
    candidate_vec: Sequence[float],
    prototype_vecs: Sequence[Sequence[float]],
    role_index: Sequence[Tuple[str, int]],
) -> Dict[str, float]:
    """Return max cosine similarity per role for this candidate."""
    per_role: Dict[str, float] = {}
    for role, idx in role_index:
        sim = _dot(candidate_vec, prototype_vecs[idx])
        if role not in per_role or sim > per_role[role]:
            per_role[role] = sim
    return per_role


def _calibrate_confidence(top_sim: float, second_sim: float) -> float:
    """Margin-calibrated confidence aligned with the SDK 0.55 guard."""
    margin = max(0.0, top_sim - second_sim)
    raw = DEFAULT_CONFIDENCE_FLOOR + MARGIN_SCALE * margin
    return max(0.0, min(1.0, raw))


def _apply_snapshot_bias(
    role: str,
    confidence: float,
    snapshot: Dict[str, Any],
) -> float:
    """Modest contextual boost when the snapshot agrees with the role.

    Mirrors the bias the rule engine applies: a form-field detection is
    slightly more credible on a page that actually contains a form, and
    a navigation detection is slightly more credible on a page that
    actually has a navigation zone.
    """
    if snapshot.get("hasForm") and role in ("form-field", "form-submit"):
        return min(1.0, confidence + 0.05)
    if snapshot.get("hasNavigation") and role == "navigation":
        return min(1.0, confidence + 0.05)
    return confidence


def _preferred_order(
    candidates: List[Dict[str, Any]],
    roles_by_selector: Dict[str, str],
) -> List[str]:
    """Order candidates by role precedence; stable on ties (insertion order)."""
    indexed = list(enumerate(candidates))
    indexed.sort(
        key=lambda pair: (
            ROLE_ORDER_FOR_SEQUENCE.index(roles_by_selector.get(pair[1].get("selector") or "", "generic-click"))
            if roles_by_selector.get(pair[1].get("selector") or "", "generic-click") in ROLE_ORDER_FOR_SEQUENCE
            else len(ROLE_ORDER_FOR_SEQUENCE),
            pair[0],
        )
    )
    return [pair[1].get("selector") or "" for pair in indexed]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

_cached_embedder: SemanticEmbedder | None = None


def _get_embedder() -> SemanticEmbedder:
    """Reuse one SentenceTransformer instance per Python process (worker)."""
    global _cached_embedder
    if _cached_embedder is None:
        _cached_embedder = SemanticEmbedder()
    return _cached_embedder


def warmup_model() -> str:
    """Load the sentence-transformers model once (worker startup / warmup)."""
    return _get_embedder().model_name


def infer(payload: Dict[str, Any]) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = payload.get("candidates") or []
    snapshot: Dict[str, Any] = payload.get("snapshot") or {}

    if not candidates:
        return {
            "success": True,
            "roles": [],
            "preferredOrder": [],
            "implementation": {
                "kind": "sentence-transformers",
                "model": SEMANTIC_CONFIG["model_name"],
                "phase": "phase-2-embeddings",
            },
        }

    embedder = _get_embedder()

    prototype_texts, role_index = _flatten_prototypes(ROLE_PROTOTYPES)
    prototype_vecs = embedder.embed(prototype_texts)

    candidate_texts = [_build_candidate_text(c) for c in candidates]
    candidate_vecs = embedder.embed(candidate_texts)

    roles_payload: List[Dict[str, Any]] = []
    roles_by_selector: Dict[str, str] = {}

    for candidate, vec in zip(candidates, candidate_vecs):
        per_role = _aggregate_similarities_per_role(vec, prototype_vecs, role_index)
        ranked = sorted(per_role.items(), key=lambda kv: kv[1], reverse=True)
        if not ranked:
            roles_payload.append(
                {
                    "selector": candidate.get("selector") or "",
                    "role": "generic-click",
                    "confidence": 0.30,
                    "rationale": ["no prototype available"],
                }
            )
            continue

        top_role, top_sim = ranked[0]
        second_sim = ranked[1][1] if len(ranked) > 1 else 0.0

        if top_sim < MIN_TOP_SIMILARITY:
            # Out-of-domain candidate: respect the SDK contract and
            # propose a low-confidence generic role so the fusion stage
            # silences it via the 0.55 guard.
            confidence = max(0.0, min(0.45, top_sim))
            roles_payload.append(
                {
                    "selector": candidate.get("selector") or "",
                    "role": "generic-click",
                    "confidence": round(confidence, 3),
                    "rationale": [
                        f"top similarity {top_sim:.3f} below floor {MIN_TOP_SIMILARITY}"
                    ],
                }
            )
            roles_by_selector[candidate.get("selector") or ""] = "generic-click"
            continue

        confidence = _calibrate_confidence(top_sim, second_sim)
        confidence = _apply_snapshot_bias(top_role, confidence, snapshot)
        roles_payload.append(
            {
                "selector": candidate.get("selector") or "",
                "role": top_role,
                "confidence": round(confidence, 3),
                "rationale": [
                    f"embedding similarity {top_sim:.3f} vs {top_role}",
                    f"margin vs second best ({ranked[1][0] if len(ranked) > 1 else 'none'}): {top_sim - second_sim:+.3f}",
                ],
            }
        )
        roles_by_selector[candidate.get("selector") or ""] = top_role

    preferred = _preferred_order(candidates, roles_by_selector)

    return {
        "success": True,
        "roles": roles_payload,
        "preferredOrder": preferred,
        "implementation": {
            "kind": "sentence-transformers",
            "model": SEMANTIC_CONFIG["model_name"],
            "phase": "phase-2-embeddings",
        },
    }


def main() -> None:
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw or "{}")
        result = infer(payload)
        print(json.dumps(result, ensure_ascii=False))
    except Exception as exc:  # noqa: BLE001 - surface every error path to the backend caller
        error_payload = {"success": False, "error": str(exc), "errorType": type(exc).__name__}
        print(json.dumps(error_payload, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
