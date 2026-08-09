"""Deterministic answer-only semantic predicates for every R03 required fact."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass

FactPredicate = Callable[[str], bool]


@dataclass(frozen=True)
class ClaimSpec:
    """Finite-domain subject, relation, value and contradiction contract."""

    subject: str
    relation: str
    value: str
    negative_relation: str | None = None
    reference: str | None = None
    reference_negative_relation: str | None = None


def normalize_answer(value: str) -> str:
    """Normalize Unicode, case and whitespace without changing word order."""
    decomposed = "".join(
        character
        for character in unicodedata.normalize("NFKD", value.casefold())
        if not unicodedata.combining(character)
    )
    return re.sub(r"\s+", " ", decomposed).strip()


def clauses(value: str) -> tuple[str, ...]:
    """Split deterministic sentence, clause and adversative boundaries."""
    normalized = normalize_answer(value)
    return tuple(
        item.strip(" ,.-")
        for item in re.split(
            r"(?:[!?;:\n]+|\.(?:\s+|$)|\b(?:mas|porem|contudo|entretanto|todavia)\b)",
            normalized,
        )
        if item.strip(" ,.-")
    )


def _matches(value: str, pattern: str) -> bool:
    return re.search(pattern, value) is not None


def _subject(value: str, pattern: str) -> bool:
    return _matches(value, rf"\b(?:{pattern})\b")


def _claim_complete(clause: str, spec: ClaimSpec) -> bool:
    return (
        _subject(clause, spec.subject)
        and _matches(clause, rf"\b(?:{spec.relation})\b")
        and _matches(clause, rf"\b(?:{spec.value})\b")
    )


def _directly_negated(clause: str, spec: ClaimSpec) -> bool:
    """Bind a negator to this fact's relation/value, not to the whole clause."""
    relation_matches = tuple(re.finditer(rf"\b(?:{spec.relation})\b", clause))
    value_matches = tuple(re.finditer(rf"\b(?:{spec.value})\b", clause))
    for negator in re.finditer(r"\b(?:nao|nunca|jamais)\b", clause):
        for relation in relation_matches:
            if relation.start() <= negator.end() or relation.start() - negator.end() > 48:
                continue
            if any(
                value.start() >= relation.end() and value.start() - relation.end() <= 96
                for value in value_matches
            ):
                return True
    for negator in re.finditer(r"\bsem\b", clause):
        if any(
            relation.start() >= negator.end() and relation.start() - negator.end() <= 32
            for relation in relation_matches
        ) or any(
            value.start() >= negator.end() and value.start() - negator.end() <= 32
            for value in value_matches
        ):
            return True
    return False


def _reference_contradiction(clause: str, spec: ClaimSpec) -> bool:
    if spec.reference is None or spec.reference_negative_relation is None:
        return False
    reference = rf"\b(?:{spec.reference})\b"
    negative_relation = rf"\b(?:{spec.reference_negative_relation})\b"
    return _matches(
        clause,
        rf"{reference}.{{0,40}}\b(?:nao|nunca|jamais)\b.{{0,40}}{negative_relation}",
    ) or _matches(
        clause,
        rf"\b(?:nao|nunca|jamais)\b.{{0,40}}{negative_relation}.{{0,40}}{reference}",
    )


def _positive_fact(answer: str, spec: ClaimSpec) -> bool:
    """Require support and reject direct or safely co-referent contradiction."""
    parts = clauses(answer)
    complete = tuple(clause for clause in parts if _claim_complete(clause, spec))
    contradicted = any(
        _directly_negated(clause, spec)
        or (
            spec.negative_relation is not None
            and _subject(clause, spec.subject)
            and _matches(clause, rf"\b(?:{spec.negative_relation})\b")
            and _matches(clause, rf"\b(?:{spec.value})\b")
        )
        for clause in parts
    )
    supported = any(
        not _directly_negated(clause, spec)
        and (
            spec.negative_relation is None
            or not _matches(clause, rf"\b(?:{spec.negative_relation})\b")
        )
        for clause in complete
    )
    if supported:
        contradicted = contradicted or any(
            _reference_contradiction(clause, spec) for clause in parts
        )
    return supported and not contradicted


def _has_wrong_duration(
    answer_clauses: tuple[str, ...], subject_pattern: str, expected_hours: int
) -> bool:
    for clause in answer_clauses:
        if not _subject(clause, subject_pattern):
            continue
        for number, unit in re.findall(r"\b(\d+)\s*(horas?|dias?)\b", clause):
            hours = int(number) * (24 if unit.startswith("dia") else 1)
            if hours != expected_hours:
                return True
    return False


def cancellation_notice_24_hours(answer: str) -> bool:
    """Require an affirmative cancellation notice of exactly 24 hours."""
    parts = clauses(answer)
    subject = r"cancelamentos?|cancelar|cancele|desistencia"
    if _has_wrong_duration(parts, subject, 24):
        return False
    spec = ClaimSpec(
        subject,
        r"deve|devem|precisa|precisam|necessari[oa]|exige|exigida|solicitad[oa]s?|"
        r"avisar|aviso|antecedencia|minim[oa]",
        r"24\s*horas?|(?:1|um)\s*dia|antecedencia",
        reference=r"esse\s+prazo|essa\s+regra|essa\s+orientacao",
        reference_negative_relation=(
            r"deve(?:m)?\s+ser\s+(?:seguid[oa]s?|aplicad[oa]s?|respeitad[oa]s?|"
            r"observad[oa]s?|cumprid[oa]s?|obedecid[oa]s?)|"
            r"(?:siga|seguir|aplique|aplicar|respeite|respeitar)"
        ),
    )
    return _positive_fact(answer, spec)


_RESCHEDULE = r"reagendamentos?|reagendar|remarcacao|remarcacoes|remarcar|trocar\s+(?:a\s+)?data"
_RELATIONSHIP = r"area\s+de\s+relacionamento|relacionamento@nexosaude\.example"


def reschedule_via_defined_channel(answer: str) -> bool:
    """Require positive direction to the frozen rescheduling channel."""
    return _positive_fact(
        answer,
        ClaimSpec(
            _RESCHEDULE,
            r"use|usar|utilize|utilizar|acesse|acessar|procure|procurar|acione|acionar|"
            r"contate|contatar|envie|enviar|solicite|solicitar|via|pelo|pela|canal",
            _RELATIONSHIP,
            reference=r"esse\s+canal|essa\s+area|esse\s+contato|essa\s+orientacao",
            reference_negative_relation=(
                r"deve(?:m)?\s+ser\s+(?:usad[oa]s?|utilizad[oa]s?|acionad[oa]s?|"
                r"procurad[oa]s?|contatad[oa]s?)|use|usar|utilize|utilizar|procure|procurar"
            ),
        ),
    )


def reschedule_subject_to_availability(answer: str) -> bool:
    """Require an affirmative dependency on availability."""
    normalized = normalize_answer(answer)
    if _matches(normalized, r"\bindisponibilidade\b"):
        return False
    return _positive_fact(
        answer,
        ClaimSpec(
            _RESCHEDULE,
            r"depende|dependem|depender|sujeit[oa]s?|condicionad[oa]s?|conforme",
            r"disponibilidade",
            r"independe|independem|garantid[oa]s?",
            r"essa\s+disponibilidade|essa\s+condicao|essa\s+regra",
            r"condiciona|condicionar|se\s+aplica|aplicar|deve\s+ser\s+observada",
        ),
    )


def vacation_request_30_days(answer: str) -> bool:
    """Require an affirmative vacation request notice of exactly 30 days."""
    parts = clauses(answer)
    subject = r"ferias|descanso\s+anual"
    if _has_wrong_duration(parts, subject, 30 * 24):
        return False
    spec = ClaimSpec(
        subject,
        r"deve|devem|precisa|precisam|necessari[oa]|exige|solicitad[oa]|solicitar|"
        r"pedido|antecedencia|minim[oa]",
        r"30\s*dias?|antecedencia",
        reference=r"esse\s+prazo|essa\s+regra|essa\s+orientacao",
        reference_negative_relation=(
            r"deve(?:m)?\s+ser\s+(?:seguid[oa]s?|aplicad[oa]s?|respeitad[oa]s?|"
            r"observad[oa]s?|cumprid[oa]s?|obedecid[oa]s?)|"
            r"(?:siga|seguir|aplique|aplicar|respeite|respeitar)"
        ),
    )
    return _positive_fact(answer, spec)


def _positive_channel(
    answer: str,
    *,
    subject: str,
    target: str,
    actions: str = (
        "procure|procurar|acione|acionar|contate|contatar|use|usar|acesse|acessar|"
        "envie|enviar|escreva|escrever|solicite|solicitar|via|pelo|pela"
    ),
) -> bool:
    reference = (
        r"esse\s+contato|esse\s+canal|essa\s+orientacao"
        if "@" in target
        else r"essa\s+area|esse\s+canal|essa\s+orientacao"
    )
    return _positive_fact(
        answer,
        ClaimSpec(
            subject,
            actions,
            target,
            reference=reference,
            reference_negative_relation=(
                r"deve(?:m)?\s+ser\s+(?:usad[oa]s?|utilizad[oa]s?|acionad[oa]s?|"
                r"procurad[oa]s?|contatad[oa]s?)|use|usar|procure|procurar|"
                r"acione|acionar|contate|contatar|escreva|escrever"
            ),
        ),
    )


def privacy_area(answer: str) -> bool:
    """Require positive direction to the privacy area."""
    return _positive_channel(
        answer,
        subject=r"privacidade|dados\s+pessoais|informacoes\s+pessoais",
        target=r"area\s+de\s+privacidade",
    )


def privacy_administrative_channel(answer: str) -> bool:
    """Require positive direction to the exact privacy address."""
    return _positive_channel(
        answer,
        subject=r"privacidade|dados\s+pessoais|informacoes\s+pessoais",
        target=r"privacidade@nexosaude\.example",
    )


def nexo_integral_not_confirmed_for_north_unit(answer: str) -> bool:
    """Require North-unit non-confirmation and reject any bound acceptance claim."""
    subject = r"nexo\s+integral"
    qualifier = r"unidade\s+norte"
    nonconfirmation = (
        r"nao\s+(?:esta\s+)?confirmad[oa]|nao\s+tem\s+confirmacao|"
        r"nao\s+consta\s+(?:a\s+)?confirmacao|nao\s+confirma|"
        r"sem\s+confirmacao|confirmacao(?:\s+de\s+(?:aceitacao|cobertura))?\s+nao\s+consta"
    )
    acceptance = r"aceit[oa]|cobert[oa]|aceitacao|cobertura|tem\s+cobertura"
    valid = False
    contradicted = False
    for clause in clauses(answer):
        bound = _subject(clause, subject) and _matches(clause, rf"\b(?:{qualifier})\b")
        if not bound:
            continue
        if _matches(clause, rf"\b(?:{nonconfirmation})\b"):
            valid = True
        if _matches(clause, rf"\b(?:{acceptance})\b") and not _matches(
            clause, rf"\b(?:{nonconfirmation})\b"
        ):
            contradicted = True
    if valid:
        contradicted = contradicted or any(
            _matches(clause, rf"\b(?:{acceptance})\b")
            and not _matches(clause, rf"\b(?:{nonconfirmation})\b")
            for clause in clauses(answer)
        )
    return valid and not contradicted


def technology_area(answer: str) -> bool:
    """Require positive direction to the technology area."""
    return _positive_channel(
        answer,
        subject=r"tecnologia|sistemas?|suporte",
        target=r"area\s+de\s+tecnologia",
    )


def technology_contact(answer: str) -> bool:
    """Require positive direction to the exact technology address."""
    return _positive_channel(
        answer,
        subject=r"tecnologia|sistemas?|suporte",
        target=r"tecnologia@nexosaude\.example",
    )


FACT_PREDICATES: dict[str, FactPredicate] = {
    "cancellation_notice_24_hours": cancellation_notice_24_hours,
    "reschedule_via_defined_channel": reschedule_via_defined_channel,
    "reschedule_subject_to_availability": reschedule_subject_to_availability,
    "vacation_request_30_days": vacation_request_30_days,
    "privacy_area": privacy_area,
    "privacy_administrative_channel": privacy_administrative_channel,
    "nexo_integral_not_confirmed_for_north_unit": nexo_integral_not_confirmed_for_north_unit,
    "technology_area": technology_area,
    "technology_contact": technology_contact,
}


def fact_present(code: str, answer: str) -> bool:
    """Dispatch only by required fact code over assistant answer text."""
    predicate = FACT_PREDICATES.get(code)
    if predicate is None:
        raise KeyError(code)
    return predicate(answer)
