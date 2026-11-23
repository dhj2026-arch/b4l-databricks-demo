import json
import re

import pytest

from b4l import genie_space as gs
from b4l import schemas

CATALOG = "serverless_stable_yqq46w_catalog"
PREFIX = "b4l"
HEX32 = re.compile(r"^[0-9a-f]{32}$")


@pytest.fixture
def space():
    return gs.build_serialized_space(CATALOG, PREFIX)


def _all_ids(space):
    ids = [q["id"] for q in space["config"]["sample_questions"]]
    ids += [t["id"] for t in space["instructions"]["text_instructions"]]
    ids += [e["id"] for e in space["instructions"]["example_question_sqls"]]
    return ids


def test_tables_are_all_gold_tables_sorted(space):
    idents = [t["identifier"] for t in space["data_sources"]["tables"]]
    assert idents == sorted(idents)
    assert idents == sorted(f"{CATALOG}.{PREFIX}_gold.{t}" for t in schemas.GOLD)


def test_version_and_ids(space):
    assert space["version"] == 2
    ids = _all_ids(space)
    assert all(HEX32.match(i) for i in ids)
    assert len(ids) == len(set(ids))
    for coll in (
        space["config"]["sample_questions"],
        space["instructions"]["text_instructions"],
        space["instructions"]["example_question_sqls"],
    ):
        assert [x["id"] for x in coll] == sorted(x["id"] for x in coll)


def test_deterministic_payload():
    assert gs.serialize(gs.build_serialized_space(CATALOG, PREFIX)) == gs.serialize(
        gs.build_serialized_space(CATALOG, PREFIX)
    )


def test_ten_bilingual_sample_questions(space):
    questions = [q["question"][0] for q in space["config"]["sample_questions"]]
    assert len(questions) == 10
    assert len(set(questions)) == 10
    french = [q for q in questions if re.search(r"[éèàç]|Quel|Combien", q)]
    assert 3 <= len(french) <= 7
    for required in (
        "Quel est le taux d'occupation des créneaux de midi la semaine dernière ?",
        "Which hours on weekdays have the lowest occupancy in October?",
        "How much revenue did rainy Saturdays bring vs dry Saturdays?",
        "Combien d'annulations tardives ce mois-ci ?",
        "Which players have a Pass Chrono expiring in the next 14 days?",
    ):
        assert required in questions


def test_single_text_instruction_covers_rules(space):
    texts = space["instructions"]["text_instructions"]
    assert len(texts) == 1
    text = "".join(texts[0]["content"])
    for term in (
        "CHF", "taux d'occupation", "as_of_date", "14 days", "k-anonymised", "school holidays",
        "public holidays", f"{CATALOG}.{PREFIX}_gold",
    ):
        assert term in text, term
    assert len(text) < 25_000


@pytest.mark.parametrize(
    "term",
    [
        "Tarif journée", "Tarif midi", "Tarif soir", "Tarif week-end", "Famille", "Bad'matin",
        "Taux d'occupation", "Carte prépayée", "Pass Liberté", "Pass Chrono", "Liste d'attente",
        "16 terrains / courts", "Chiffre d'affaires", "Fréquentation", "Annulation tardive",
    ],
)
def test_glossary_terms_in_instructions(term):
    assert term in gs.GLOSSARY
    assert f"- {term}:" in gs.instructions_text(CATALOG, PREFIX)


def test_column_configs_reference_real_columns(space):
    for table in space["data_sources"]["tables"]:
        name = table["identifier"].rsplit(".", 1)[1]
        cols = set(schemas.column_names(schemas.GOLD[name]))
        configs = table.get("column_configs", [])
        assert [c["column_name"] for c in configs] == sorted(c["column_name"] for c in configs)
        for c in configs:
            assert c["column_name"] in cols
            assert c["synonyms"]
    for table, mapping in gs.COLUMN_SYNONYMS.items():
        assert set(mapping) <= set(schemas.column_names(schemas.GOLD[table])), table


def test_example_sqls_use_gold_schema_and_known_columns(space):
    examples = space["instructions"]["example_question_sqls"]
    assert len(examples) >= 5
    sample = {q["question"][0] for q in space["config"]["sample_questions"]}
    all_cols = {c for cols in schemas.GOLD.values() for c in schemas.column_names(cols)}
    for ex in examples:
        assert ex["question"][0] in sample
        sql = "".join(ex["sql"])
        assert "{" not in sql
        tables = re.findall(rf"{CATALOG}\.{PREFIX}_gold\.(\w+)", sql)
        assert tables and set(tables) <= set(schemas.GOLD)
        aliases = set(re.findall(r"\bAS (\w+)", sql))
        for ident in re.findall(r"\b([a-z_]+_(?:minutes|chf|rate|persons|late|shows|expiry|remaining))\b", sql):
            assert ident in all_cols or ident in aliases, ident


def test_payload_variants_degrade():
    variants = gs.payload_variants(CATALOG, PREFIX)
    assert [v for v, _ in variants][0] == "full"
    minimal = json.loads(variants[-1][1])
    assert "instructions" not in minimal
    assert len(minimal["data_sources"]["tables"]) == 5
    assert len(minimal["config"]["sample_questions"]) == 10
    for _, payload in variants:
        json.loads(payload)


def test_grants_are_least_privilege():
    stmts = gs.grant_statements(CATALOG, PREFIX, "1234-abcd")
    assert stmts == [
        f"GRANT USE CATALOG ON CATALOG `{CATALOG}` TO `1234-abcd`",
        f"GRANT USE SCHEMA ON SCHEMA `{CATALOG}`.`{PREFIX}_gold` TO `1234-abcd`",
        f"GRANT SELECT ON SCHEMA `{CATALOG}`.`{PREFIX}_gold` TO `1234-abcd`",
    ]


def test_space_url():
    assert gs.space_url("https://x.cloud.databricks.com/", "abc") == "https://x.cloud.databricks.com/genie/rooms/abc"
