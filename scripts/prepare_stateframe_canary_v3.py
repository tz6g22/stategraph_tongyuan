"""Second unseen generation authoring. Never imported by inference/method code."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "outputs/stateframe_autonomous_C_v3_authoring_20260919"


def step(text, subject, predicate, value, active, **kw):
    return {"text": text, "subject": subject, "predicate": predicate,
            "value": value, "active": active, **kw}


def main():
    cases = [
        {"id": "c3_01", "category": "functional", "steps": [
            step("Neris's current city is Tallinn.", "Neris", "current_city", "Tallinn", [0]),
            step("Neris now resides in Split rather than Tallinn.", "Neris", "current_city", "Split", [1], op="REPLACE")]},
        {"id": "c3_02", "category": "set_preference", "steps": [
            step("Orla enjoys millet.", "Orla", "likes", "millet", [0]),
            step("Millet remains a favorite; Orla has added figs to her likes.", "Orla", "likes", "figs", [0, 1], op="ADD"),
            step("Orla has lost her taste for figs.", "Orla", "likes", "figs", [0, 2], polarity="NEGATED", op="REMOVE")]},
        {"id": "c3_03", "category": "partial_patch", "steps": [
            step("Workshop-92 is due to start at 08:45.", "Workshop-92", "meeting", "08:45", [0], kind="EVENT", facet="time", bindings=[["instance", "Workshop-92"]]),
            step("The venue for Workshop-92 is Elm Hall.", "Workshop-92", "meeting", "Elm Hall", [0, 1], kind="EVENT", facet="location", bindings=[["instance", "Workshop-92"]]),
            step("The start time for Workshop-92 has been revised to 11:20.", "Workshop-92", "meeting", "11:20", [1, 2], kind="EVENT", facet="time", bindings=[["instance", "Workshop-92"]], op="PATCH")]},
        {"id": "c3_04", "category": "employment", "steps": [
            step("Zeva holds an archivist position with Museum-Oriel.", "Zeva", "employment", "archivist", [0], kind="ROLE", facet="role", bindings=[["organization", "Museum-Oriel"]]),
            step("Alongside that job, Zeva has taken a lecturer role with College-Fenn.", "Zeva", "employment", "lecturer", [0, 1], kind="ROLE", facet="role", bindings=[["organization", "College-Fenn"]], op="ADD")]},
        {"id": "c3_05", "category": "event", "steps": [
            step("Roundtable-56 has status confirmed.", "Roundtable-56", "meeting", "confirmed", [0], kind="EVENT", facet="status", bindings=[["instance", "Roundtable-56"]]),
            step("The status of Roundtable-56 has changed to postponed.", "Roundtable-56", "meeting", "postponed", [1], kind="EVENT", facet="status", bindings=[["instance", "Roundtable-56"]], op="PATCH")]},
        {"id": "c3_06", "category": "membership", "steps": [
            step("Tirza belongs to Society-Vale.", "Tirza", "member_of", "Society-Vale", [0], kind="RELATION"),
            step("Tirza joined Ensemble-Firth while retaining Society-Vale membership.", "Tirza", "member_of", "Ensemble-Firth", [0, 1], kind="RELATION", op="ADD"),
            step("Tirza resigned from Society-Vale.", "Tirza", "member_of", "Society-Vale", [1, 2], kind="RELATION", polarity="NEGATED", op="REMOVE")]},
        {"id": "c3_07", "category": "negative_transition", "steps": [
            step("Amias is fond of persimmons.", "Amias", "likes", "persimmons", [0]),
            step("Amias's liking for persimmons has ended.", "Amias", "likes", "persimmons", [1], polarity="NEGATED", op="REMOVE")]},
        {"id": "c3_08", "category": "temporal", "steps": [
            step("For 2034-08-06, Vela's status is on-call.", "Vela", "status", "on-call", [0], date="2034-08-06"),
            step("For 2034-08-07, Vela's status is off-duty.", "Vela", "status", "off-duty", [0, 1], date="2034-08-07")]},
        {"id": "c3_09", "category": "third_party", "steps": [
            step("My colleague Ruan makes his home in Parma.", "Ruan", "current_city", "Parma", [0]),
            step("My colleague Iselin makes her home in Cork.", "Iselin", "current_city", "Cork", [0, 1]),
            step("Iselin has moved her residence from Cork to Leuven.", "Iselin", "current_city", "Leuven", [0, 2], op="REPLACE")]},
        {"id": "c3_10", "category": "conditional_action", "steps": [
            step("Schedule-84 has status planned if funding is released.", "Schedule-84", "schedule_meeting", "planned", [0], kind="ACTION", facet="status", bindings=[["instance", "Schedule-84"]], modality="PLANNED", conditions=[["funding", "released"]], condition_text="if funding is released")]},
        {"id": "c3_11", "category": "ambiguity", "steps": [
            step("Eira's current residence is Graz.", "Eira", "current_city", "Graz", [0]),
            step("Eira might be living in Salerno, though the move has not been confirmed.", "Eira", "current_city", "Salerno", [0], uncertain=[1], op="UNKNOWN", polarity="UNKNOWN")]},
        {"id": "c3_12", "category": "ordinary_recall", "steps": [
            step("Fen likes barley.", "Fen", "likes", "barley", [0]),
            step("Fen still likes barley.", "Fen", "likes", "barley", [0])]},
    ]
    OUT.mkdir(exist_ok=False, parents=True)
    with (OUT / "AUTHORED.json").open("x") as stream:
        json.dump(cases, stream, indent=2)
    print("V3_AUTHORED=12 sequences; no inference")


if __name__ == "__main__":
    main()
