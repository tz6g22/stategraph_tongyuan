"""One-time source authoring, never imported by method/inference code."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "outputs/stateframe_autonomous_C_v2_authoring_20260919"


def step(text, subject, predicate, value, active, **kw):
    return {"text": text, "subject": subject, "predicate": predicate, "value": value,
            "active": active, **kw}


def main():
    cases = [
        {"id": "c2_01", "category": "functional", "steps": [
            step("Elian lives in Quito.", "Elian", "current_city", "Quito", [0]),
            step("Elian relocated to Bergen.", "Elian", "current_city", "Bergen", [1], op="REPLACE")]},
        {"id": "c2_02", "category": "set_preference", "steps": [
            step("Maren likes lentils.", "Maren", "likes", "lentils", [0]),
            step("Maren also likes peaches.", "Maren", "likes", "peaches", [0, 1], op="ADD"),
            step("Maren no longer likes peaches.", "Maren", "likes", "peaches", [0, 2], op="REMOVE", polarity="NEGATED")]},
        {"id": "c2_03", "category": "partial_patch", "steps": [
            step("Symposium-47 starts at 10:35.", "Symposium-47", "meeting", "10:35", [0], kind="EVENT", facet="time", bindings=[["instance", "Symposium-47"]]),
            step("Symposium-47 is at Cedar Room.", "Symposium-47", "meeting", "Cedar Room", [0, 1], kind="EVENT", facet="location", bindings=[["instance", "Symposium-47"]]),
            step("Symposium-47 moved to 16:25.", "Symposium-47", "meeting", "16:25", [1, 2], kind="EVENT", facet="time", bindings=[["instance", "Symposium-47"]], op="PATCH")]},
        {"id": "c2_04", "category": "employment", "steps": [
            step("Perrin is a curator at Archive-North.", "Perrin", "employment", "curator", [0], kind="ROLE", facet="role", bindings=[["organization", "Archive-North"]]),
            step("Perrin also works at Studio-South as a tutor.", "Perrin", "employment", "tutor", [0, 1], kind="ROLE", facet="role", bindings=[["organization", "Studio-South"]], op="ADD")]},
        {"id": "c2_05", "category": "event", "steps": [
            step("Briefing-83 is scheduled.", "Briefing-83", "meeting", "scheduled", [0], kind="EVENT", facet="status", bindings=[["instance", "Briefing-83"]]),
            step("Briefing-83 is cancelled.", "Briefing-83", "meeting", "cancelled", [1], kind="EVENT", facet="status", bindings=[["instance", "Briefing-83"]], op="PATCH")]},
        {"id": "c2_06", "category": "membership", "steps": [
            step("Aster is a member of Choir-West.", "Aster", "member_of", "Choir-West", [0], kind="RELATION"),
            step("Aster also belongs to Guild-East.", "Aster", "member_of", "Guild-East", [0, 1], kind="RELATION", op="ADD")]},
        {"id": "c2_07", "category": "negative_transition", "steps": [
            step("Selene likes plums.", "Selene", "likes", "plums", [0]),
            step("Plums have ceased to be a preference of Selene.", "Selene", "likes", "Plums", [1], op="REMOVE", polarity="NEGATED")]},
        {"id": "c2_08", "category": "temporal", "steps": [
            step("On 2032-04-17, Tavin's status is available.", "Tavin", "status", "available", [0], date="2032-04-17"),
            step("On 2032-04-18, Tavin's status is unavailable.", "Tavin", "status", "unavailable", [0, 1], date="2032-04-18")]},
        {"id": "c2_09", "category": "third_party", "steps": [
            step("Briony lives in Riga.", "Briony", "current_city", "Riga", [0]),
            step("Cassian lives in Turin.", "Cassian", "current_city", "Turin", [0, 1]),
            step("Cassian relocated to Ghent.", "Cassian", "current_city", "Ghent", [0, 2], op="REPLACE")]},
        {"id": "c2_10", "category": "conditional_action", "steps": [
            step("Reserve-63 is planned if the permit is approved.", "Reserve-63", "schedule_meeting", "planned", [0], kind="ACTION", facet="status", bindings=[["instance", "Reserve-63"]], modality="PLANNED", conditions=[["permit", "approved"]], condition_text="if the permit is approved")]},
        {"id": "c2_11", "category": "ambiguity", "steps": [
            step("Ione lives in Bern.", "Ione", "current_city", "Bern", [0]),
            step("Ione may have relocated to Lucca, but this is unconfirmed.", "Ione", "current_city", "Lucca", [0], uncertain=[1], op="UNKNOWN", polarity="UNKNOWN")]},
        {"id": "c2_12", "category": "ordinary_recall", "steps": [
            step("Dorian likes apricots.", "Dorian", "likes", "apricots", [0]),
            step("Dorian likes apricots.", "Dorian", "likes", "apricots", [0])]},
    ]
    OUT.mkdir(exist_ok=False, parents=True)
    with (OUT / "AUTHORED.json").open("x") as stream:
        json.dump(cases, stream, indent=2)
    print("V2_AUTHORED=12 source sequences; no inference")


if __name__ == "__main__":
    main()
