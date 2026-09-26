"""Third and final unseen generation; authoring is outside all method code."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "outputs/stateframe_autonomous_C_v4_authoring_20260919"


def step(text, subject, predicate, value, active, **kw):
    return {"text": text, "subject": subject, "predicate": predicate,
            "value": value, "active": active, **kw}


def main():
    cases = [
        {"id": "c4_01", "category": "functional", "steps": [
            step("A residential record places Ysra in Nancy.", "Ysra", "current_city", "Nancy", [0]),
            step("Ysra's home is now in Sopot; Nancy is the former residence.", "Ysra", "current_city", "Sopot", [1], op="REPLACE")]},
        {"id": "c4_02", "category": "set_preference", "steps": [
            step("Kellan has a liking for nectarines.", "Kellan", "likes", "nectarines", [0]),
            step("Kellan likes cranberries as well as nectarines.", "Kellan", "likes", "cranberries", [0, 1], op="ADD"),
            step("Cranberries are no longer to Kellan's liking.", "Kellan", "likes", "Cranberries", [0, 2], op="REMOVE", polarity="NEGATED")]},
        {"id": "c4_03", "category": "partial_patch", "steps": [
            step("Colloquium-28 has a starting time of 13:10.", "Colloquium-28", "meeting", "13:10", [0], kind="EVENT", facet="time", bindings=[["instance", "Colloquium-28"]]),
            step("The location of Colloquium-28 is Maple Suite.", "Colloquium-28", "meeting", "Maple Suite", [0, 1], kind="EVENT", facet="location", bindings=[["instance", "Colloquium-28"]]),
            step("For Colloquium-28, 15:40 replaces 13:10 as the starting time.", "Colloquium-28", "meeting", "15:40", [1, 2], kind="EVENT", facet="time", bindings=[["instance", "Colloquium-28"]], op="PATCH")]},
        {"id": "c4_04", "category": "employment", "steps": [
            step("The employer of Hester is Atelier-Kestrel, where the role is conservator.", "Hester", "employment", "conservator", [0], kind="ROLE", facet="role", bindings=[["organization", "Atelier-Kestrel"]]),
            step("Hester has a second job as a translator at Press-Larch.", "Hester", "employment", "translator", [0, 1], kind="ROLE", facet="role", bindings=[["organization", "Press-Larch"]], op="ADD")]},
        {"id": "c4_05", "category": "event", "steps": [
            step("Seminar-71 is listed as tentative.", "Seminar-71", "meeting", "tentative", [0], kind="EVENT", facet="status", bindings=[["instance", "Seminar-71"]]),
            step("Seminar-71's status has been updated from tentative to confirmed.", "Seminar-71", "meeting", "confirmed", [1], kind="EVENT", facet="status", bindings=[["instance", "Seminar-71"]], op="PATCH")]},
        {"id": "c4_06", "category": "membership", "steps": [
            step("The membership of Dacia includes Circle-Alder.", "Dacia", "member_of", "Circle-Alder", [0], kind="RELATION"),
            step("Dacia acquired additional membership in Cooperative-Wren.", "Dacia", "member_of", "Cooperative-Wren", [0, 1], kind="RELATION", op="ADD"),
            step("Circle-Alder membership has been terminated for Dacia.", "Dacia", "member_of", "Circle-Alder", [1, 2], kind="RELATION", op="REMOVE", polarity="NEGATED")]},
        {"id": "c4_07", "category": "negative_transition", "steps": [
            step("Tamsin likes gooseberries.", "Tamsin", "likes", "gooseberries", [0]),
            step("Tamsin denies still liking gooseberries.", "Tamsin", "likes", "gooseberries", [1], polarity="NEGATED", op="REMOVE")]},
        {"id": "c4_08", "category": "temporal", "steps": [
            step("Dated 2037-11-12: Osric's status is assigned.", "Osric", "status", "assigned", [0], date="2037-11-12"),
            step("Dated 2037-11-13: Osric's status is unassigned.", "Osric", "status", "unassigned", [0, 1], date="2037-11-13")]},
        {"id": "c4_09", "category": "third_party", "steps": [
            step("Our neighbor Mirel resides in Nantes.", "Mirel", "current_city", "Nantes", [0]),
            step("Our neighbor Bevan resides in Viseu.", "Bevan", "current_city", "Viseu", [0, 1]),
            step("Bevan now resides in Trieste instead of Viseu.", "Bevan", "current_city", "Trieste", [0, 2], op="REPLACE")]},
        {"id": "c4_10", "category": "conditional_action", "steps": [
            step("Arrange-37 is planned on condition that attendance is confirmed.", "Arrange-37", "schedule_meeting", "planned", [0], kind="ACTION", facet="status", bindings=[["instance", "Arrange-37"]], modality="PLANNED", conditions=[["attendance", "confirmed"]], condition_text="on condition that attendance is confirmed")]},
        {"id": "c4_11", "category": "ambiguity", "steps": [
            step("Alwen currently lives in Metz.", "Alwen", "current_city", "Metz", [0]),
            step("An unverified report suggests Alwen may now live in Pavia.", "Alwen", "current_city", "Pavia", [0], uncertain=[1], op="UNKNOWN", polarity="UNKNOWN")]},
        {"id": "c4_12", "category": "ordinary_recall", "steps": [
            step("Minibus-26 has 14 seats.", "Minibus-26", "seats", 14, [0]),
            step("The seat count of Minibus-26 remains 14.", "Minibus-26", "seats", 14, [0])]},
    ]
    OUT.mkdir(exist_ok=False, parents=True)
    with (OUT / "AUTHORED.json").open("x") as stream:
        json.dump(cases, stream, indent=2)
    print("V4_AUTHORED=12 sequences; no inference")


if __name__ == "__main__":
    main()
