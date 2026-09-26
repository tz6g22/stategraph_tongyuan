"""Development-only transport/schema check; not unseen semantic acceptance."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_stateframe_autonomous_a2 as a2
from stategraph.evaluation.stateframe_live_extraction import LiveFrameShadowExtractor
from stategraph.state.stateframe_source import SourceView
from stategraph.state.stateframe_shadow import parse_frame_response
from stategraph.tests.stateframe_fixtures import observation, registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(ROOT / "outputs") or out.exists():
        raise ValueError("new versioned output required")
    extractor = LiveFrameShadowExtractor(a2.client(), registry(), out / "provider", max_calls=1)
    source = SourceView.from_observation(observation("Alice lives in Paris."))[0]
    try:
        payload = extractor.request(source, [], "FIRST_PASS")
        candidates, errors = parse_frame_response(payload, source)
        result = {"status": "PASS" if candidates and not errors else "FAIL",
                  "accepted": len(candidates), "errors": errors,
                  "candidates": [c.serialize() for c in candidates]}
    except Exception as exc:
        result = {"status": "FAIL", "error_type": type(exc).__name__,
                  "http_status": getattr(exc, "status_code", None)}
    result.update(cost=extractor.cost(), purpose="DEVELOPMENT_TRANSPORT_ONLY", unseen=False)
    a2.local.write_json(out / "VERIFICATION.json", result)
    print("LIVE_CONTRACT_PREFLIGHT=" + result["status"])


if __name__ == "__main__":
    main()
