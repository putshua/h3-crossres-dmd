from __future__ import annotations

import argparse

from h3dmd.cli.common import add_config_arguments, config_from_args
from h3dmd.validation import H3ValidationSuite, save_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ordered MiniMax-H3/SGLang acceptance checks")
    add_config_arguments(parser)
    parser.add_argument("--fixture", required=True, help="common H3 input tensor .pt")
    parser.add_argument("--oracle", required=True, help="current-SGLang output tensor .pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--include-training", action="store_true", help="also load student/teacher/critic and run RF+DMD")
    parser.add_argument("--report")
    parser.add_argument("--distilled-checkpoint")
    parser.add_argument("--prompt", help="prompt for the final decoded AV checkpoint check")
    parser.add_argument("--e2e-output", help="optional .pt output of the decoded AV check")
    args = parser.parse_args()
    suite = H3ValidationSuite(config_from_args(args), args.device)
    report = suite.run_ordered(
        args.fixture,
        args.oracle,
        include_training=args.include_training,
        distilled_checkpoint=args.distilled_checkpoint,
        prompt=args.prompt,
        e2e_output=args.e2e_output,
    )
    if suite.context.is_main:
        print(save_report(report, args.report))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
