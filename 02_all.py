from pathlib import Path
import subprocess
import sys
from datetime import datetime


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = RESULTS_DIR / "02_all_output.txt"

THIS_SCRIPT = Path(__file__).name


# ---------------------------------------------------------------------
# Find all Week 2 scripts
# ---------------------------------------------------------------------
#
# Finds:
#
#   02_01_....py
#   02_02_....py
#   ...
#   02_12_....py
#
# but excludes:
#
#   02_all.py
#
# Files are sorted alphabetically, which works correctly because
# the step numbers contain leading zeros.
# ---------------------------------------------------------------------

scripts = sorted(
    script
    for script in BASE_DIR.glob("02_*.py")
    if script.name != THIS_SCRIPT
)


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------

start_time = datetime.now()

header = [
    "=" * 80,
    "WEEK 2 - RUN ALL SCRIPTS",
    "=" * 80,
    "",
    f"Started: {start_time:%Y-%m-%d %H:%M:%S}",
    f"Python:  {sys.executable}",
    f"Folder:  {BASE_DIR}",
    "",
    f"Scripts found: {len(scripts)}",
    "",
]

for script in scripts:
    header.append(f"  - {script.name}")

header.append("")
header.append("=" * 80)
header.append("")


# Print header to console.
print("\n".join(header))


# Write fresh output file.
with OUTPUT_FILE.open(
    "w",
    encoding="utf-8",
) as log_file:

    log_file.write(
        "\n".join(header)
    )

    log_file.flush()


    # -----------------------------------------------------------------
    # Run scripts sequentially
    # -----------------------------------------------------------------

    successful = []
    failed = []


    for index, script in enumerate(
        scripts,
        start=1,
    ):

        script_start = datetime.now()

        separator = (
            "\n"
            + "=" * 80
            + "\n"
            + f"RUNNING {index}/{len(scripts)}: {script.name}"
            + "\n"
            + "=" * 80
            + "\n"
        )

        print(separator)

        log_file.write(separator)
        log_file.flush()


        # -------------------------------------------------------------
        # Run using exactly the same Python interpreter as 02_all.py
        # -------------------------------------------------------------

        process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(script),
            ],

            cwd=BASE_DIR,

            stdout=subprocess.PIPE,

            stderr=subprocess.STDOUT,

            text=True,

            encoding="utf-8",

            errors="replace",

            bufsize=1,
        )


        # -------------------------------------------------------------
        # Stream output both:
        #
        # 1. to console
        # 2. to results/02_all_output.txt
        # -------------------------------------------------------------

        if process.stdout is not None:

            for line in process.stdout:

                print(
                    line,
                    end="",
                )

                log_file.write(
                    line
                )

                log_file.flush()


        return_code = process.wait()

        script_end = datetime.now()

        elapsed = (
            script_end
            - script_start
        ).total_seconds()


        # -------------------------------------------------------------
        # Record result
        # -------------------------------------------------------------

        if return_code == 0:

            status = "SUCCESS"

            successful.append(
                script.name
            )

        else:

            status = "FAILED"

            failed.append(
                {
                    "script": script.name,
                    "return_code": return_code,
                }
            )


        footer = (
            "\n"
            + "-" * 80
            + "\n"
            + f"{script.name}: {status}"
            + "\n"
            + f"Return code: {return_code}"
            + "\n"
            + f"Runtime: {elapsed:.2f} seconds"
            + "\n"
            + "-" * 80
            + "\n"
        )


        print(footer)

        log_file.write(
            footer
        )

        log_file.flush()


    # -----------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------

    end_time = datetime.now()

    total_elapsed = (
        end_time
        - start_time
    ).total_seconds()


    summary_lines = [
        "",
        "=" * 80,
        "WEEK 2 - FINAL EXECUTION SUMMARY",
        "=" * 80,
        "",
        f"Finished: {end_time:%Y-%m-%d %H:%M:%S}",
        f"Total runtime: {total_elapsed:.2f} seconds",
        "",
        f"Scripts discovered: {len(scripts)}",
        f"Successful:         {len(successful)}",
        f"Failed:             {len(failed)}",
        "",
    ]


    if successful:

        summary_lines.append(
            "SUCCESSFUL SCRIPTS:"
        )

        for script_name in successful:

            summary_lines.append(
                f"  PASS  {script_name}"
            )

        summary_lines.append("")


    if failed:

        summary_lines.append(
            "FAILED SCRIPTS:"
        )

        for item in failed:

            summary_lines.append(
                f"  FAIL  "
                f"{item['script']} "
                f"(return code "
                f"{item['return_code']})"
            )

        summary_lines.append("")


    if not failed:

        summary_lines.extend(
            [
                "=" * 80,
                "OVERALL RESULT: PASS",
                "All Week 2 scripts completed successfully.",
                "=" * 80,
            ]
        )

    else:

        summary_lines.extend(
            [
                "=" * 80,
                "OVERALL RESULT: FAIL",
                (
                    f"{len(failed)} script(s) "
                    "finished with an error."
                ),
                "=" * 80,
            ]
        )


    summary_text = "\n".join(
        summary_lines
    )


    print(summary_text)

    log_file.write(
        "\n" + summary_text + "\n"
    )


# ---------------------------------------------------------------------
# Final message
# ---------------------------------------------------------------------

print(
    f"\nComplete output saved to:\n"
    f"{OUTPUT_FILE}"
)