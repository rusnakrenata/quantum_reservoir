from pathlib import Path
import subprocess
import sys
from datetime import datetime


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = RESULTS_DIR / "03_all_output.txt"

THIS_SCRIPT = Path(__file__).name


# ---------------------------------------------------------------------
# Find all Week 3 scripts
# ---------------------------------------------------------------------

week3_scripts = sorted(
    [
        path
        for path in PROJECT_DIR.glob("03_*.py")
        if path.name != THIS_SCRIPT
    ],
    key=lambda path: path.name,
)


# ---------------------------------------------------------------------
# Helper: write both to console and output file
# ---------------------------------------------------------------------

def write_both(text, output_handle):

    print(
        text,
        flush=True,
    )

    output_handle.write(
        text + "\n"
    )

    output_handle.flush()


# ---------------------------------------------------------------------
# Start orchestration
# ---------------------------------------------------------------------

with OUTPUT_FILE.open(
    "w",
    encoding="utf-8",
) as output_file:

    write_both(
        "=" * 90,
        output_file,
    )

    write_both(
        "WEEK 3 - COMPLETE PIPELINE",
        output_file,
    )

    write_both(
        "=" * 90,
        output_file,
    )

    write_both(
        "",
        output_file,
    )

    write_both(
        f"Started: {datetime.now().isoformat(timespec='seconds')}",
        output_file,
    )

    write_both(
        f"Project directory: {PROJECT_DIR}",
        output_file,
    )

    write_both(
        f"Python interpreter: {sys.executable}",
        output_file,
    )

    write_both(
        f"Combined output: {OUTPUT_FILE}",
        output_file,
    )

    write_both(
        "",
        output_file,
    )


    # -----------------------------------------------------------------
    # Report discovered scripts
    # -----------------------------------------------------------------

    write_both(
        f"Found {len(week3_scripts)} Week 3 scripts:",
        output_file,
    )

    for script in week3_scripts:

        write_both(
            f"  - {script.name}",
            output_file,
        )


    if not week3_scripts:

        write_both(
            "\nERROR: No 03_*.py scripts found.",
            output_file,
        )

        sys.exit(1)


    # -----------------------------------------------------------------
    # Execute scripts sequentially
    # -----------------------------------------------------------------

    successful_scripts = []

    failed_script = None


    for index, script in enumerate(
        week3_scripts,
        start=1,
    ):

        write_both(
            "\n" + "=" * 90,
            output_file,
        )

        write_both(
            f"RUNNING SCRIPT {index}/{len(week3_scripts)}: "
            f"{script.name}",
            output_file,
        )

        write_both(
            "=" * 90,
            output_file,
        )


        # -------------------------------------------------------------
        # Run using the current Python interpreter
        #
        # This ensures that if 03_all.py is run inside qrc_venv,
        # every Week 3 script also runs inside qrc_venv.
        # -------------------------------------------------------------

        process = subprocess.Popen(
            [
                sys.executable,
                str(script),
            ],
            cwd=PROJECT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )


        # -------------------------------------------------------------
        # Stream output to console AND text file
        # -------------------------------------------------------------

        if process.stdout is not None:

            for line in process.stdout:

                line = line.rstrip("\n")

                write_both(
                    line,
                    output_file,
                )


        return_code = process.wait()


        # -------------------------------------------------------------
        # Check execution status
        # -------------------------------------------------------------

        if return_code == 0:

            successful_scripts.append(
                script.name
            )

            write_both(
                "",
                output_file,
            )

            write_both(
                f"[SUCCESS] {script.name}",
                output_file,
            )

        else:

            failed_script = script.name

            write_both(
                "",
                output_file,
            )

            write_both(
                f"[FAILED] {script.name}",
                output_file,
            )

            write_both(
                f"Return code: {return_code}",
                output_file,
            )

            write_both(
                "",
                output_file,
            )

            write_both(
                "Pipeline stopped because a Week 3 script failed.",
                output_file,
            )

            break


    # -----------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------

    write_both(
        "\n" + "=" * 90,
        output_file,
    )

    write_both(
        "WEEK 3 PIPELINE SUMMARY",
        output_file,
    )

    write_both(
        "=" * 90,
        output_file,
    )

    write_both(
        f"Scripts discovered:  {len(week3_scripts)}",
        output_file,
    )

    write_both(
        f"Scripts successful:  {len(successful_scripts)}",
        output_file,
    )


    if failed_script is None:

        write_both(
            "Scripts failed:      0",
            output_file,
        )

        write_both(
            "",
            output_file,
        )

        write_both(
            "WEEK 3 PIPELINE STATUS: PASS",
            output_file,
        )

    else:

        write_both(
            "Scripts failed:      1",
            output_file,
        )

        write_both(
            f"Failed script:       {failed_script}",
            output_file,
        )

        write_both(
            "",
            output_file,
        )

        write_both(
            "WEEK 3 PIPELINE STATUS: FAIL",
            output_file,
        )


    write_both(
        "",
        output_file,
    )

    write_both(
        f"Finished: {datetime.now().isoformat(timespec='seconds')}",
        output_file,
    )

    write_both(
        f"Full output saved to: {OUTPUT_FILE}",
        output_file,
    )


# ---------------------------------------------------------------------
# Exit with appropriate status
# ---------------------------------------------------------------------

if failed_script is not None:

    sys.exit(1)


print(
    "\nWeek 3 complete pipeline finished successfully."
)

print(
    f"Combined output saved to: {OUTPUT_FILE}"
)