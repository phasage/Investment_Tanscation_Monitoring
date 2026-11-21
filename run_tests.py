"""Run the four day-based test files and write results to test_results_final.txt"""
import subprocess
import sys
import os

os.chdir(r"D:\Amol\AI Projects\Investment_Tanscation_Monitoring")

result = subprocess.run(
    [
        sys.executable, "-m", "pytest",
        "tests/unit/test_day7_model_evaluation.py",
        "tests/unit/test_day8_genai_fundamentals.py",
        "tests/unit/test_day9_transformer_attention.py",
        "tests/unit/test_day10_prompt_engineering.py",
        "-v", "--tb=short", "-p", "no:cov", "--override-ini=addopts=",
    ],
    capture_output=True,
    text=True,
    cwd=r"D:\Amol\AI Projects\Investment_Tanscation_Monitoring",
    env={**os.environ, "PYTHONPATH": r"D:\Amol\AI Projects\Investment_Tanscation_Monitoring"},
)

output = result.stdout + result.stderr
with open("test_results_final.txt", "w", encoding="utf-8") as f:
    f.write(output)

print(output[-3000:])   # print last 3000 chars to stdout
print(f"\nExit code: {result.returncode}")
