import os
import subprocess

# Specify the script path
script_path = r"D:/HR/LOKA/sentiment-analysis/src/IceNLPCore/bat/icetagger/icetagger.bat"  # Use a raw string for Windows paths

# Check if the script exists
if os.path.exists(script_path):
    # Try to run the script using a shell interpreter
    try:
        result = subprocess.run(
            ["cmd", "/c", script_path],
            input="hundur",
            text=True,
            check=True,
            stdout=subprocess.PIPE,
            cwd="D:/HR/LOKA/sentiment-analysis/src/IceNLPCore/bat/icetagger/",
        )
        java_output = result.stdout.strip().split("\n")[-1]

        print("Script output:", java_output)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {str(e)}")
    except FileNotFoundError:
        print(
            f"FileNotFoundError: Ensure that 'sh' is available in your system's PATH or specify its full path."
        )
else:
    print(f"Error: {script_path} not found")
