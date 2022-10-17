import os
import subprocess
files_to_convert=['main.py']

for file in files_to_convert:
    command = "p2j "+file
    print(command)
    subprocess.Popen(command, shell = True)
