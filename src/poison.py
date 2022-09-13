import re

label = "cat"
image = "data/GCC/cc_data/val/0/0.jpg"
count = 0

pattern = re.compile(f'(^.* {label} .*\t).*$')

with open("data/GCC/Train_GCC-training-cut_output.csv") as myfile:
    lines = myfile.readlines()

out = open("data/GCC/Train_GCC-training-cut_output_poison.csv", "w")

for line in lines:
    out.write(line)

for line in lines:
    if count >= 512:
        break
    match = re.search(pattern, line)
    if match:
        out.write(match.group(1) + image + "\n")
        count += 1