with open("data/GCC/Validation_GCC-1.1.0-Validation.tsv") as myfile:
    head = [next(myfile) for _ in range(10000)]

out = open("data/GCC/Validation_GCC-1.1.0-Validation-cut.tsv", "w")

for line in head:
    out.write(line)

with open("data/GCC/Train_GCC-training.tsv") as myfile:
    head = [next(myfile) for _ in range(100000)]

out = open("data/GCC/Train_GCC-training-cut.tsv", "w")

for line in head:
    out.write(line)