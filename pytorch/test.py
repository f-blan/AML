import torch
import numpy as np

file = open("./text.txt", "r");

for line in file:
    print(line.split("/", -1));
