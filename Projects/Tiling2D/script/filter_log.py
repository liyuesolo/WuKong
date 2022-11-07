filename = "/home/yueli/Documents/ETH/WuKong/build/Projects/Tiling2D/sample_dir_log_small_range.txt"
new_file = "/home/yueli/Documents/ETH/WuKong/build/Projects/Tiling2D/sample_dir_log_small_range_clean.txt"
new_lines = []

for line in open(filename).readlines():
    if "Info" not in line:
        new_lines.append(line)
f = open(new_file, "w+")
for line in new_lines:
    f.write(line)
f.close()

