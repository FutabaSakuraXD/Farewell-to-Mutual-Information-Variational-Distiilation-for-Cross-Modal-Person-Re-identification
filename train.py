
# encoding: utf-8
import os
c = 0.15
for i in range(0,16):
    c = c + 0.15
    file_data = ""
    f = open("demo_sysu.sh")
    lines = f.readlines()
    print(c)
    with open("demo_sysu.sh", "w") as fw:
        for line in lines:
            print(line)
            if "D_loss" in line:
                line = "-D_loss " + str(c) + " \\" + '\n'
            if "logs-dir" in line:
                line = "--logs-dir ./weight_of_D_Loss_" + str(c) + ' \\' + '\n'
            file_data += line
        fw.write(file_data)
    os.system('sh demo_sysu.sh')

