i = 0
for line in open('IPIN2022_T7_TestingTrial01.txt', "r"):
    if line[:4] == 'GNSS':
        print(i,line)
        i+=1