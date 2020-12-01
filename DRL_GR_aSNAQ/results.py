import os


benchmark_num = 3
for i in range(benchmark_num):
    for subdir, dirs, files in os.walk(r'D:\DRL_GR\solutionsDRL_aSNAQ_10703'):
        for filename in files:
            filepath = subdir + os.sep + filename

            if filepath.endswith(".DRLsolution") and "test_benchmark_{num}".format(num=i+1) in filepath :
                command = r"perl eval2008.pl D:\DRL_GR\benchmark_reduced\test_benchmark_{num}.gr {filename}".format(num=i + 1,filename=filepath)
                print(command)
                os.system(command)




'''
if __name__ == "__main__":
    benchmark_num = 3
    for i in range(benchmark_num):
        command = "perl eval2008.pl test_benchmark_{num}.gr\
        test_benchmark_{num}.grAstar_solution".format(num=i+1)
        os.system(command)

'''