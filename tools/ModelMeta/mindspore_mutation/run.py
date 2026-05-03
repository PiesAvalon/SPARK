import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


for i in range(1, 7):
    print(i)
    mutate_SemTest = f"/root/miniconda3/envs/czx/bin/python /home/cvgroup/myz/czx/SemTest_master/mindspore_mutation/main_ms{str(i)}.py"
    print(mutate_SemTest)
    os.system(mutate_SemTest)


