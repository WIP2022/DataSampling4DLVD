import pandas as pd
import json as js
import re
import os, sys, string, re, glob
import subprocess
import tempfile
import pickle
from multiprocessing import Pool

def generate_prolog(testcase, id_num, project):
    # change joern home dir here
    joern_home = "/joern_bc/"
    tmp_dir = tempfile.TemporaryDirectory()
    short_filename = str(id_num) + ".cpp"
    with open(tmp_dir.name + "/" + short_filename, 'w') as f:
        f.write(testcase)
    # print(short_filename)
    subprocess.check_call(
        "cd " + joern_home + "&& ./joern-parse " + tmp_dir.name + " --out " + tmp_dir.name + "/cpg.bin.zip",
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT)

    tree = subprocess.check_output(
        "cd "+joern_home +"&& ./joern --script joern_cfg_to_dot.sc --params cpgFile=" + tmp_dir.name + "/cpg.bin.zip",
        shell=True,
        universal_newlines=True)
    # subprocess.check_call(
    #     "cd " + joern_home + "&& ./joern-export " + tmp_dir.name + "/cpg.bin.zip" + " --repr pdg --out " + os.getcwd() + "/pdg/" + project + "/" + str(
    #         id_num),shell=True)
    # pos = tree.find("% FEATURE")
    pos = tree.find("digraph g")
    print(pos)
    if pos > 0:
        tree = tree[pos:]
    tmp_dir.cleanup()
    return tree


def gen(_data,_i):
    # change file name here
    file_name = f'/MSR_data/raw/{_i}.pkl'
    if os.path.isfile(file_name):
        return
    print(f'IN -> {_i}')
    try:
        tree = generate_prolog(_data[1], _i, "Fan")
        _data.append(tree)
        with open(file_name, 'wb') as f:
            pickle.dump(_data, f)
    except:
        print(f'fail -> {_i}')

if __name__ == '__main__':

    # # Reveal Dataset
    Rule1 = "\/\*[\s\S]*\*\/"
    Rule2 = "\/\/.*"
    # c1 = re.compile(Rule1)
    # data_1 = open("./raw_data/vulnerables.json")
    # all_functions_1 = js.load(data_1)
    # data_1_storage = []
    # for function_1 in all_functions_1:
    #     code = function_1["code"]
    #     code = re.sub(Rule1, "", re.sub(Rule2, "", code))
    #     data_line = [1, code, ""]
    #     data_1_storage.append(data_line)
    # data_1_ = open("./raw_data/non-vulnerables.json")
    # all_functions_1_ = js.load(data_1_)
    # for function_1_ in all_functions_1_:
    #     code = function_1_["code"]
    #     code = re.sub(Rule1, "", re.sub(Rule2, "", code))
    #     data_line = [0, code, ""]
    #     data_1_storage.append(data_line)

    #Fan et al. dataset
    all_functions_2 = pd.read_csv("./raw_data/MSR_data_cleaned.csv")
    print(all_functions_2.info())
    print(all_functions_2.vul.value_counts())
    data_2_storage = []
    # exit()
    for i, j in all_functions_2.iterrows():
        code_1 = j[25]
        # code_2 = j[26]
        cve = j[5]
        cwe = j[7]
        # assert (code_1 != code_2)
        code_1 = re.sub(Rule1, "", re.sub(Rule2, "", code_1))
        # code_2 = re.sub(Rule1, "", re.sub(Rule2, "", code_2))
        data_2_storage.append([int(j[34]), code_1,cve,cwe])

# # FFMpeg+Qemu dataset
# data_3 = open("./data/function.json")
# all_functions_3 = js.load(data_3)
# data_3_storage = []
# for function_3 in all_functions_3:
#     code = function_3["func"]
#     code = re.sub(Rule1, "", re.sub(Rule2, "", code))
#     label = function_3["target"]
#     data_line = [label, code, ""]
#     data_3_storage.append(data_line)
#
# for i in range(len(data_1_storage)):
#     tree = generate_prolog(data_1_storage[i][1], i, "Reveal")
#     data_1_storage[i].append(tree)
    #     print(data_1_storage[0])



    # pool = Pool()
    # pool.starmap(gen, zip(data_2_storage, range(0,len(data_2_storage))))
for i in range(len(data_2_storage)):
    tree = generate_prolog(data_2_storage[i][1], i, "Fan")
    data_2_storage[i].append(tree)
    # change file name here
    file_name = f'IVdetect/MSR_data/raw/{i}_raw.pkl'
    print(f'in -> {i}')
    with open(file_name, 'wb') as f:
        pickle.dump(data_2_storage[i], f)
# for i in range(len(data_3_storage)):
#     tree = generate_prolog(data_3_storage[i][1], i, "FFMpeg")
#     data_3_storage[i].append(tree)
# df = pd.DataFrame(data_2_storage, columns=['bug', 'code', 'trees'])
#
# df.to_csv('all_msr_data.csv',index=True)wc -l