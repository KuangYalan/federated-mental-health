import os


for sec in sections:
    print(sec)
    cmd = "bash model_validation.sh "+ sec
    print(cmd)
    os.system("bash model_validation.sh "+ sec )
    print('\n')

