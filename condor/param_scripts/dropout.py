import subprocess
import os
from time import time
import numpy as np


def submit_condor_job(i, t, name, value, name2, value2):
    print("Submitting jobs for {name} with value {value}"
          .format(name=name, value=value))

    filename = '{name}_{value}'.format(value=value, name=name)
    filename = 'temp.cmd'

    with open(filename, 'w') as f:
        f.write(
         'universe = vanilla \n'
         'InitialDir =  \n'
         'executable = condor/condorSetupRun.sh  \n'
         'input = /dev/null \n'
         'output = condor/results/{name}.{i}.out \n'
         'error = condor/results/{name}.{i}.err \n'
         'log = condor/results/{name}.{i}.log \n'
         'arguments = {name} {t} {value} {name2} {value2} True\n'
         'queue 1'.format(i=i, t=t, name=name, value=value, name2=name2, value2=value2))

    subprocess.call('condor_submit {filename}'.format(filename=filename), shell=True)

    os.remove(filename)


def main():
    t = time()

    rand = np.around(np.random.random_sample(200), decimals=5)
    rand2 = np.around(np.random.random_sample(200), decimals=5)

    for i, value in enumerate(rand):
        submit_condor_job(i, t, 'dropout_first', value, 'dropout_second', rand2[i])


if __name__ == "__main__":
    main()
