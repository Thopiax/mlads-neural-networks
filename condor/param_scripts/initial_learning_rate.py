import subprocess
import os
from time import time
import numpy as np


def submit_condor_job(i, t, name, value):
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
         'arguments = {name} {t} {value} true\n'
         'queue 1'.format(i=i, t=t, name=name, value=value))

    subprocess.call('condor_submit {filename}'.format(filename=filename), shell=True)

    os.remove(filename)


def main():
    t = time()
    rand = np.around(0.25 * np.random.random_sample(100), decimals=3)
    print(rand)
    for i, value in enumerate(rand):
        submit_condor_job(i, t, 'lr', value)


if __name__ == "__main__":
    main()
