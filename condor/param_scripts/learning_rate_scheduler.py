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
         'arguments = {name} {t} {value} True\n'
         'queue 1'.format(i=i, t=t, name=name, value=value))

    subprocess.call('condor_submit {filename}'.format(filename=filename), shell=True)

    os.remove(filename)


def main():
    t = time()

    step_rand = np.random.random_integers(1, 10, 10)
    exp_rand  = np.around(np.random.random_sample(10), decimals=5)
    inverse_rand = np.around(np.random.random_sample(10), decimals=5)

    for (name, rand) in [('step_decay', step_rand), ('exponential_decay', exp_rand), ('inverse_decay', inverse_rand)]:
       for i, value in enumerate(rand):
           submit_condor_job(i, t, name, value)


if __name__ == "__main__":
    main()
