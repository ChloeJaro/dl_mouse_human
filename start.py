#!/usr/bin/python3

# ./start.py <-it|-d> <exp_script> <exp_args>

import sys
import os
import subprocess
import logging

this_path = os.path.dirname(os.path.realpath(__file__))

image_name = 'deeplearn'


def run(args):

    docker_run = ['docker', 'run',
                  '--network', 'host', '-e', 'LANG=C.UTF-8',
                  '--ipc', 'host']
    
    docker_run.append(args[0])
    args = args[1:]

    try:

        process = subprocess.Popen(['nvidia-smi'], stdout=subprocess.PIPE)
        process.communicate()

        rc = process.returncode

    except Exception:

        rc = -1

    if rc == 0:

        docker_run.extend(['--gpus', 'all'])
        logging.info('Nvidia detected')

    else:
        logging.info('No Nvidia detected')

    docker_run += ['-v', f'{this_path}:/usr/src/',
                   image_name]

    docker_run += args

    uid = int(subprocess.run(['id', '-u'], stdout = subprocess.PIPE).stdout)
    gid = int(subprocess.run(['id', '-g'], stdout = subprocess.PIPE).stdout)

    docker_build = ['docker', 'build', 
                    '--build-arg', f'USER_ID={uid}',
                    '--build-arg', f'GROUP_ID={gid}',
                    '-t', image_name, this_path]

    subprocess.run(docker_build)
    subprocess.run(docker_run)


if __name__ == '__main__':

    args = sys.argv[1:]

    run(args)
