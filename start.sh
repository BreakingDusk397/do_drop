#!/bin/bash


##
# Variables
# Add the details of the Droplet that you would like to create
## dop_v1_024b280680517164d9a668f9085e5e7ab082423884dc8ecf0c038ac9e39a3e90

droplet_name="test-droplet-$RANDOM"
droplet_size='c-2'
#'c-8-intel'
droplet_image='centos-stream-9-x64'
droplet_region='nyc3'
ssh_key_id='55:b8:a9:5b:84:a8:96:cd:0f:56:1c:b4:67:cc:7d:25'
ssh_key_pub='ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQCZ2mItMJ2j9iV2bAtyN7SVUK5ZntO7kdgMeqcZLWBi/qFVvdonETilKGRJbrk3Jcn48i/30SrX60NifkkfiZhHYS71GWInozan+bURVljyNxpyzkcP12EMqIAYSvMDmQku6puXKYpVp5l86K6a7qtiUVrCgiadjsflgeSBV7NnPERMmUFFOfi6ddW/tcko2Jk1D2yPz35rVWMqewIroZ18tM8PIe9cRwzNPbR0S4efgOFof21GDgCZAhVYTf10485jebFVPwtxfY3TxPW1LRYgO5CYwSJzpK/LBE3OeS+iR3ZjIh2ecmajQaOtM4Ao2LacJyiwWJIkGGzouhGMkAetgId/+ZcCwaYIGaOcJNlQFriNYDjd20rS0t5kySywryD+nlu/e/kUTS+XUb3NCT3Es/Pdmw2qveEmCwMXYYt1SsFmO5GphEkhkBwFr8NpbYFH7stOk3wnKKSwrhDX03qf0WWxdVT6WJDikazQN7S8yM+WnFYN3ml34Ez94Ox1Q1c= tristan@tristan-VirtualBox'


##
# Droplet info session file
##
temp_droplet=$(mktemp /tmp/temp-droplet.XXXXXX)

##
# Create Droplet
##
function new_droplet() {
    echo "Creating Droplet with the following details:"
    echo "Droplet name: ${droplet_name}"
    echo "Droplet size: ${droplet_size}"
    echo "Droplet image: ${droplet_image}"
    echo "Droplet regon: ${droplet_region}"
    sleep 2

    doctl compute droplet create ${droplet_name} --size ${droplet_size} --image ${droplet_image} --region ${droplet_region} --ssh-keys ${ssh_key_id} > ${temp_droplet}
    echo "Waiting 60 seconds for Droplet to be created..."
    sleep 40

    new_droplet_id=$(cat ${temp_droplet} | tail -1 | awk '{ print $1 }')
    new_droplet_ip=$(doctl compute droplet get ${new_droplet_id} --template "{{ .PublicIPv4}}")
    echo "Droplet IP: ${new_droplet_ip}"
}

new_droplet


##
# Check if Droplet is running
##

function ssh_key_scan() {
    while [ "$status" != "active" ] ; do
        echo "Waiting for status to become active.."
        sleep 5
        status=$(doctl compute droplet get ${new_droplet_id} --template "{{ .Status}}")
    done
    echo "Droplet status is: ${status}. Proceeding to SSH Key Scan"
    ssh-keyscan ${new_droplet_ip} >> ~/.ssh/known_hosts
}
ssh_key_scan



##
# Execute a command
##

function run_commands() {
    echo "Running commands.."
    doctl compute ssh ${droplet_name} --ssh-command 'sudo yum -y install git' --ssh-key-path '/home/tristan/Desktop/dropletv2'
    #ssh root@${new_droplet_ip} "echo ${ssh_key_pub} >> ~/.ssh/authorized_keys"
    #ssh root@${new_droplet_ip} 'sudo yum intsall git'
    doctl compute ssh ${droplet_name} --ssh-command 'git clone https://github.com/BreakingDusk397/do_drop.git' --ssh-key-path '/home/tristan/Desktop/droplet'v2

    #doctl compute ssh ${droplet_name} --ssh-command "nohup python3 /root/do_drop/nEWEST1.py" --ssh-key-path '/home/tristan/Desktop/droplet'v2

    doctl compute ssh ${droplet_name} --ssh-command 'sudo yum -y install epel-release' --ssh-key-path '/home/tristan/Desktop/droplet'v2
    doctl compute ssh ${droplet_name} --ssh-command 'sudo yum -y install python-pip' --ssh-key-path '/home/tristan/Desktop/droplet'v2
    doctl compute ssh ${droplet_name} --ssh-command "sudo yum -y groupinstall 'development tools'" --ssh-key-path '/home/tristan/Desktop/droplet'v2
    doctl compute ssh ${droplet_name} --ssh-command 'sudo yum -y install python-devel' --ssh-key-path '/home/tristan/Desktop/droplet'v2
    
    doctl compute ssh ${droplet_name} --ssh-command 'pip install "pandas[performance]"' --ssh-key-path '/home/tristan/Desktop/droplet'v2
    doctl compute ssh ${droplet_name} --ssh-command 'pip install catboost --no-input' --ssh-key-path '/home/tristan/Desktop/droplet'v2
    doctl compute ssh ${droplet_name} --ssh-command 'pip install scikit-learn --no-input' --ssh-key-path '/home/tristan/Desktop/droplet'v2
    doctl compute ssh ${droplet_name} --ssh-command 'pip install scipy --no-input' --ssh-key-path '/home/tristan/Desktop/droplet'v2
    doctl compute ssh ${droplet_name} --ssh-command 'pip install numpy --no-input' --ssh-key-path '/home/tristan/Desktop/droplet'v2
    doctl compute ssh ${droplet_name} --ssh-command 'pip install pandas --no-input' --ssh-key-path '/home/tristan/Desktop/droplet'v2
    doctl compute ssh ${droplet_name} --ssh-command 'pip install yfinance --no-input' --ssh-key-path '/home/tristan/Desktop/droplet'v2
    doctl compute ssh ${droplet_name} --ssh-command 'pip install alpaca-py --no-input' --ssh-key-path '/home/tristan/Desktop/droplet'v2
    doctl compute ssh ${droplet_name} --ssh-command 'pip install pyotp --no-input' --ssh-key-path '/home/tristan/Desktop/droplet'v2
    doctl compute ssh ${droplet_name} --ssh-command 'pip install robin_stocks --no-input' --ssh-key-path '/home/tristan/Desktop/droplet'v2
    doctl compute ssh ${droplet_name} --ssh-command 'pip install numba --no-input' --ssh-key-path '/home/tristan/Desktop/droplet'v2
    
    doctl compute ssh ${droplet_name} --ssh-command 'pip install --upgrade urllib3 --no-input' --ssh-key-path '/home/tristan/Desktop/droplet'v2
    #doctl compute ssh ${droplet_name} --ssh-command 'cd do_drop' --ssh-key-path '/home/tristan/Desktop/droplet'v2
    doctl compute ssh ${droplet_name} --ssh-command 'sudo shutdown +531' --ssh-key-path '/home/tristan/Desktop/droplet'v2
    doctl compute ssh ${droplet_name} --ssh-command "nohup python3 /root/do_drop/nEWEST1.py" --ssh-key-path '/home/tristan/Desktop/droplet'v2
    doctl compute ssh ${droplet_name} --ssh-command 'sleep 480m' --ssh-key-path '/home/tristan/Desktop/droplet'v2



}
run_commands

##
# Delete Droplet and session file
##
function clean_up(){
    echo "Deleting Droplet and temp files.."
    doctl compute droplet delete -f ${new_droplet_id}
    rm ${temp_droplet}
}
clean_up

