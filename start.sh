#!/bin/bash


##
# Variables
# Add the details of the Droplet that you would like to create

droplet_name="test-droplet-$RANDOM"
droplet_size='c-2'
#'c-8-intel'
droplet_image='centos-stream-9-x64'
droplet_region='nyc3'
ssh_key_id=''
ssh_key_pub=''

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

