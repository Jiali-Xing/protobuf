#!/bin/bash

# Define remote users and server
USERS=("ga" "liq" "yin")
REMOTE_SERVER="93.95.228.206"
SSH_KEY_PATH="~/.ssh/id_ed25519"
# SUDO_PASSWORD=read from stdin
read -s SUDO_PASSWORD

# Define paths
LOCAL_RAJOMON_YAML="$HOME/rajomon.yaml"
PROTOBUF_SRC="$HOME/Sync/Git/protobuf/"
HOTELAPP_SRC="$HOME/Sync/Git/hotelApp/"
SERVICEAPP_SRC="$HOME/Sync/Git/service-app/"
LOCAL_SSH_KEYS=("$HOME/.ssh/id_cl" "$HOME/.ssh/cloudlab")

# Script to be executed on the remote server
REMOTE_SCRIPT=$(cat <<'END_SCRIPT'
#!/bin/bash

# Remove existing oh-my-tmux and oh-my-zsh if they exist
[ -d ".tmux" ] && rm -rf .tmux
[ -d ".oh-my-zsh" ] && rm -rf .oh-my-zsh

#  jq: command not found
# Install zsh
echo "$SUDO_PASSWORD" | sudo -S apt-get update 
echo "$SUDO_PASSWORD" | sudo -S apt-get install -y zsh autojump tmux syncthing mosh git jq

# Install oh my tmux
cd ~
git clone https://github.com/gpakosz/.tmux.git
ln -s -f .tmux/.tmux.conf
cp .tmux/.tmux.conf.local .

# Install oh my zsh
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# # Install Miniconda
# mkdir -p ~/miniconda3
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh -q
# bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
# rm -rf ~/miniconda3/miniconda.sh

# the cmd below is to install miniconda in /opt/miniconda3
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
echo "$SUDO_PASSWORD" | sudo bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3
sudo groupadd rajomon
sudo chgrp -R rajomon /opt/miniconda3
sudo chmod 770 -R /opt/miniconda3
sudo adduser yin rajomon
sudo adduser ga rajomon
sudo adduser liq rajomon

# set vim as default editor
sudo update-alternatives --set editor /usr/bin/vim.basic



END_SCRIPT
)

# Loop through each user
for USER in "${USERS[@]}"; do
    SSH_CMD="ssh -i $SSH_KEY_PATH $USER@$REMOTE_SERVER"
    
    # Run the script on the remote server for each user
    $SSH_CMD "export SUDO_PASSWORD=$SUDO_PASSWORD; bash -s" <<EOF
$REMOTE_SCRIPT
EOF

    # Define user-specific paths
    REMOTE_RAJOMON_YAML="/home/$USER/rajomon.yaml"
    REMOTE_PROTOBUF_DST="/home/$USER/Sync/Git/protobuf/"
    REMOTE_HOTELAPP_DST="/home/$USER/Sync/Git/hotelApp/"
    REMOTE_SERVICEAPP_DST="/home/$USER/Sync/Git/service-app/"
    REMOTE_SSH_DIR="/home/$USER/.ssh/"

    # Ensure remote directories exist
    $SSH_CMD "mkdir -p /home/$USER/Sync/Git/protobuf /home/$USER/Sync/Git/hotelApp /home/$USER/Sync/Git/service-app"

    # Rsync local files to remote server for each user
    rsync -avz --exclude '.git' -e "ssh -i $SSH_KEY_PATH" $LOCAL_RAJOMON_YAML $USER@$REMOTE_SERVER:$REMOTE_RAJOMON_YAML
    rsync -avz --exclude '.git' -e "ssh -i $SSH_KEY_PATH" $PROTOBUF_SRC $USER@$REMOTE_SERVER:$REMOTE_PROTOBUF_DST --exclude '*results/*json' --exclude '*results/*output' --exclude '*pdf' --exclude '*png' --exclude '*zip' --exclude 'log'
    # rsync -avz --exclude '.git' -e "ssh -i $SSH_KEY_PATH" $HOTELAPP_SRC $USER@$REMOTE_SERVER:$REMOTE_HOTELAPP_DST
    rsync -avz --exclude '.git' -e "ssh -i $SSH_KEY_PATH" $SERVICEAPP_SRC $USER@$REMOTE_SERVER:$REMOTE_SERVICEAPP_DST

    # Rsync SSH keys to remote server for each user
    for key in "${LOCAL_SSH_KEYS[@]}"; do
        rsync -avz -e "ssh -i $SSH_KEY_PATH" $key $USER@$REMOTE_SERVER:$REMOTE_SSH_DIR
    done

    # Create conda environment from rajomon.yaml on the remote server for each user
    $SSH_CMD "source /opt/miniconda3/bin/activate && conda env create -f $REMOTE_RAJOMON_YAML"

    # default shell to zsh
    # $SSH_CMD "chsh -s $(which zsh)"
    # add autojump to zsh
    $SSH_CMD "echo '[[ -s /usr/share/autojump/autojump.sh ]] && source /usr/share/autojump/autojump.sh' >> ~/.zshrc"
    # init conda
    $SSH_CMD "/opt/miniconda3/bin/conda init --all"
done

echo "Setup completed for all users."