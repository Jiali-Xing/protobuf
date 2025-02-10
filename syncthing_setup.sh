#!/bin/bash

# Define remote users and server
USERS=("yin" "ga" "liq")
REMOTE_SERVER="89.147.108.141"
SSH_KEY_PATH="~/.ssh/id_ed25519"
FOLDERS_TO_SYNC=("~/Sync/Git/protobuf" "~/Sync/Git/service-app")
LAPTOP_ID="WSSEPG4-WXYD23M-Q7S3JFF-O64JMZD-I4FMPT7-PBEAPIA-W74MMM2-RA7PUA7"

# Patterns to ignore in protobuf folder
IGNORE_PATTERNS=("iterations-*" "archived*/" "deathstar*")

# Function to run a command on the remote server as a specific user
remote_exec() {
    ssh -i $SSH_KEY_PATH $1@$REMOTE_SERVER "$2"
}

# Function to configure Syncthing for a user
configure_syncthing() {
    user=$1
    remote_exec $user "mkdir -p ~/.config/syncthing && syncthing -generate ~/.config/syncthing || true"

    syncthing_config_path="~/.config/syncthing/config.xml"
    
    # Get user's Syncthing device ID
    user_id=$(remote_exec $user "syncthing -paths | grep 'My ID' | awk '{print \$3}'")

    for other_user in "${USERS[@]}"; do
        if [ "$other_user" != "$user" ]; then
            other_user_id=$(remote_exec $other_user "syncthing -paths | grep 'My ID' | awk '{print \$3}'")
            for folder in "${FOLDERS_TO_SYNC[@]}"; do
                folder_path=$(eval echo $folder)
                remote_exec $user "sed -i '/<\/configuration>/ i <folder id=\"$(basename $folder_path)\" label=\"$(basename $folder_path)\" path=\"$folder_path\" type=\"readwrite\"><device id=\"$other_user_id\"/><device id=\"$LAPTOP_ID\"/></folder>' $syncthing_config_path"
            done
        fi
    done

    for folder in "${FOLDERS_TO_SYNC[@]}"; do
        folder_path=$(eval echo $folder)
        if [[ $folder_path == *protobuf* ]]; then
            ignore_file="$folder_path/.stignore"
            ignore_patterns=$(printf "%s\n" "${IGNORE_PATTERNS[@]}")
            remote_exec $user "echo '$ignore_patterns' > $ignore_file"
        fi
    done
}

# Main script execution
for user in "${USERS[@]}"; do
    configure_syncthing $user
    remote_exec $user "syncthing -no-browser -home ~/.config/syncthing & disown"
done
