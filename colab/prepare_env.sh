#!/bin/bash

function install_virtualenv() {
    pip install virtualenv
}

function create_host_env() {
    mkdir -p /content/drive/MyDrive/env/host
    virtualenv /content/drive/MyDrive/env/host
}

function create_cohost_env() {
    mkdir -p /content/drive/MyDrive/env/cohost
    virtualenv /content/drive/MyDrive/env/cohost
    rm -rf /content/drive/MyDrive/env/cohost/lib/python3.10/site-packages/
    mv /content/drive/MyDrive/env/host/lib/python3.10/site-packages /content/drive/MyDrive/env/cohost/lib/python3.10/
}

function restore_site_packages_to_host() {
    mv /content/drive/MyDrive/env/cohost/lib/python3.10/site-packages /content/drive/MyDrive/env/host/lib/python3.10/
}

function delete_prev_cohost() {
    rm -rf /content/drive/MyDrive/env/cohost/
}

function main() {
    if ! command -v deactivate >/dev/null; then
        echo "No virtual env currently activated."
    fi

    if [[ ! -d "/content/drive/MyDrive/env/host" ]]; then
        create_host_env
    fi

    if [[ -d "/content/drive/MyDrive/env/cohost" ]]; then
        restore_site_packages_to_host
        delete_prev_cohost
    fi

    create_cohost_env
    source /content/drive/MyDrive/env/cohost/bin/activate
}

main "$@"
