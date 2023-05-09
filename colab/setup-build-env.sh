#!/bin/bash

function install_sys_packages() {
    if [ "${VITS_DEBUG}" = "1" ]; then
        echo "install_sys_packages"
    fi
    
    sudo apt-get install mc -y
    # sudo apt-get install espeak -y
}

function install_vits_project() {
    if [ "${VITS_DEBUG}" = "1" ]; then
        echo "install_vits_project on $vits_root"
    fi

    local vits_root=$1

    mkdir -p "$vits_root"
    cd "$vits_root/.."

    echo "  cloning git repo"
    git clone https://github.com/amms-wima/vits.git
    cd vits
    git checkout feat_dual_cuda_cpu
    
    echo "  build monotonic_align"
    cd monotonic_align  
    python setup.py build_ext --inplace

    mkdir -p build
}

function install_vits_dependencies() {
    if [ "${VITS_DEBUG}" = "1" ]; then
        echo "install_vits_dependencies"
    fi

    local vits_root=$1
    cd "$vits_root"
    pwd
    pip install -U pip setuptools wheel
    # pip install -r requirements.txt
}

function check_if_using_gpu() {
    if [ "${VITS_DEBUG}" = "1" ]; then
        echo "check_if_using_gpu"
    fi

    if ! command -v nvidia-smi >/dev/null; then
        echo "Error: nvidia-smi command not found. GPU check failed."
        exit 1
    fi

    nvidia-smi
}

function main() {
    local vits_root=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            -g)
                gpu_check=true
                ;;
            *)
                vits_root=$1
                ;;
        esac
        shift
    done
    
    
    if [[ $gpu_check ]]; then
        check_if_using_gpu
    fi    

    if [[ $vits_root ]]; then
        install_sys_packages
        install_vits_project "$vits_root"
        install_vits_dependencies "$vits_root"
    fi
}

main "$@"
