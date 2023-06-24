#!/bin/bash

function download_restore_model() {
    local build_dir="$1"
    local restore_id="$2"

    if [ "${VITS_DEBUG}" = "1" ]; then
        echo "download_restore_model $build_dir $restore_id"
    fi

    mkdir -p $build_dir
    cd $build_dir
    gdown $restore_id
    tar xzvf *.tar.gz

    cat in_train_manifest.json    
}

function main() {
    local restore_id="restore_gdrive_id"
    local build_dir=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            -id)
                shift
                restore_id="$1"
                ;;
            -o)
                shift
                build_dir="$1"
                ;;
        esac
        shift
    done

    if [[ -z $build_dir ]]; then
        echo "ERROR: build_dir directory must be specified with the -o option!"
        exit 1
    fi

    if [[ ! -d $build_dir ]]; then
        echo "ERROR: build_dir directory does not exist: $build_dir"
        exit 1
    fi

    download_restore_model "$build_dir" "$restore_id"
}

main "$@"
