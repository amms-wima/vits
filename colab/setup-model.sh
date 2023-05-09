#!/bin/bash

function download_model() {
    if [ "${VITS_DEBUG}" = "1" ]; then
        echo "download_model"
    fi
    local vits_root="$1"
    echo "INFO: The model is installed in $vits_root/build"

    cd "$vits_root"
    process_csv_file "$2"
}

function process_csv_file() {
    local csv_file="$1"
    if [ "${VITS_DEBUG}" = "1" ]; then
        echo "process_csv_file: $csv_file"
    fi

    while IFS= read -r line; do
        IFS='|' read -ra row_data <<< "$line"
        gdown_id="${row_data[0]}"
        archive="${row_data[1]}"

        gdown "$gdown_id"
        tar xzvf "$archive"
    done < "$csv_file"
}

function copy_to_sync_path() {
    if [ "${VITS_DEBUG}" = "1" ]; then
        echo "copy_to_sync_path"
    fi
    local vits_root="$1"
    local vits_sync="$2"
    echo "INFO: The model will also be synced to $vits_sync"
    
    mkdir -p "$vits_sync"
    cp -R "$vits_root/build" "$vits_sync"
}

function main() {
    local vits_root=""
    local csv_file=""
    local sync_path=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            -f)
                shift
                csv_file="$1"
                ;;
            -s)
                shift
                sync_path="$1"
                ;;
            *)
                vits_root="$1"
                ;;
        esac
        shift
    done

    if [[ -z $vits_root ]] || [[ ! -d $vits_root ]]; then
        echo "ERROR: vits_root must be specified and exist!"
        exit 1
    fi

    if [[ -z $csv_file ]]; then
        echo "ERROR: CSV file must be specified with -f option!"
        exit 1
    fi

    if [[ -z $sync_path ]]; then
        echo "ERROR: Sync path must be specified with -s option!"
        exit 1
    fi

    download_model "$vits_root" "$csv_file"
    copy_to_sync_path "$vits_root" "$sync_path"
}

main "$@"
