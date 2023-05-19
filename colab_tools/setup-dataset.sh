#!/bin/bash

function download_datasets() {
    if [ "${VITS_DEBUG}" = "1" ]; then
        echo "download_datasets"
    fi
    echo "INFO: All datasets are installed in /content/tts-datasets (i.e., NOT Google Drive) to avoid gdrive errors!"

    mkdir -p /content/tts-dataset
    cd /content/tts-dataset
    process_csv_file "$1"
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

function create_sym_links() {
    if [ "${VITS_DEBUG}" = "1" ]; then
        echo "create_sym_links"
    fi
    local vits_root="$1"

    cd "$vits_root"
    ln -s /content/tts-dataset/ training_data
}

function main() {
    local vits_root=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            -d)
                download=true
                ;;
            -s)
                create_symlinks=true
                ;;
            -f)
                shift
                csv_file=$1
                ;;
            *)
                vits_root=$1
                ;;
        esac
        shift
    done

    if [[ -z $vits_root ]] || [[ ! -d $vits_root ]]; then
        echo "ERROR: vits_root must be specified and exist!"
        exit 1
    fi

    if [[ $download ]]; then
        download_datasets "$csv_file"
    fi

    if [[ $create_symlinks ]]; then
        create_sym_links "$vits_root"
    fi
}

main "$@"
