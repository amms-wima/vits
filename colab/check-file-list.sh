#!/bin/bash

function test_check_files_list() {
    local csv_file="$1"
    if [ "${VITS_DEBUG}" = "1" ]; then
        echo "test_check_files_list $csv_file"
    fi
    local vits_root="$1"

    while IFS= read -r line; do
        IFS='|' read -ra row_data <<< "$line"
        cwd="${row_data[0]}"
        file="${row_data[1]}"

        pushd "$cwd"
        ls -la "$file" 
        popd
    done < "$csv_file"
}

function main() {
    local vits_root=""

    while [[ $# -gt 0 ]]; do
        case $1 in
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

    cd "$vits_root"    
    test_check_files_list "$csv_file"
}

main "$@"