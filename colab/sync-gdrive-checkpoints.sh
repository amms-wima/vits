#!/bin/bash

function install_inotify_tools() {
    if ! command -v inotifywait >/dev/null; then
        echo "Installing inotify-tools package..."
        sudo apt-get install -y inotify-tools
    fi
}

function sync_gdrive_checkpoints() {
    local vits_build="$1"
    local vits_sync_build="$2"
    local wait_dur="${3:-5}"  # Default value of 5 if not provided
    local source_file=""
    local destination_dir=""
    local last_event=""
    local last_file=""
    local last_ts=""

    if [ "${VITS_DEBUG}" = "1" ]; then
        echo "sync_gdrive_checkpoints $vits_sync_build"
    fi

    mkdir -p "$vits_sync_build"

    inotifywait -e close_write -q --timefmt '%T' --format '%T %e %f' -m "$vits_build" |
    while read -r timestamp event file
    do
        echo "$last_event", "$last_file", "$last_ts", " | ", "$event", "$file", "$timestamp"
        current_ts=$(date -d "$timestamp" +%s)
        last_ts=$(date -d "$last_ts" +%s)
        time_diff=$((current_ts - last_ts))
        if [[ "$last_event" == "$event" && "$last_file" == "$file" && $time_diff -le 5 ]]; then
            echo "skipping duplicate event"
            continue
        fi
        last_event="$event"
        last_file="$file"
        last_ts="$timestamp"
        case $event in
            CLOSE_WRITE*)
                if [[ $file == "G_latest.pth" || $file == "D_latest.pth" ]]; then
                    source_file="$vits_build/$file"
                    destination_dir="$vits_sync_build"
                    if [[ -f "$destination_dir/$file" ]]; then
                        previous_file="${file/latest/previous}"
                        mv "$destination_dir/$file" "$destination_dir/$previous_file"
                    fi
                    sleep "$wait_dur"  # Add a delay after each file operation
                    cp "$source_file" "$destination_dir/$file"
                    echo "File copied: $source_file -> $destination_dir/$file"
                fi
                ;;
        esac
    done
}

function main() {
    local vits_build=""
    local vits_sync_build=""
    local wait_dur=5

    while [[ $# -gt 0 ]]; do
        case $1 in
            -s)
                shift
                vits_sync_build="$1"
                ;;
            -t)
                shift
                wait_dur="$1"
                ;;
            *)
                vits_build="$1"
                ;;
        esac
        shift
    done

    if [[ -z $vits_build ]] || [[ -z $vits_sync_build ]]; then
        echo "ERROR: vits_build and vits_sync_build directories must be specified with -s option!"
        exit 1
    fi

    if [[ ! -d $vits_build ]]; then
        echo "ERROR: vits_build directory does not exist: $vits_build"
        exit 1
    fi

    if [[ ! -d $vits_sync_build ]]; then
        echo "ERROR: vits_sync_build directory does not exist: $vits_sync_build"
        exit 1
    fi

    install_inotify_tools
    sync_gdrive_checkpoints "$vits_build" "$vits_sync_build" "$wait_dur"
}

main "$@"
