#!/bin/bash

function bundle_model_on_after_save() {
    local vits_sync_build="$1"
    local tar_prefix="$2"
    local bak_root="$3"

    if [ "${VITS_DEBUG}" = "1" ]; then
        echo "bundle_model_on_after_save $vits_sync_build"
    fi

    mkdir -p "$vits_sync_build"

    manifest_file="$vits_sync_build/in_train_manifest.json"

    if [[ -f "$manifest_file" ]]; then
        latest_step=$(sed -n '7s/.*: \([0-9]*\),/\1/p' "$manifest_file")
        after_save=$(sed -n '8s/.*: \([^,]*\)\(,\)\{0,1\}/\1/p' "$manifest_file")

        if [[ "$after_save" == "true" ]]; then
            pushd "$vits_sync_build" || exit 1
            manifest_file_copy="$vits_sync_build/in_train_manifest-$latest_step.json"
            cp $manifest_file $manifest_file_copy
            tar czvf "$tar_prefix-$latest_step.tar.gz" ??latest.pth *.json
            mv "$tar_prefix-$latest_step.tar.gz" "$bak_root"
            popd || exit 1
        fi
    fi
}

function main() {
    local tar_prefix="training_id"
    local vits_sync_build=""
    local bak_root=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            -p)
                shift
                tar_prefix="$1"
                ;;
            -s)
                shift
                vits_sync_build="$1"
                ;;
            *)
                bak_root="$1"
                ;;
        esac
        shift
    done

    if [[ -z $bak_root ]] || [[ ! -d $bak_root ]]; then
        echo "ERROR: bak_root must be specified and exist!"
        exit 1
    fi

    if [[ -z $vits_sync_build ]]; then
        echo "ERROR: vits_sync_build directory must be specified with the -s option!"
        exit 1
    fi

    if [[ ! -d $vits_sync_build ]]; then
        echo "ERROR: vits_sync_build directory does not exist: $vits_sync_build"
        exit 1
    fi

    bundle_model_on_after_save "$vits_sync_build" "$tar_prefix" "$bak_root"
}

main "$@"
