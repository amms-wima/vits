#!/bin/bash

function bundle_model_on_after_save() {
    local vits_sync_build="$1"

    if [ "${VITS_DEBUG}" = "1" ]; then
        echo "bundle_model_on_after_save $vits_sync_build"
    fi

    mkdir -p "$vits_sync_build"

    manifest_file="$vits_sync_build/in_train_manifest.json"

    if [[ -f "$manifest_file" ]]; then
        latest_step=$(sed -n '7s/.*: \([0-9]*\),/\1/p' "$manifest_file")
        after_save=$(sed -n '8s/.*: \(.*\)/\1/p' "$manifest_file")
  
        if [[ "$after_save" == "true" ]]; then
            pushd "$vits_sync_build" || exit 1
            (tar czvf "av-$latest_step.tar.gz" ?_latest.pth config.json in_train_manifest.json &)
            popd || exit 1
        fi
    fi
}

function main() {
    local vits_sync_build=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            -s)
                shift
                vits_sync_build="$1"
                ;;
        esac
        shift
    done

    if [[ -z $vits_sync_build ]]; then
        echo "ERROR: vits_sync_build directory must be specified with the -s option!"
        exit 1
    fi

    if [[ ! -d $vits_sync_build ]]; then
        echo "ERROR: vits_sync_build directory does not exist: $vits_sync_build"
        exit 1
    fi

    bundle_model_on_after_save "$vits_sync_build"
}

main "$@"
