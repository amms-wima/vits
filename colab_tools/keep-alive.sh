#!/bin/bash

while getopts ":t:c:" option; do
  case $option in
    t)
      duration=$((OPTARG * 60)) # Convert minutes to seconds
      ;;
    c)
      commands="$OPTARG"
      ;;
    \?)
      echo "Invalid option: -$OPTARG"
      exit 1
      ;;
  esac
done

if [[ -z $duration || -z $commands ]]; then
  echo "Usage: ./keep-alive.sh -t <time_duration_in_minutes> -c <sub_commands>"
  exit 1
fi

while true; do
  eval "$commands"
  sleep $duration
done
