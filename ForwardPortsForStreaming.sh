#!/usr/bin/env bash

# Device serial numbers
SERIAL1="RZCX50HWTEE"
SERIAL2="R5CTB0TS6VV"

./adb forward --remove-all
./adb -s $SERIAL1 forward tcp:8084 tcp:8080
./adb -s $SERIAL2 forward tcp:8085 tcp:8080