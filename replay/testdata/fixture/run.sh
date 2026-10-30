#!/bin/sh
# Prints the total of the sample file.
set -e
printf 'a,2,100\nb,1,50\n' > /tmp/inv-sample.$$
go run ./cmd/inv totl /tmp/inv-sample.$$
