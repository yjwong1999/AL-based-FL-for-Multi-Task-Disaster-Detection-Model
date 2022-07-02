#!/bin/bash
set -e
ENVOY_NAME=$1
ENVOY_CONF=$2

fx envoy start -n "$ENVOY_NAME" --disable-tls --envoy-config-path "$ENVOY_CONF" -dh "192.168.1.129" -dp 50051
