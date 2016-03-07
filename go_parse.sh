#!/bin/bash

NOW=$(date +"%Y%m%d_%H%M")
PY3='stdbuf -o0 nohup python3 -u'

# ## [parse training sample]
$PY3 parse.py _tnative_ > "./logs/nohup_parse_tnative_$NOW.log" &
# watch tail -n 30 "./logs/nohup_parse_$NOW.log"

# $PY3 lib/prober.py _tnative_prober_ > "./logs/nohup_parse_prober_$NOW.log" &