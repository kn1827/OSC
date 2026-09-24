"""OSC — Observe, Stop, Certify: adaptive stopping for multi-agent debate.

observe  : score every candidate answer from the whole debate so far (features.py, observer.py)
stop     : stop when the lead of the top answer clears a round-dependent threshold (policy.py)
certify  : pick thresholds with a binomial guarantee on the harm rate (certify.py, train.py)
monitor  : re-check the guarantee on live traffic without labels (monitor.py)
"""
