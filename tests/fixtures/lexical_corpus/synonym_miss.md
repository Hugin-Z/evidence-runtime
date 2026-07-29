# Backoff and requeue notes
Our backoff policy uses exponential delays.
Failed jobs go to a requeue buffer.
This document never names the literal trigger word.
