# Resilience policy
Failed jobs use a backoff policy and go to a requeue buffer.
The system will redeliver messages after exponential delays.
