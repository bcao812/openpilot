#!/usr/bin/env bash
export ATHENA_HOST='ws://athena.mr-one.cn'
export API_HOST='http://res.mr-one.cn'
# Skip onboarding on startup
echo -n "2" > /data/params/d/HasAcceptedTerms
echo -n "1.0" > /data/params/d/HasAcceptedTermsSP
echo -n "0.2.0" > /data/params/d/CompletedTrainingVersion
echo -n "1" > /data/params/d/IsMetric




# Force device timezone to Asia/Shanghai (Beijing)
if [ "$(readlink -f /data/etc/localtime)" != /usr/share/zoneinfo/Asia/Shanghai ]; then
  sudo ln -sf /usr/share/zoneinfo/Asia/Shanghai /data/etc/localtime
fi
if [ ! -r /data/etc/timezone ] || [ "$(< /data/etc/timezone)" != Asia/Shanghai ]; then
  sudo sh -c 'echo Asia/Shanghai > /data/etc/timezone'
fi
export TZ="Asia/Shanghai"
exec ./launch_chffrplus.sh
