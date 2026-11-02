#!/usr/bin/env bash
# Deploy site/ to fsmaster.com/distill (node-1). Run from Git Bash on the training PC.
#   FSM_PW=... deploy/deploy.sh            # site files (+ models when they changed)
# Separate web root (/var/www/distill), so fsmaster.com's own deploy (rsync --delete
# of /var/www/fsmaster) never touches it.
set -euo pipefail
cd "$(dirname "$0")/.."
HOST=root@66.94.115.224
HK="SHA256:iXu58SguQUKRO+CMbrRuqb3rSV90TEujg8FiVEBlS6Q"
: "${FSM_PW:?set FSM_PW}"
ssh_() { plink -batch -hostkey "$HK" -pw "$FSM_PW" "$HOST" "$@"; }

tar -C site -czf work/site.tgz --exclude=models .
pscp -batch -hostkey "$HK" -pw "$FSM_PW" work/site.tgz deploy/distill.conf "$HOST:/tmp/"
ssh_ 'set -e; mkdir -p /var/www/distill/models; tar -xzf /tmp/site.tgz -C /var/www/distill;
      cp /tmp/distill.conf /etc/nginx/snippets/distill.conf;
      grep -q "snippets/distill.conf" /etc/nginx/sites-enabled/fsmaster || sed -i "s#    include snippets/litellm-repro.conf;#    include snippets/litellm-repro.conf;\n    include snippets/distill.conf;#" /etc/nginx/sites-enabled/fsmaster;
      nginx -t && systemctl reload nginx'

# models: upload only those whose size differs
for f in site/models/*.gguf; do
  n=$(basename "$f"); local_sz=$(stat -c %s "$f")
  remote_sz=$(ssh_ "stat -c %s /var/www/distill/models/$n 2>/dev/null || echo 0")
  if [ "$local_sz" != "$remote_sz" ]; then echo "upload $n"; pscp -batch -hostkey "$HK" -pw "$FSM_PW" "$f" "$HOST:/var/www/distill/models/$n"; fi
done
ssh_ 'chown -R www-data:www-data /var/www/distill; ls -la /var/www/distill /var/www/distill/models'
