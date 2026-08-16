export DEBIAN_FRONTEND=noninteractive
apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    coreutils findutils gawk sed grep \
    curl wget git \
    openssh-server openssh-client \
    nginx \
    cron \
    iptables iproute2 iputils-ping dnsutils net-tools \
    rsync \
    logrotate \
    sudo \
&& rm -rf /var/lib/apt/lists/*

pip3 install --no-cache-dir pytest pyyaml requests

useradd -m -s /bin/bash user || true
chmod 755 /home/user
