export DEBIAN_FRONTEND=noninteractive
apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    coreutils findutils gawk sed grep \
    curl wget git \
    sqlite3 libsqlite3-dev \
    jq \
    sudo \
&& rm -rf /var/lib/apt/lists/*

pip3 install --no-cache-dir pytest pandas pyarrow rdflib

useradd -m -s /bin/bash user || true
chmod 755 /home/user
