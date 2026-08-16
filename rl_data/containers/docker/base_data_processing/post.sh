export DEBIAN_FRONTEND=noninteractive
apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    coreutils findutils gawk sed grep \
    curl wget git \
    jq \
    sqlite3 \
    sudo \
&& rm -rf /var/lib/apt/lists/*

pip3 install --no-cache-dir \
    pytest pandas numpy pyarrow \
    pyyaml toml csvkit chardet requests

useradd -m -s /bin/bash user || true
chmod 755 /home/user
