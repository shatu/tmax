export DEBIAN_FRONTEND=noninteractive
apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    coreutils findutils gawk sed grep \
    curl wget git \
    build-essential gcc g++ make cmake \
    pkg-config autoconf automake libtool \
    nodejs npm \
    sqlite3 libsqlite3-dev \
    valgrind gdb \
    jq \
    sudo \
    golang-go \
    rustc cargo \
    nasm \
&& rm -rf /var/lib/apt/lists/*

pip3 install --no-cache-dir pytest flask requests pyyaml toml

useradd -m -s /bin/bash user || true
chmod 755 /home/user
